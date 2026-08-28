import atexit
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if TYPE_CHECKING:
    from cursor_sdk import Client as CursorSdkClient

_CURSOR_SDK_CLIENT: Optional["CursorSdkClient"] = None
_CURSOR_SDK_CLIENT_LOCK = threading.Lock()


def _launch_cursor_bridge_client() -> "CursorSdkClient":
    from cursor_sdk import Bridge, BridgeEndpoint, Client
    from cursor_sdk._bridge import parse_discovery_line
    from cursor_sdk._vendor import resolve_bridge_path

    process = subprocess.Popen(
        [resolve_bridge_path(), "--workspace", str(PROJECT_ROOT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    discovery_payload: dict = {}
    ready = threading.Event()
    errors: list[str] = []

    def _read_discovery_stderr() -> None:
        try:
            if process.stderr is None:
                errors.append("bridge stderr unavailable")
                return
            for line in process.stderr:
                parsed = parse_discovery_line(line)
                if parsed is not None:
                    discovery_payload["value"] = parsed
                    return
            if process.poll() is not None:
                errors.append(f"bridge exited early with code {process.poll()}")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            ready.set()

    threading.Thread(target=_read_discovery_stderr, daemon=True).start()
    if not ready.wait(timeout=30):
        process.terminate()
        raise RuntimeError("Timed out waiting for cursor-sdk-bridge")
    if "value" not in discovery_payload:
        process.terminate()
        detail = f": {'; '.join(errors)}" if errors else ""
        raise RuntimeError(f"Failed to start cursor-sdk-bridge{detail}")

    endpoint = BridgeEndpoint.from_discovery(discovery_payload["value"])
    bridge = Bridge(endpoint, process)
    client = Client(endpoint, allow_api_key_env_fallback=True)
    client._owned_bridge = bridge
    return client


def _get_cursor_sdk_client() -> "CursorSdkClient":
    global _CURSOR_SDK_CLIENT
    with _CURSOR_SDK_CLIENT_LOCK:
        if _CURSOR_SDK_CLIENT is not None:
            return _CURSOR_SDK_CLIENT
        from cursor_sdk import Client

        bridge_url = os.environ.get("CURSOR_SDK_BRIDGE_URL")
        bridge_token = os.environ.get("CURSOR_SDK_BRIDGE_TOKEN") or os.environ.get(
            "CURSOR_SDK_BRIDGE_AUTH_TOKEN"
        )
        if bridge_url and bridge_token:
            _CURSOR_SDK_CLIENT = Client.connect(
                bridge_url, bridge_token, allow_api_key_env_fallback=True
            )
        else:
            _CURSOR_SDK_CLIENT = _launch_cursor_bridge_client()
        return _CURSOR_SDK_CLIENT


def _close_cursor_sdk_client() -> None:
    global _CURSOR_SDK_CLIENT
    with _CURSOR_SDK_CLIENT_LOCK:
        if _CURSOR_SDK_CLIENT is not None:
            _CURSOR_SDK_CLIENT.close()
            _CURSOR_SDK_CLIENT = None


atexit.register(_close_cursor_sdk_client)


class ModelClient:
    """
    Unified model interface. Swap providers by changing provider + model_name.
    Supported: groq / openai / anthropic / gemini / deepseek / mlx_local / mock / cursor_sdk

    mlx_local targets a local `mlx_lm.server` instance (OpenAI-compatible REST
    API) for self-hosted models, e.g. Qwen/Llama running via MLX on Apple
    Silicon. Pass `base_url` (e.g. "http://localhost:8081/v1"); no API key
    needed.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        base_url: Optional[str] = None,
        prompt_cache: bool = True,
        api_key_env: Optional[str] = None,
    ):
        if provider == "cursor":
            provider = "cursor_sdk"
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        # api_key_env lets a config block point at a custom env var (e.g. a
        # relay auth token stored under AICODE_AUTH_TOKEN) instead of the
        # provider-default variable name. Falls back to the provider default
        # when unset, so plain configs keep working.
        env_var = api_key_env or (
            None if self.provider in ("cursor_sdk", "mlx_local", "mock") else f"{provider.upper()}_API_KEY"
        )
        if self.provider == "cursor_sdk":
            self.api_key = api_key or os.environ.get("CURSOR_API_KEY")
        elif self.provider == "mlx_local":
            self.api_key = api_key or "not-needed"
        else:
            self.api_key = api_key or (os.environ.get(env_var) if env_var else None)
        self.default_temperature = temperature
        # Anthropic only: mark the system prompt with cache_control so the stable
        # prefix (rules + case record) is cached across the many calls of one
        # consultation. Ignored server-side below the minimum cacheable length,
        # so safe to leave on.
        self.prompt_cache = prompt_cache
        self._usage_lock = threading.Lock()
        self.usage_totals: Dict[str, float] = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "latency_s": 0.0,
        }
        self._client = self._init_client()

    def _record_usage(self, meta: Dict) -> None:
        with self._usage_lock:
            self.usage_totals["calls"] += 1
            for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
                if meta.get(key):
                    self.usage_totals[key] += meta[key]
            self.usage_totals["latency_s"] += meta.get("latency_s") or 0.0

    def _init_client(self):
        if self.provider == "mock":
            return None
        if self.provider == "cursor_sdk":
            try:
                import cursor_sdk  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "cursor-sdk is not installed. Run: pip install cursor-sdk"
                ) from e
            return None
        if self.provider == "groq":
            from groq import Groq
            return Groq(api_key=self.api_key)
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        if self.provider == "deepseek":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        if self.provider == "mlx_local":
            from openai import OpenAI
            if not self.base_url:
                raise ValueError("mlx_local provider requires base_url, e.g. http://localhost:8081/v1")
            # 4-minute per-request timeout: a healthy local decode of a
            # 10-turn dialogue prompt on a 4-8B MLX model finishes in <60s.
            # Any request that hangs longer is almost certainly a server-side
            # exception that left the socket dangling — fail fast so the
            # retry loop (or the pipeline) can move on instead of blocking
            # for the client's 10-minute default.
            return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=240.0)
        if self.provider == "anthropic":
            import anthropic
            # base_url lets us point the Anthropic SDK at a compatible relay
            # (e.g. api.aicode007.com) that fronts non-Anthropic models under
            # the same wire protocol; leave unset for direct api.anthropic.com.
            kwargs: Dict[str, object] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return anthropic.Anthropic(**kwargs)
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model_name)
        raise ValueError(f"Unknown provider: {self.provider}")

    def chat(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """
        messages: [{"role": "user"/"assistant", "content": "..."}]
        Returns the model's reply as a string. Retries on rate-limit errors.
        """
        text, _ = self.chat_ex(
            messages, system_prompt=system_prompt, temperature=temperature, max_retries=max_retries
        )
        return text

    def chat_ex(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ):
        """Like chat(), but returns (text, meta) where meta carries
        {"input_tokens", "output_tokens", "cache_read_tokens", "latency_s"}
        (None where the provider does not report a value)."""
        return self._chat_ex_with_retry(
            messages, system_prompt=system_prompt, temperature=temperature, max_retries=max_retries
        )

    def _format_cursor_prompt(
        self,
        messages: List[Dict],
        system_prompt: Optional[str],
        temperature: float,
    ) -> str:
        role_label = {"user": "User", "assistant": "Assistant"}
        conv_lines = []
        for msg in messages:
            label = role_label.get(msg["role"], msg["role"].title())
            conv_lines.append(f"{label}: {msg['content']}")

        return (
            "You are a plain text completion backend for a clinical dialogue simulator.\n"
            "Reply with ONLY the next assistant message text.\n"
            "Do NOT use tools, do NOT read files, do NOT add markdown fences or explanations.\n"
            f"Temperature hint: {temperature} (follow style only; be concise).\n"
            "\n"
            "[System instructions]\n"
            f"{system_prompt or '(none)'}\n"
            "\n"
            "[Conversation so far]\n"
            + "\n".join(conv_lines)
            + "\n"
            "\n"
            "[Your reply — assistant message only]"
        )

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        text = text.strip()
        if not text.startswith("```"):
            return text
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _empty_meta() -> Dict:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "latency_s": None,
        }

    def _chat_once(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        temp = temperature if temperature is not None else self.default_temperature
        meta = self._empty_meta()
        started = time.monotonic()

        if self.provider == "mock":
            meta["latency_s"] = 0.0
            return "[MOCK] This is a placeholder response.", meta

        if self.provider == "cursor_sdk":
            from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

            prompt = self._format_cursor_prompt(messages, system_prompt, temp)
            result = Agent.prompt(
                prompt,
                AgentOptions(
                    api_key=self.api_key,
                    model=self.model_name,
                    local=LocalAgentOptions(cwd=str(PROJECT_ROOT)),
                ),
                client=_get_cursor_sdk_client(),
            )
            if result.status == "error":
                raise RuntimeError(f"Cursor SDK run failed (run_id={result.id})")
            text = self._strip_markdown_fence(result.result or "")
            if not text:
                raise RuntimeError("Cursor SDK returned empty result")
            meta["latency_s"] = time.monotonic() - started
            return text, meta

        if self.provider in ("groq", "openai", "deepseek", "mlx_local"):
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            kwargs = {
                "model": self.model_name,
                "messages": full_messages,
                "temperature": temp,
                # Thinking models (Qwen3.5 on mlx_lm.server, o-series, ...) spend
                # completion budget on reasoning before the reply; a small server
                # default can leave content=None once reasoning eats it all.
                "max_tokens": 4096,
            }
            # Parameter-compatibility fallback loop: reasoning models (gpt-5*,
            # o-series) reject non-default temperature AND reject max_tokens in
            # favor of max_completion_tokens — and report only one violation per
            # request, so both may need fixing across successive attempts.
            for _ in range(3):
                try:
                    response = self._client.chat.completions.create(**kwargs)
                    break
                except Exception as e:
                    err = str(e).lower()
                    unsupported = "does not support" in err or "unsupported" in err or "not supported" in err
                    if unsupported and "temperature" in err and "temperature" in kwargs:
                        kwargs.pop("temperature")
                    elif unsupported and "max_tokens" in err and "max_tokens" in kwargs:
                        kwargs.pop("max_tokens")
                        # Reasoning tokens count against this budget; give headroom.
                        kwargs["max_completion_tokens"] = 16384
                    else:
                        raise
            content = response.choices[0].message.content
            if content is None or not content.strip():
                finish = getattr(response.choices[0], "finish_reason", None)
                raise RuntimeError(
                    f"{self.provider}/{self.model_name} returned empty content "
                    f"(finish_reason={finish})"
                )
            usage = getattr(response, "usage", None)
            if usage is not None:
                meta["input_tokens"] = getattr(usage, "prompt_tokens", None)
                meta["output_tokens"] = getattr(usage, "completion_tokens", None)
                details = getattr(usage, "prompt_tokens_details", None)
                if details is not None:
                    meta["cache_read_tokens"] = getattr(details, "cached_tokens", None)
            meta["latency_s"] = time.monotonic() - started
            return content, meta

        if self.provider == "anthropic":
            if system_prompt and self.prompt_cache:
                system: object = [
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                system = system_prompt or ""
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system,
                messages=messages,
                temperature=temp,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                meta["input_tokens"] = getattr(usage, "input_tokens", None)
                meta["output_tokens"] = getattr(usage, "output_tokens", None)
                meta["cache_read_tokens"] = getattr(usage, "cache_read_input_tokens", None)
            meta["latency_s"] = time.monotonic() - started
            # Some relay-fronted models (e.g. gpt-5-family via aicode007) emit
            # a ThinkingBlock before the TextBlock; only text blocks carry a
            # .text attribute. Concatenate every text-block segment, in order.
            text_parts = [
                getattr(block, "text", None)
                for block in (response.content or [])
                if getattr(block, "type", None) == "text"
            ]
            text = "".join(p for p in text_parts if p)
            if not text.strip():
                stop_reason = getattr(response, "stop_reason", None)
                raise RuntimeError(
                    f"anthropic/{self.model_name} returned no text blocks "
                    f"(stop_reason={stop_reason})"
                )
            return text, meta

        if self.provider == "gemini":
            chat = self._client.start_chat()
            if system_prompt:
                chat.send_message(system_prompt)
            for msg in messages[:-1]:
                chat.send_message(msg["content"])
            response = chat.send_message(messages[-1]["content"])
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                # Last send only; earlier sends in the same call are not counted.
                meta["input_tokens"] = getattr(usage, "prompt_token_count", None)
                meta["output_tokens"] = getattr(usage, "candidates_token_count", None)
            meta["latency_s"] = time.monotonic() - started
            return response.text, meta

        raise ValueError(f"Unknown provider: {self.provider}")

    def chat_with_retry(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        text, _ = self._chat_ex_with_retry(
            messages, system_prompt=system_prompt, temperature=temperature, max_retries=max_retries
        )
        return text

    def _chat_ex_with_retry(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ):
        for attempt in range(max_retries):
            try:
                text, meta = self._chat_once(messages, system_prompt, temperature)
                self._record_usage(meta)
                return text, meta
            except Exception as e:
                retryable = False
                try:
                    from cursor_sdk import CursorAgentError, RateLimitError

                    if isinstance(e, RateLimitError):
                        retryable = True
                    elif isinstance(e, CursorAgentError) and getattr(e, "is_retryable", False):
                        retryable = True
                except ImportError:
                    pass

                if not retryable:
                    err = str(e).lower()
                    retryable = any(
                        k in err
                        for k in (
                            "rate_limit", "rate limit", "429", "too many",
                            "overloaded", "529", "500", "502", "503",
                            "timeout", "timed out", "connection",
                            "empty content",
                        )
                    )

                if retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Transient API error ({type(e).__name__}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
