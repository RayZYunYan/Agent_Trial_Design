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
    ):
        if provider == "cursor":
            provider = "cursor_sdk"
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        if self.provider == "cursor_sdk":
            self.api_key = api_key or os.environ.get("CURSOR_API_KEY")
        elif self.provider == "mlx_local":
            self.api_key = api_key or "not-needed"
        else:
            self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.default_temperature = temperature
        self._client = self._init_client()

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
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
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
        return self.chat_with_retry(
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

    def _chat_once(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        temp = temperature if temperature is not None else self.default_temperature

        if self.provider == "mock":
            return "[MOCK] This is a placeholder response."

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
            return text

        if self.provider in ("groq", "openai", "deepseek", "mlx_local"):
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            kwargs = {
                "model": self.model_name,
                "messages": full_messages,
                "temperature": temp,
            }
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as e:
                # Reasoning models (gpt-5*, o-series) and some newer models
                # (e.g. gpt-5.6 family) only accept the default temperature.
                err = str(e).lower()
                if "temperature" in err and ("does not support" in err or "unsupported" in err):
                    kwargs.pop("temperature", None)
                    response = self._client.chat.completions.create(**kwargs)
                else:
                    raise
            return response.choices[0].message.content

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system_prompt or "",
                messages=messages,
                temperature=temp,
            )
            return response.content[0].text

        if self.provider == "gemini":
            chat = self._client.start_chat()
            if system_prompt:
                chat.send_message(system_prompt)
            for msg in messages[:-1]:
                chat.send_message(msg["content"])
            return chat.send_message(messages[-1]["content"]).text

    def chat_with_retry(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        for attempt in range(max_retries):
            try:
                return self._chat_once(messages, system_prompt, temperature)
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
                    retryable = any(k in err for k in ("rate_limit", "rate limit", "429", "too many"))

                if retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
