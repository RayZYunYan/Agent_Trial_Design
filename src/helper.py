import logging
import os
from keys import mykey

# A dictionary to cache models and tokenizers to avoid reloading

global models
models = {}

_API_BACKENDS = ("openai", "groq", "anthropic")


def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger: 
        if mode == "error": logger.error(message)
        if mode == "warning": logger.warning(message)
        else: logger.info(message)
    if print_to_std: print(message + "\n")


def infer_use_api(model_name, use_api=None):
    """Pick API backend from model id so expert/patient can differ in one run."""
    name = (model_name or "").lower()
    if "claude" in name:
        return "anthropic"
    if (
        "gpt" in name
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    ):
        return "openai"
    if use_api in _API_BACKENDS:
        return use_api
    return use_api


class ModelCache:
    def __init__(self, model_name, use_vllm=False, use_api=None, use_mlx=False, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_mlx = bool(use_mlx) and use_api not in _API_BACKENDS
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        if self.use_api == "openai":
            from openai import OpenAI
            self.api_account = self.args.get("api_account", "openai")
            key = mykey.get(self.api_account) or mykey.get("openai") or ""
            if not key:
                raise ValueError("OPENAI_API_KEY missing in .env (needed for OpenAI models)")
            self.client = OpenAI(api_key=key)
        elif self.use_api == "groq":
            from groq import Groq
            key = mykey.get("groq") or ""
            if not key:
                raise ValueError("GROQ_API_KEY missing in .env (needed for --use_api groq)")
            self.client = Groq(api_key=key)
        elif self.use_api == "anthropic":
            import anthropic
            key = mykey.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY") or ""
            if not key:
                raise ValueError("ANTHROPIC_API_KEY missing in .env (needed for Claude models)")
            self.client = anthropic.Anthropic(api_key=key)
        elif self.use_mlx:
            try:
                from mlx_lm import load as mlx_load

                log_info(f"[{self.model_name}]: loading with mlx-lm (Apple Silicon)", print_to_std=True)
                self.model, self.tokenizer = mlx_load(self.model_name)
                self.use_vllm = False
            except Exception as e:
                log_info(
                    f"[ERROR] [{self.model_name}]: mlx-lm load failed ({e}); falling back to transformers/MPS.",
                    mode="error",
                    print_to_std=True,
                )
                self.use_mlx = False
                # MLX community ids are not always loadable via transformers; prefer HF original id.
                fallback = self.args.get("hf_fallback_name") or self.args.get("fallback_model_name")
                if fallback and fallback != self.model_name:
                    log_info(
                        f"[{self.model_name}]: switching load id to HF fallback {fallback}",
                        print_to_std=True,
                    )
                    self.model_name = fallback
        if self.use_api in _API_BACKENDS:
            return
        if self.use_mlx:
            return
        if self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", False)
                dtype = self.args.get("dtype", "auto")
                self.model = LLM(
                    model=self.model_name,
                    enable_prefix_caching=enable_prefix_caching,
                    dtype=dtype,
                )
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                self.terminators = [self.tokenizer.eos_token_id]
                if eot is not None and eot != self.tokenizer.unk_token_id:
                    self.terminators.append(eot)
            except Exception as e:
                log_info(f"[ERROR] [{self.model_name}]: If using a custom local model, it is not compatible with VLLM, will load using Huggingfcae and you can ignore this error: {str(e)}", mode="error")
                self.use_vllm = False
        if not self.use_vllm and self.use_api not in _API_BACKENDS:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            trust_remote = bool(self.args.get("trust_remote_code", True))
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=trust_remote
            )
            load_kwargs = {"trust_remote_code": trust_remote}
            load_in_4bit = self.args.get("load_in_4bit", False)
            if isinstance(load_in_4bit, str):
                load_in_4bit = load_in_4bit.strip().lower() in ("1", "true", "yes")
            if not load_in_4bit:
                load_in_4bit = os.environ.get("HF_LOAD_IN_4BIT", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )

            use_mps = torch.backends.mps.is_available()
            # Mac / Apple Silicon: avoid bitsandbytes + device_map quirks; use MPS fp16.
            if use_mps and not load_in_4bit:
                load_kwargs["torch_dtype"] = torch.float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **load_kwargs
                )
                self.model.to("mps")
            else:
                load_kwargs["device_map"] = self.args.get("device_map", "auto")
                load_kwargs["torch_dtype"] = "auto"
                if load_in_4bit:
                    try:
                        from transformers import BitsAndBytesConfig

                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True
                        )
                    except Exception as e:
                        log_info(
                            f"[{self.model_name}]: 4-bit load requested but bitsandbytes unavailable ({e}); loading full precision.",
                            mode="warning",
                        )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **load_kwargs
                )
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            self.terminators = [self.tokenizer.eos_token_id]
            if eot is not None and eot != getattr(self.tokenizer, "unk_token_id", None):
                self.terminators.append(eot)
    
    def generate(self, messages):
        log_info(f"[{self.model_name}][INPUT]: {messages}")

        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_tokens", 256)
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)

        if self.use_api in ("openai", "groq"):
            return self.openai_generate(messages)
        if self.use_api == "anthropic":
            return self.anthropic_generate(messages)
        if self.use_mlx:
            return self.mlx_generate(messages)
        if self.use_vllm:
            return self.vllm_generate(messages)
        return self.huggingface_generate(messages)

    def mlx_generate(self, messages):
        from mlx_lm import generate as mlx_generate

        try:
            prompt = self._apply_chat_template(messages, tokenize=False)
        except Exception:
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            prompt = "\n\n".join([m["content"] for m in messages])
        if not isinstance(prompt, str):
            prompt = self.tokenizer.decode(prompt) if hasattr(self.tokenizer, "decode") else str(prompt)

        temp = float(self.temperature) if self.temperature is not None else 0.6
        gen_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": int(self.max_tokens),
            "verbose": False,
        }
        # Newer mlx-lm: make_sampler(...); older: temp=/top_p= kwargs.
        try:
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=max(temp, 0.0), top_p=float(self.top_p))
            response_text = mlx_generate(**gen_kwargs, sampler=sampler)
        except Exception:
            try:
                if temp <= 0:
                    response_text = mlx_generate(**gen_kwargs, temp=0.0)
                else:
                    response_text = mlx_generate(**gen_kwargs, temp=temp, top_p=float(self.top_p))
            except TypeError:
                response_text = mlx_generate(
                    self.model, self.tokenizer, prompt=prompt, max_tokens=int(self.max_tokens), verbose=False
                )

        response_text = self._strip_thinking(response_text or "")
        usage = {
            "input_tokens": self._token_len(prompt),
            "output_tokens": self._token_len(response_text),
        }
        log_info(f"[{self.model_name}][OUTPUT]: {response_text}")
        return response_text, None, usage

    def _token_len(self, text: str) -> int:
        if not text:
            return 0
        try:
            ids = self.tokenizer.encode(text)
            return len(ids)
        except Exception:
            return 0

    def _input_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return getattr(self.model, "device", "cpu")
    
    def _apply_chat_template(self, messages, *, tokenize):
        """Apply chat template; disable Qwen3.5 default thinking when supported."""
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": tokenize,
        }
        if tokenize:
            kwargs["return_tensors"] = "pt"
        # Qwen3.5 / DeepSeek-R1 (Qwen3) often think by default; MediQ needs direct answers.
        name = (self.model_name or "").lower()
        if any(
            key in name
            for key in (
                "qwen3.5",
                "qwen3_5",
                "deepseek-r1",
                "r1-0528",
                "qwen3-8b",
            )
        ):
            kwargs["enable_thinking"] = False
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            kwargs.pop("chat_template_kwargs", None)
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                raise

    @staticmethod
    def _strip_thinking(text: str) -> str:
        import re

        if not text:
            return text
        # Drop Qwen / DeepSeek-R1 chain-of-thought blocks if still present.
        cleaned = re.sub(
            r"<think>.*?</think>\s*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<thinking>.*?</thinking>\s*",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Truncated R1 outputs often leave an unclosed <think> ...
        cleaned = re.sub(
            r"<think>.*\Z",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return cleaned.strip() or text.strip()

    def huggingface_generate(self, messages):
        device = self._input_device()
        try:
            inputs = self._apply_chat_template(messages, tokenize=True).to(device)
        except Exception:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            prompt = "\n\n".join([m['content'] for m in messages])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.terminators,
        }
        # temperature=0 → greedy (cross-finalize / deterministic)
        if self.temperature is not None and float(self.temperature) <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(float(self.temperature), 1e-5)
            gen_kwargs["top_p"] = self.top_p

        outputs = self.model.generate(inputs, **gen_kwargs)
        # TODO: If top_logprobs > 0, return logprobs of generation
        response_text = self.tokenizer.decode(
            outputs[0][inputs.shape[-1] :], skip_special_tokens=True
        )
        response_text = self._strip_thinking(response_text)
        usage = {"input_tokens": inputs.shape[-1], "output_tokens": outputs.shape[-1]-inputs.shape[-1]}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, None, usage
        
    def vllm_generate(self, messages):
        try:
            inputs = self._apply_chat_template(messages, tokenize=False)
        except Exception:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        temp = float(self.temperature) if self.temperature is not None else 0.6
        if temp <= 0:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.max_tokens,
                top_p=1.0,
                logprobs=self.top_logprobs or None,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
        else:
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                logprobs=self.top_logprobs or None,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
        
        outputs = self.model.generate(inputs, sampling_params)
        response_text = self._strip_thinking(outputs[0].outputs[0].text)
        logprobs = outputs[0].outputs[0].cumulative_logprob
        # TODO: If top_logprobs > 0, return logprobs of generation
        # if self.top_logprobs > 0: logprobs = outputs[0].outputs[0].logprobs
        usage = {"input_tokens": len(outputs[0].prompt_token_ids), "output_tokens": len(outputs[0].outputs[0].token_ids)}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        # GPT-5+ chat models reject max_tokens; use max_completion_tokens instead.
        name = (self.model_name or "").lower()
        if name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_logprobs > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.top_logprobs
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Fallback if temperature/top_p unsupported on reasoning models.
            err = str(e).lower()
            if "temperature" in err or "top_p" in err or "unsupported" in err:
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise

        num_input_tokens = response.usage.prompt_tokens
        num_output_tokens = response.usage.completion_tokens
        response_text = (response.choices[0].message.content or "").strip()
        log_probs = None
        if self.top_logprobs > 0 and response.choices[0].logprobs is not None:
            log_probs = response.choices[0].logprobs.content

        log_info(f"[{self.model_name}][OUTPUT]: {response_text}")
        return response_text, log_probs, {"input_tokens": num_input_tokens, "output_tokens": num_output_tokens}

    def anthropic_generate(self, messages):
        system_parts = []
        converted = []
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                converted.append({"role": role, "content": content})
            else:
                converted.append({"role": "user", "content": content})

        # Anthropic requires the first message to be from the user.
        if converted and converted[0]["role"] != "user":
            converted.insert(0, {"role": "user", "content": "(continue)"})

        kwargs = {
            "model": self.model_name,
            "messages": converted,
            "max_tokens": int(self.max_tokens),
            "temperature": self.temperature,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = self.client.messages.create(**kwargs)
        chunks = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        response_text = "".join(chunks).strip()
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
        log_info(f"[{self.model_name}][OUTPUT]: {response_text}")
        return response_text, None, usage


def get_response(messages, model_name, use_vllm=False, use_api=None, use_mlx=False, **kwargs):
    use_api = infer_use_api(model_name, use_api)
    # API backends ignore local accelerators.
    effective_vllm = bool(use_vllm) and use_api not in _API_BACKENDS
    effective_mlx = bool(use_mlx) and use_api not in _API_BACKENDS and not effective_vllm
    cache_key = f"{use_api}:vllm={effective_vllm}:mlx={effective_mlx}:{model_name}"

    model_cache = models.get(cache_key, None)
    if model_cache is None:
        model_cache = ModelCache(
            model_name,
            use_vllm=effective_vllm,
            use_api=use_api,
            use_mlx=effective_mlx,
            **kwargs,
        )
        models[cache_key] = model_cache

    response_text, log_probs, usage = model_cache.generate(messages)
    # Callers accumulate usage with +=; never return None counts.
    if not isinstance(usage, dict):
        usage = {"input_tokens": 0, "output_tokens": 0}
    else:
        usage = {
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
        }
    return response_text, log_probs, usage
