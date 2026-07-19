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
    def __init__(self, model_name, use_vllm=False, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
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
        elif self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", False)
                self.model = LLM(model=self.model_name, enable_prefix_caching=enable_prefix_caching)
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            except Exception as e:
                log_info(f"[ERROR] [{self.model_name}]: If using a custom local model, it is not compatible with VLLM, will load using Huggingfcae and you can ignore this error: {str(e)}", mode="error")
                self.use_vllm = False
        if not self.use_vllm and self.use_api not in _API_BACKENDS:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
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
        if self.use_vllm:
            return self.vllm_generate(messages)
        return self.huggingface_generate(messages)
    
    def huggingface_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            prompt = "\n\n".join([m['content'] for m in messages])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            inputs,
            do_sample=True,
            max_new_tokens=self.max_tokens, 
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.terminators
        )
        # TODO: If top_logprobs > 0, return logprobs of generation
        response_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        usage = {"input_tokens": inputs.shape[-1], "output_tokens": outputs.shape[-1]-inputs.shape[-1]}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, None, usage
        
    def vllm_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, logprobs=self.top_logprobs, 
                                        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        outputs = self.model.generate(inputs, sampling_params)
        response_text = outputs[0].outputs[0].text
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
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.top_logprobs > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.top_logprobs
        response = self.client.chat.completions.create(**kwargs)

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


def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    use_api = infer_use_api(model_name, use_api)
    cache_key = f"{use_api}:{model_name}"

    model_cache = models.get(cache_key, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **kwargs)
        models[cache_key] = model_cache
    
    return model_cache.generate(messages)
