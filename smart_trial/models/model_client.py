import os
import time
from typing import List, Dict, Optional


class ModelClient:
    """
    Unified model interface. Swap providers by changing provider + model_name.
    Supported: groq / openai / anthropic / gemini / mock (for testing without API keys)
    """

    def __init__(self, provider: str, model_name: str,
                 api_key: Optional[str] = None, temperature: float = 0.5):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.default_temperature = temperature
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "mock":
            return None
        if self.provider == "groq":
            from groq import Groq
            return Groq(api_key=self.api_key)
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model_name)
        raise ValueError(f"Unknown provider: {self.provider}")

    def chat(self, messages: List[Dict], system_prompt: Optional[str] = None,
             temperature: Optional[float] = None) -> str:
        """
        messages: [{"role": "user"/"assistant", "content": "..."}]
        Returns the model's reply as a string.
        """
        temp = temperature if temperature is not None else self.default_temperature

        if self.provider == "mock":
            return "[MOCK] This is a placeholder response."

        if self.provider in ("groq", "openai"):
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                temperature=temp,
            )
            return response.choices[0].message.content

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=1024,
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

    def chat_with_retry(self, messages: List[Dict], system_prompt: Optional[str] = None,
                        temperature: Optional[float] = None, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                return self.chat(messages, system_prompt, temperature)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(k in err for k in ("rate_limit", "rate limit", "429", "too many"))
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
