# coding: utf-8
import logging
from typing import List, Dict, Optional

from openai import OpenAI

from config import get_config

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self) -> None:
        cfg = get_config()
        self.cfg = cfg
        self.init_error: Optional[str] = None
        self.client = None
        self.default_model = cfg.DEFAULT_LLM_MODEL

        api_key = cfg.OPENAI_API_KEY
        if api_key:
            self.client = OpenAI(base_url=cfg.OPENAI_BASE_URL, api_key=api_key)
        else:
            self.init_error = "OPENAI_API_KEY环境变量未设置"
            raise RuntimeError(self.init_error or "OpenAI API KEY not initialized")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        if not self.client:
            raise RuntimeError(self.init_error or "OpenAI client not initialized")

        resolved_model = model or self.default_model
        logger.debug("Start chat completions...")
        logger.debug(f"default_model={self.default_model}")
        logger.debug(f"resolved_model={resolved_model}")
        try:
            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            logger.debug("Finished chat completions.")
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API调用失败：{str(e)}")

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        if not self.client:
            yield f"错误：{self.init_error or 'OpenAI client not initialized'}"
            return

        resolved_model = model or self.default_model
        try:
            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"错误：{str(e)}"


openai_service = OpenAIService()
