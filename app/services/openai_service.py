import os
from typing import List, Dict, Optional
from openai import OpenAI


class OpenAIService:
    def __init__(self):
        # 从环境变量获取API密钥；缺失时不要在导入阶段直接抛错，
        # 以免导致整个服务无法启动。
        api_key = os.getenv("OPENAI_API_KEY")
        self.init_error: Optional[str] = None
        self.client = None

        if api_key:
            self.client = OpenAI(base_url="https://api.lingyaai.cn/v1/", api_key=api_key)
        else:
            self.init_error = "OPENAI_API_KEY环境变量未设置"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        调用OpenAI聊天补全API
        """
        if not self.client:
            raise RuntimeError(self.init_error or "OpenAI client not initialized")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API调用失败：{str(e)}")

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        流式聊天补全（用于实时打字效果）
        """
        if not self.client:
            yield f"错误：{self.init_error or 'OpenAI client not initialized'}"
            return

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"错误：{str(e)}"


# 创建全局实例
openai_service = OpenAIService()
