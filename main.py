from openai import OpenAI
from typing import Optional, Tuple
import os


class ChatClient:
    """阿里云百炼 API 聊天客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.5-plus",
    ):
        """
        初始化聊天客户端

        Args:
            api_key: API 密钥，默认从环境变量读取或使用内置密钥
            base_url: API 基础 URL
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv(
            "DASHSCOPE_API_KEY", "sk-99706481a3154761a984918e1a26baa1"
        )
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        message: str,
        enable_thinking: bool = True,
        stream: bool = True,
    ) -> Tuple[str, str]:
        """
        发送聊天请求并获取回复

        Args:
            message: 用户消息
            enable_thinking: 是否启用思考模式
            stream: 是否使用流式输出

        Returns:
            (reasoning_content, answer_content) 思考内容和回复内容
        """
        messages = [{"role": "user", "content": message}]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"enable_thinking": enable_thinking} if enable_thinking else {},
            stream=stream,
            stream_options={"include_usage": True} if stream else {},
        )

        if not stream:
            return self._handle_non_stream(completion)

        return self._handle_stream(completion, enable_thinking)

    def _handle_non_stream(self, completion) -> Tuple[str, str]:
        """处理非流式响应"""
        choice = completion.choices[0].message
        reasoning_content = getattr(choice, "reasoning_content", "") or ""
        answer_content = choice.content or ""
        return reasoning_content, answer_content

    def _handle_stream(self, completion, enable_thinking: bool) -> Tuple[str, str]:
        """处理流式响应"""
        reasoning_content = ""
        answer_content = ""
        is_answering = False

        if enable_thinking:
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            if not chunk.choices:
                if enable_thinking:
                    print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
                    print(chunk.usage)
                continue

            delta = chunk.choices[0].delta

            if (
                enable_thinking
                and hasattr(delta, "reasoning_content")
                and delta.reasoning_content is not None
            ):
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
                reasoning_content += delta.reasoning_content

            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    if enable_thinking:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                print(delta.content, end="", flush=True)
                answer_content += delta.content

        print()
        return reasoning_content, answer_content


def main():
    """主函数"""
    client = ChatClient()
    message = "你是谁"
    reasoning, answer = client.chat(message)


if __name__ == "__main__":
    main()
