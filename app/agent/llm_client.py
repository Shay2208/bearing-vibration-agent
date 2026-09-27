"""大模型客户端：兼容 OpenAI 协议，所有失败都归一成 LLMError，不向外抛原始异常。

设计要点：
  - 允许被继承 / 被测试用假客户端替换（编排层只依赖 available 与 chat 两个成员）；
  - 未配置 Key 时立即返回 llm_unavailable，不触发任何网络请求；
  - 超时与其它异常分开编码，便于上层区分「模型不可用」与「模型超时」；
  - 结构化输出由 extract_json_object 解析（容忍 ```json 代码块与前后多余文字），
    解析失败抛 ValueError，由编排层降级为模板报告。
"""

from __future__ import annotations

import json

from app.config import settings

CODE_UNAVAILABLE = "llm_unavailable"
CODE_TIMEOUT = "llm_timeout"
CODE_ERROR = "llm_error"


class LLMError(Exception):
    """模型调用失败。属性：code ∈ {llm_unavailable, llm_timeout, llm_error}、message。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


def extract_json_object(text: str) -> dict:
    """从模型输出中解出 JSON 对象；无法解析时抛 ValueError（由编排层降级为模板）。

    结构化输出要求模型「只输出一个 JSON 对象」，但真实模型常会包一层 ```json 代码块
    或前后带一句说明，这里统一剥掉围栏后取最外层 { ... }。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("模型输出为空，未找到 JSON 对象")
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出中未找到 JSON 对象")
    payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型输出的 JSON 不是对象")
    return payload


class LLMClient:
    """极简同步客户端。每次 chat 调用建一次连接（演示项目，不做连接池）。"""

    @property
    def available(self) -> bool:
        return settings.llm_available

    # ------------------------------------------------------------------
    def _build_client(self):
        from openai import OpenAI

        # max_retries=0：重试策略由本类统一控制，避免与 SDK 内部重试叠加
        return OpenAI(
            base_url=settings.model_api_base,
            api_key=settings.model_api_key,
            timeout=settings.llm_timeout,
            max_retries=0,
        )

    @staticmethod
    def _wrap(exc: Exception) -> LLMError:
        name = type(exc).__name__
        if "Timeout" in name or isinstance(exc, TimeoutError):
            return LLMError(CODE_TIMEOUT, f"调用大模型超时（{type(exc).__name__}）：{exc}")
        return LLMError(CODE_ERROR, f"调用大模型失败（{type(exc).__name__}）：{exc}")

    def chat(self, system: str, user: str) -> str:
        """单轮对话，返回助手文本。失败抛 LLMError。"""
        if not self.available:
            raise LLMError(
                CODE_UNAVAILABLE,
                "未配置模型接口（需要同时设置 MODEL_API_BASE 与 MODEL_API_KEY），无法调用大模型",
            )

        attempts = max(1, int(settings.llm_max_retries) + 1)
        last: LLMError | None = None
        for _ in range(attempts):
            try:
                response = self._build_client().chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    last = LLMError(CODE_ERROR, "大模型返回了空内容")
                    continue
                return content
            except LLMError as exc:  # 空内容等自造错误
                last = exc
            except Exception as exc:
                last = self._wrap(exc)
        raise last or LLMError(CODE_ERROR, "调用大模型失败")