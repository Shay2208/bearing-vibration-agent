"""诊断事件模型与发射器（P1 实时事件流的统一信封）。

事件结构（任务书 §4 P1）：
    {session_id, event, node, tool, status, label, summary, elapsed_ms, payload}

事件类型（9 种）：
    diagnosis_started / node_started / tool_started / tool_completed /
    node_completed / branch_selected / warning / diagnosis_completed / diagnosis_failed

发射器契约（对应任务书 §5.1「SSE 断开安全」）：
    - emit() 永不抛异常：发送失败、队列满、连接关闭一律静默丢弃，诊断本身继续；
    - close() 之后停止对外发送，但 events 列表仍完整记录（供会话追溯）；
    - elapsed_ms 由发射器统一计算（自创建时刻起，毫秒整型）；
    - session_id 只是一个 uuid，不含密钥、原始文件内容等敏感信息。
"""

from __future__ import annotations

import asyncio
import time
import uuid

EVENT_TYPES = (
    "diagnosis_started",
    "node_started",
    "tool_started",
    "tool_completed",
    "node_completed",
    "branch_selected",
    "warning",
    "diagnosis_completed",
    "diagnosis_failed",
)


def new_session_id() -> str:
    return uuid.uuid4().hex


class DiagnosisEmitter:
    """事件发射器：统一组装信封并记录到内存，子类决定是否对外发送。

    - 同步旧路径（/api/diagnose）注入本类实例：只记录不发网络；
    - SSE 路径注入 QueueEmitter：事件写入 asyncio.Queue 由响应流消费。
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or new_session_id()
        self._start = time.perf_counter()
        self._closed = False
        self.events: list[dict] = []

    def close(self) -> None:
        """停止对外发送；已记录的 events 保留。重复调用无副作用。"""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def build(
        self,
        event: str,
        *,
        node: str | None = None,
        tool: str | None = None,
        status: str | None = None,
        label: str | None = None,
        summary: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        return {
            "session_id": self.session_id,
            "event": event,
            "node": node,
            "tool": tool,
            "status": status,
            "label": label,
            "summary": summary,
            "elapsed_ms": int((time.perf_counter() - self._start) * 1000),
            "payload": payload,
        }

    async def emit(self, event: str, **fields) -> dict | None:
        """组装并发送一个事件；任何失败都只丢事件，绝不影响诊断流程。"""
        try:
            record = self.build(event, **fields)
            self.events.append(record)
            if not self._closed:
                await self._send(record)
            return record
        except Exception:
            return None

    async def _send(self, record: dict) -> None:
        """默认实现：只记录（events 列表），不对外发送。"""


class QueueEmitter(DiagnosisEmitter):
    """SSE 用：事件写入有界队列，消费端是 StreamingResponse 生成器。

    队列满（消费端断开或跟不上）时直接关闭自身：视为消费端已死，
    后续事件只进内存列表，诊断继续执行不阻塞。
    """

    def __init__(self, session_id: str | None = None, maxsize: int = 256):
        super().__init__(session_id)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def _send(self, record: dict) -> None:
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            self._closed = True
