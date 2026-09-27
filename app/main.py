"""FastAPI 接口层：暴露健康检查、内置样例清单、诊断接口（同步 + SSE）与受控追问，并挂载静态前端。

业务判定全部交给编排层；本层只负责表单解析、输入来源选择、临时文件清理与错误归一化。
P1：/api/diagnose/stream 用 SSE 推送真实节点/工具事件；P2：/api/sessions/{id}/ask 受控追问。
"""

import asyncio
import json
import logging
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent import orchestrator, sessions
from app.agent.events import DiagnosisEmitter, QueueEmitter
from app.config import settings
from app.mcp import adapter
from app.rag import retriever
from app.schemas import (
    AskRequest,
    AskResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DiagnoseResponse,
    HealthResponse,
    SampleInfo,
    SampleListResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionSummary,
)

logger = logging.getLogger("bearing.api")

app = FastAPI(title="工业轴承振动故障诊断助手", version="0.1.0")

DEFAULT_SENSOR_POSITION = "drive_end"
DEFAULT_DEVICE_TYPE = "bearing"


# ---------------------------------------------------------------------------
# 内置样例元数据
# ---------------------------------------------------------------------------
def _load_samples() -> list[dict]:
    """直接读取 data/samples/*.json，按 sample_id 排序，并附上内部文件路径。"""
    items: list[dict] = []
    for meta_file in sorted(settings.samples_dir.glob("*.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("跳过无法解析的样本元数据 %s：%s", meta_file, exc)
            continue
        if not isinstance(meta, dict):
            logger.warning("跳过结构不是对象的样本元数据 %s", meta_file)
            continue
        meta.setdefault("sample_id", meta_file.stem)
        meta["path"] = str((settings.samples_dir / str(meta.get("file") or f"{meta_file.stem}.csv")).resolve())
        items.append(meta)
    items.sort(key=lambda item: str(item.get("sample_id") or ""))
    return items


def _sample_index() -> dict[str, dict]:
    return {str(item["sample_id"]): item for item in _load_samples()}


# ---------------------------------------------------------------------------
# 输入来源解析
# ---------------------------------------------------------------------------
async def _resolve_slot(
    field: str,
    label: str,
    upload: UploadFile | None,
    sample_id: str | None,
    index: dict[str, dict],
    tmp_dir: str,
) -> tuple[str | None, dict | None]:
    """上传文件优先于内置样例 id；返回 (内部 CSV 路径, 样例元数据)，都未提供则 (None, None)。"""
    if upload is not None and (upload.filename or "").strip():
        if Path(upload.filename).suffix.lower() != ".csv":
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_file_type", "message": f"{label}上传文件必须是 .csv 文件，收到：{upload.filename}"},
            )
        if upload.size is not None and upload.size > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "file_too_large",
                    "message": f"{label}上传文件超过大小限制（最大 {limit_mb:g} MB）。",
                },
            )
        target = Path(tmp_dir) / f"{field}.csv"
        total = 0
        with target.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    limit_mb = settings.max_upload_bytes / (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "file_too_large",
                            "message": f"{label}上传文件超过大小限制（最大 {limit_mb:g} MB）。",
                        },
                    )
                output.write(chunk)
        return str(target.resolve()), None

    key = (sample_id or "").strip()
    if key:
        meta = index.get(key)
        if meta is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "sample_not_found",
                    "message": f"{label}指定的内置样例 id 不存在：{key}；可用 id：{'、'.join(sorted(index))}",
                },
            )
        return meta["path"], meta

    return None, None


def _input_label(upload: UploadFile | None, sample_id: str | None) -> str:
    if upload is not None and (upload.filename or "").strip():
        return Path(upload.filename).name
    return (sample_id or "").strip() or "-"


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        mcp_backend=adapter.describe_backend(),
        llm_available=settings.llm_available,
        retrieval_mode=retriever.get_store().retrieval_mode,
        sample_count=len(_load_samples()),
    )


@app.get("/api/samples", response_model=SampleListResponse)
async def list_samples() -> SampleListResponse:
    items = [{key: value for key, value in item.items() if key != "path"} for item in _load_samples()]
    return SampleListResponse(count=len(items), samples=[SampleInfo(**item) for item in items])


# ---------------------------------------------------------------------------
# 输入解析（同步接口与 SSE 接口共用同一套逻辑）
# ---------------------------------------------------------------------------
async def _build_request_payload(
    sampling_rate: float,
    rotation_speed: float,
    normal_file: UploadFile | None,
    abnormal_file: UploadFile | None,
    normal_sample: str | None,
    abnormal_sample: str | None,
    sensor_position: str | None,
    device_type: str | None,
    tmp_dir: str,
) -> dict:
    """解析输入来源并组装编排层请求；文件落到 tmp_dir，输入缺失/非法时抛 HTTPException。"""
    index = _sample_index()
    normal_path, normal_meta = await _resolve_slot(
        "normal", "正常（基线）", normal_file, normal_sample, index, tmp_dir
    )
    abnormal_path, abnormal_meta = await _resolve_slot(
        "abnormal", "异常", abnormal_file, abnormal_sample, index, tmp_dir
    )

    missing = [
        text
        for text, path in (("normal（正常/基线）", normal_path), ("abnormal（异常）", abnormal_path))
        if path is None
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "missing_input",
                "message": "缺少" + "、".join(missing) + "振动数据来源：请上传对应 CSV 文件或指定内置样例 id（二者至少给一个）。",
            },
        )

    fallback_meta = normal_meta or abnormal_meta or {}
    return {
        "normal_path": normal_path,
        "abnormal_path": abnormal_path,
        "sampling_rate": sampling_rate,
        "rotation_speed": rotation_speed,
        "sensor_position": (sensor_position or "").strip()
        or str(fallback_meta.get("sensor_position") or DEFAULT_SENSOR_POSITION),
        "device_type": (device_type or "").strip() or DEFAULT_DEVICE_TYPE,
    }


def _session_summary(
    request_payload: dict,
    normal_file: UploadFile | None,
    abnormal_file: UploadFile | None,
    normal_sample: str | None,
    abnormal_sample: str | None,
) -> dict:
    """会话保存的输入摘要（脱敏：只有标签与工况参数，不含文件内容与路径）。"""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "normal_input": _input_label(normal_file, normal_sample),
        "abnormal_input": _input_label(abnormal_file, abnormal_sample),
        "sampling_rate": request_payload.get("sampling_rate"),
        "rotation_speed": request_payload.get("rotation_speed"),
        "sensor_position": request_payload.get("sensor_position"),
        "device_type": request_payload.get("device_type"),
    }


@app.post("/api/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    sampling_rate: float = Form(..., gt=0),
    rotation_speed: float = Form(..., gt=0),
    normal_file: UploadFile | None = File(default=None),
    abnormal_file: UploadFile | None = File(default=None),
    normal_sample: str | None = Form(default=None),
    abnormal_sample: str | None = Form(default=None),
    sensor_position: str | None = Form(default=None),
    device_type: str | None = Form(default=None),
):
    # 每个请求一个专属临时目录，无论成功失败都在 finally 里清理
    with tempfile.TemporaryDirectory(prefix="bearing-diagnose-") as tmp_dir:
        request_payload = await _build_request_payload(
            sampling_rate, rotation_speed, normal_file, abnormal_file,
            normal_sample, abnormal_sample, sensor_position, device_type, tmp_dir,
        )
        # P2：同步路径也注入发射器（只记录不发网络），与会话保存共用事件
        emitter = DiagnosisEmitter()
        result = await orchestrator.diagnose(request_payload, emitter=emitter)

    sessions.store.save(
        result.get("session_id") or emitter.session_id,
        request_summary=_session_summary(request_payload, normal_file, abnormal_file, normal_sample, abnormal_sample),
        result=result,
        events=emitter.events,
    )

    trace = result.get("trace") or {}
    logger.info(
        "诊断完成 normal=%s abnormal=%s status=%s tool_calls=%s elapsed_ms=%s llm_mode=%s session=%s",
        _input_label(normal_file, normal_sample),
        _input_label(abnormal_file, abnormal_sample),
        result.get("status"),
        trace.get("tool_calls"),
        trace.get("elapsed_ms"),
        trace.get("llm_mode"),
        result.get("session_id"),
    )
    return result


# ---------------------------------------------------------------------------
# P1：SSE 实时事件流（与同步接口共用同一套诊断逻辑）
# ---------------------------------------------------------------------------
@app.post("/api/diagnose/stream")
async def diagnose_stream(
    sampling_rate: float = Form(..., gt=0),
    rotation_speed: float = Form(..., gt=0),
    normal_file: UploadFile | None = File(default=None),
    abnormal_file: UploadFile | None = File(default=None),
    normal_sample: str | None = Form(default=None),
    abnormal_sample: str | None = Form(default=None),
    sensor_position: str | None = Form(default=None),
    device_type: str | None = Form(default=None),
):
    # 输入解析在流开始之前完成：解析失败按普通 HTTP 4xx 返回 JSON
    tmp_dir = tempfile.mkdtemp(prefix="bearing-stream-")
    try:
        request_payload = await _build_request_payload(
            sampling_rate, rotation_speed, normal_file, abnormal_file,
            normal_sample, abnormal_sample, sensor_position, device_type, tmp_dir,
        )
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    emitter = QueueEmitter()
    session_id = emitter.session_id
    summary = _session_summary(request_payload, normal_file, abnormal_file, normal_sample, abnormal_sample)

    async def run() -> dict:
        # 后台诊断任务：SSE 断开后仍跑完并保存会话，最后清理临时目录
        try:
            result = await orchestrator.diagnose(request_payload, emitter=emitter)
            sessions.store.save(
                result.get("session_id") or session_id,
                request_summary=summary,
                result=result,
                events=emitter.events,
            )
            return result
        finally:
            emitter.close()
            try:
                emitter.queue.put_nowait(None)  # 哨兵：告知消费端事件已全部发出
            except asyncio.QueueFull:
                pass
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def event_stream():
        task = asyncio.create_task(run())
        try:
            while True:
                item = await emitter.queue.get()
                if item is None:
                    break
                yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
        finally:
            # 客户端断开：只停止向连接写入；后台诊断继续跑完，不产生未捕获异常
            emitter.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# P2：受控追问（只能基于本次会话保存的上下文回答）
# ---------------------------------------------------------------------------
@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """历史会话列表（按创建时间倒序），供前端刷新后恢复会话窗口。"""
    items = sessions.store.list_sessions()
    return SessionListResponse(count=len(items), sessions=[SessionSummary(**item) for item in items])


@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str) -> SessionDetailResponse:
    """单个会话的完整记录：事件轨迹、诊断结果与落盘的对话消息。"""
    session = sessions.store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "session_not_found",
                "message": f"诊断会话不存在或已过期：{session_id}。请从会话列表中选择已有会话，或先完成一次诊断。",
            },
        )
    return SessionDetailResponse(
        session_id=session_id,
        created_at=session["created_at"],
        request_summary=session["request_summary"],
        result=session["result"],
        events=session["events"],
        messages=[ChatMessage(**item) for item in sessions.store.messages(session_id)],
    )


# ---------------------------------------------------------------------------
# 原始波形音频（诊断过程中旁挂落盘，按需懒加载；不进响应体与会话数据库）
# ---------------------------------------------------------------------------
_AUDIO_SLOTS = {"normal", "abnormal"}
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


@app.get("/api/sessions/{session_id}/audio/{slot}.wav", include_in_schema=False)
async def session_audio(session_id: str, slot: str) -> FileResponse:
    """返回某次诊断落盘的波形 WAV（正常 / 异常两路之一）。"""
    if slot not in _AUDIO_SLOTS or not _SESSION_ID_RE.match(session_id):
        raise HTTPException(
            status_code=404,
            detail={"code": "audio_not_found", "message": f"音频标识不合法：{session_id}/{slot}，仅支持 normal 或 abnormal。"},
        )
    path = settings.audio_dir / f"{session_id}_{slot}.wav"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "audio_not_found",
                "message": f"该会话没有可播放的 {slot} 波形音频（可能诊断未完成特征提取，或旁挂文件已按上限清理）。",
            },
        )
    return FileResponse(path, media_type="audio/wav", filename=f"{session_id}_{slot}.wav")


@app.post("/api/sessions/{session_id}/ask", response_model=AskResponse)
async def ask_session(session_id: str, body: AskRequest) -> AskResponse:
    question = (body.question or "").strip()
    try:
        answer = sessions.answer_question(session_id, question)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "session_not_found",
                "message": f"诊断会话不存在或已过期：{session_id}。请先完成一次诊断，再使用返回的 session_id 追问。",
            },
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "question_not_allowed",
                "message": "该问题不在受控追问清单内，请从下列问题中选择：",
                "allowed_questions": [item["text"] for item in sessions.ALLOWED_QUESTIONS],
            },
        )
    return AskResponse(**answer)


# ---------------------------------------------------------------------------
# P4：会话内自由问答（问题自由输入，回答被限制在本次会话事实内）
# ---------------------------------------------------------------------------
@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat_session(session_id: str, body: ChatRequest) -> ChatResponse:
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "empty_question",
                "message": "问题不能为空：请输入要基于本次诊断会话提问的内容（最多 300 字）。",
            },
        )
    try:
        # 模型 SDK 是同步阻塞的，放线程池里执行，避免阻塞事件循环
        answer = await asyncio.to_thread(sessions.answer_free_question, session_id, question)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "session_not_found",
                "message": f"诊断会话不存在或已过期：{session_id}。请先完成一次诊断，再使用返回的 session_id 提问。",
            },
        )
    return ChatResponse(**answer)


# ---------------------------------------------------------------------------
# 异常归一化
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = [
        ".".join(str(part) for part in err.get("loc", []) if part not in ("body", "query", "form", "path"))
        for err in exc.errors()
    ]
    fields = [name for name in fields if name]
    detail = "字段 " + "、".join(fields) if fields else "请求参数"
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_argument",
                "message": f"请求参数校验未通过（{detail}）：sampling_rate 与 rotation_speed 必须是正数，且不可为空。",
            }
        },
    )


@app.exception_handler(Exception)
async def _on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("处理 %s %s 时出现未预期异常", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "internal_error", "message": f"服务内部错误（{type(exc).__name__}），请查看服务日志。"}},
    )


@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
async def app_alias() -> RedirectResponse:
    """Keep the portfolio/demo URL used by resume links compatible with the root page."""
    return RedirectResponse(url="/", status_code=307)


# 挂载必须放在所有 API 路由之后；check_dir=False 保证 app/web 尚未创建时也能启动
app.mount("/", StaticFiles(directory=str(settings.web_dir), html=True, check_dir=False), name="web")
