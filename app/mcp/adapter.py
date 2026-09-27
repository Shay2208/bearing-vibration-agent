"""振动分析适配层：对上层（Agent 编排）暴露固定契约，屏蔽后端差异。

两种后端：
  - BACKEND_LOCAL = "local"      ：进程内直接调用 app/mcp/vibration.py（同步函数包成 async）
  - BACKEND_MCP   = "mcp:stdio"  ：通过 MCP 客户端调用真实 stdio MCP 服务（app/mcp/server.py）

后端由 settings.mcp_mode 决定（"local" 走本地，其余走 MCP stdio，默认 local）。

统一信封（三个 async 函数都只返回信封，绝不抛异常）：
  成功 {"ok": True,  "tool": <工具名>, "backend": <后端>, "data": {...}}
  失败 {"ok": False, "tool": <工具名>, "backend": <后端>,
        "error": {"code": "...", "message": "...", "field": None}}

将来替换成用户提供的正式 MCP 函数时，只需替换本模块的后端实现
（_server_parameters / _call_mcp），上层契约与返回信封保持不变。
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import PROJECT_ROOT, settings
from app.mcp import vibration

BACKEND_LOCAL = "local"
BACKEND_MCP = "mcp:stdio"

# 单次 MCP 调用（连接 + 调用 + 断开）的总超时
_MCP_TIMEOUT_SECONDS = 60.0

_ERROR_MCP_CONNECTION = "mcp_connection_error"
_ERROR_MCP_TIMEOUT = "mcp_timeout"
_ERROR_MCP_PROTOCOL = "mcp_protocol_error"
_ERROR_INTERNAL = "internal_error"
_ERROR_INVALID_SERVER_CMD = "invalid_server_command"


# ---------------------------------------------------------------------------
# 信封构造
# ---------------------------------------------------------------------------
def describe_backend() -> str:
    """返回当前实际使用的后端标识。"""
    return BACKEND_LOCAL if settings.mcp_mode == "local" else BACKEND_MCP


def _ok(tool: str, backend: str, data: Any) -> dict:
    return {"ok": True, "tool": tool, "backend": backend, "data": data}


def _fail(tool: str, backend: str, code: str, message: str, field: str | None = None) -> dict:
    return {"ok": False, "tool": tool, "backend": backend, "error": {"code": code, "message": message, "field": field}}


def _normalize_error(raw: Any) -> dict:
    """把工具返回的 error 归一成 {code, message, field}。"""
    if isinstance(raw, dict):
        return {
            "code": str(raw.get("code") or _ERROR_MCP_PROTOCOL),
            "message": str(raw.get("message") or "MCP 工具返回了错误但未提供 message"),
            "field": raw.get("field"),
        }
    return {"code": _ERROR_MCP_PROTOCOL, "message": f"MCP 工具返回的 error 结构不符合约定：{raw!r}", "field": None}


# ---------------------------------------------------------------------------
# 本地后端：进程内直接调用 vibration.py
# ---------------------------------------------------------------------------
async def _call_local(tool: str, func, *args, **kwargs) -> dict:
    try:
        data = await asyncio.to_thread(func, *args, **kwargs)
    except vibration.VibrationError as exc:
        return _fail(tool, BACKEND_LOCAL, exc.code, exc.message, exc.field)
    except Exception as exc:
        return _fail(tool, BACKEND_LOCAL, _ERROR_INTERNAL, f"本地分析函数执行异常：{exc!r}")
    return _ok(tool, BACKEND_LOCAL, data)


# ---------------------------------------------------------------------------
# MCP 后端：每次调用建立一次 stdio 连接
# ---------------------------------------------------------------------------
class ServerCommandError(ValueError):
    """MCP_SERVER_CMD 无法解析成可执行命令；由 _call_mcp 转成 invalid_server_command 信封。"""


def _strip_wrapping_quotes(token: str) -> str:
    """去掉 shlex 非 POSIX 模式下保留的包裹引号（单/双引号）。"""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def parse_server_command(raw: str) -> list[str]:
    """把 MCP_SERVER_CMD 解析成 argv 列表，支持以下两种配置形式：

    1) JSON 字符串数组（路径含空格 / 反斜杠时最稳妥，写法见 .env.example）：
         ["C:\\Program Files\\Python\\python.exe", "-m", "app.mcp.server"]
       整体按 JSON 解析，每个元素原样作为一个 argv，不再二次切分。

    2) 普通命令行字符串：可执行文件用引号包住即可，路径允许含空格：
         "C:\\path with space\\python.exe" -m app.mcp.server
       用 shlex 的**非 POSIX 模式**切分：该模式不启用转义符，
       Windows 路径里的反斜杠不会被当成转义符吃掉，只需再去掉包裹引号。

    解析失败一律抛 ServerCommandError（不返回半成品命令，也不静默兜底）。
    """
    text = (raw or "").strip()
    if not text:
        raise ServerCommandError("MCP_SERVER_CMD 为空，至少需要给出可执行文件")

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise ServerCommandError(f"MCP_SERVER_CMD 形似 JSON 数组但解析失败：{exc}") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) and item.strip() for item in parsed):
            raise ServerCommandError(
                'MCP_SERVER_CMD 的 JSON 形式必须是字符串数组，例如 ["python.exe", "-m", "app.mcp.server"]'
            )
        argv = list(parsed)
    else:
        try:
            tokens = shlex.split(text, posix=False)
        except ValueError as exc:  # 引号不配对
            raise ServerCommandError(f"MCP_SERVER_CMD 的引号不匹配，无法解析：{exc}") from exc
        argv = [_strip_wrapping_quotes(token) for token in tokens]

    if not argv or not argv[0].strip():
        raise ServerCommandError("MCP_SERVER_CMD 解析后没有得到可执行文件")
    return argv


def _server_command_argv() -> list[str]:
    """返回实际使用的 argv：留空回退到当前解释器 + app/mcp/server.py；否则解析并校验可执行文件。"""
    raw = settings.mcp_server_cmd.strip()
    if not raw:
        return [sys.executable, str(PROJECT_ROOT / "app" / "mcp" / "server.py")]

    argv = parse_server_command(raw)
    executable = argv[0]
    if Path(executable).is_absolute() or any(sep in executable for sep in ("/", "\\")):
        found = Path(executable).is_file()
    else:
        found = shutil.which(executable) is not None
    if not found:
        raise ServerCommandError(f"找不到可执行文件：{executable}（MCP_SERVER_CMD={raw!r}）")
    return argv


def _server_parameters() -> StdioServerParameters:
    """构造 stdio 启动参数：启动命令来自 _server_command_argv()。

    命令解析或可执行文件校验失败时抛 ServerCommandError，由 _call_mcp 转成结构化错误信封。
    cwd 设为项目根，保证子进程能 import app 包（容器内同样成立）。
    """
    argv = _server_command_argv()
    return StdioServerParameters(command=argv[0], args=argv[1:], cwd=str(PROJECT_ROOT))


def _extract_payload(result: Any) -> tuple[dict | None, str]:
    """从 CallToolResult 解出 {"ok": ...} 信封，返回 (信封, 失败原因)。"""
    if getattr(result, "is_error", False):
        texts = [getattr(item, "text", "") for item in getattr(result, "content", None) or []]
        return None, f"工具在协议层报错：{' '.join(t for t in texts if t)[:200]}"

    payload = getattr(result, "structured_content", None)
    if not isinstance(payload, dict):
        # 兼容只返回文本内容的服务实现
        payload = None
        for item in getattr(result, "content", None) or []:
            text = getattr(item, "text", None)
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break
    if not isinstance(payload, dict):
        return None, f"工具返回内容不是 JSON 对象：{result!r}"[:300]
    if not isinstance(payload.get("ok"), bool):
        return None, f"工具返回对象缺少布尔字段 ok，实际字段：{list(payload)[:10]}"
    return payload, ""


async def _call_mcp_once(params: StdioServerParameters, tool: str, arguments: dict) -> dict:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)

    payload, reason = _extract_payload(result)
    if payload is None:
        return _fail(tool, BACKEND_MCP, _ERROR_MCP_PROTOCOL, f"MCP 工具 {tool} 返回结构不符合约定：{reason}")
    if payload["ok"]:
        if not isinstance(payload.get("data"), dict):
            return _fail(tool, BACKEND_MCP, _ERROR_MCP_PROTOCOL, f"MCP 工具 {tool} 成功返回但缺少 data 对象")
        return _ok(tool, BACKEND_MCP, payload["data"])
    return {"ok": False, "tool": tool, "backend": BACKEND_MCP, "error": _normalize_error(payload.get("error"))}


async def _call_mcp(tool: str, arguments: dict) -> dict:
    """建立一次 stdio 连接 → 调用工具 → 断开（演示项目，不做连接池）。"""
    try:
        params = _server_parameters()
    except ServerCommandError as exc:
        # 命令配置有问题时直接返回结构化错误，不去启动子进程，也不抛裸异常
        return _fail(tool, BACKEND_MCP, _ERROR_INVALID_SERVER_CMD, f"MCP 服务启动命令无效：{exc}")
    try:
        return await asyncio.wait_for(_call_mcp_once(params, tool, arguments), timeout=_MCP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return _fail(tool, BACKEND_MCP, _ERROR_MCP_TIMEOUT, f"调用 MCP 工具 {tool} 超时（超过 {_MCP_TIMEOUT_SECONDS:.0f}s）")
    except Exception as exc:
        command = " ".join([params.command, *[str(a) for a in params.args]])
        return _fail(tool, BACKEND_MCP, _ERROR_MCP_CONNECTION, f"无法与 MCP 服务通信（命令：{command}）：{exc!r}")


# ---------------------------------------------------------------------------
# 对外契约
# ---------------------------------------------------------------------------
async def validate_vibration_data(
    normal_path,
    abnormal_path,
    sampling_rate,
    sensor_position=None,
) -> dict:
    """校验两路信号与采样率。数据本身的问题（如坏值、采样率不符）放在返回的 data.errors 里。"""
    tool = "validate_vibration_data"
    if describe_backend() == BACKEND_LOCAL:
        return await _call_local(
            tool, vibration.validate_vibration_data, normal_path, abnormal_path, sampling_rate, sensor_position
        )
    payload: dict[str, Any] = {
        "normal_path": str(normal_path),
        "abnormal_path": str(abnormal_path),
        "sampling_rate": sampling_rate,
    }
    if sensor_position is not None:
        payload["sensor_position"] = sensor_position
    return await _call_mcp(tool, payload)


async def extract_vibration_features(
    csv_path, sampling_rate, band_edges=None, rotation_speed=None, envelope_band=None, include_series=False
) -> dict:
    """提取单路信号的时域 / 频域特征；rotation_speed 非空时附包络谱特征频率分析。

    include_series=True 时返回 data 内额外带 series 块（降采样曲线 + WAV base64），
    供上层落盘旁挂音频与前端手绘；两条后端路径都透传该参数。
    """
    tool = "extract_vibration_features"
    if describe_backend() == BACKEND_LOCAL:
        return await _call_local(
            tool, vibration.extract_vibration_features,
            csv_path, sampling_rate, band_edges, rotation_speed, envelope_band, include_series,
        )
    payload: dict[str, Any] = {"csv_path": str(csv_path), "sampling_rate": sampling_rate}
    if band_edges is not None:
        payload["band_edges"] = band_edges
    if rotation_speed is not None:
        payload["rotation_speed"] = rotation_speed
    if envelope_band is not None:
        payload["envelope_band"] = envelope_band
    if include_series:
        payload["include_series"] = True
    return await _call_mcp(tool, payload)


async def compare_normal_abnormal(normal_features, abnormal_features, rotation_speed=None) -> dict:
    """对比正常 / 异常特征，返回变化与证据，不做故障判定。"""
    tool = "compare_normal_abnormal"
    if describe_backend() == BACKEND_LOCAL:
        return await _call_local(
            tool, vibration.compare_normal_abnormal, normal_features, abnormal_features, rotation_speed
        )
    payload: dict[str, Any] = {"normal_features": normal_features, "abnormal_features": abnormal_features}
    if rotation_speed is not None:
        payload["rotation_speed"] = rotation_speed
    return await _call_mcp(tool, payload)
