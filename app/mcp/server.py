"""MCP stdio 服务：把 app/mcp/vibration.py 的信号分析能力暴露成 MCP 工具。

启动方式（无参数，默认 stdio 传输）：

    python app/mcp/server.py

约定：
  - 只注册 3 个工具，只返回计算结果，绝不返回任何故障结论 / 故障名称。
  - 统一信封：成功 {"ok": true, "data": {...}}；失败 {"ok": false, "error": {code, message, field}}。
    业务异常在工具内部被捕获并转成结构化错误，不会穿透协议层。
  - stdout 是 MCP 协议通道，本模块任何日志都必须写 stderr（见 _stderr）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 允许 `python app/mcp/server.py` 直接启动时导入 app 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# mcp 2.x：FastMCP 已重命名为 MCPServer（from mcp.server.mcpserver import MCPServer）
from mcp.server import MCPServer

from app.mcp import vibration


def _stderr(message: str) -> None:
    """所有日志写 stderr，保持 stdout 只有协议报文。"""
    print(f"[vibration-mcp] {message}", file=sys.stderr, flush=True)


def _envelope(call) -> dict[str, Any]:
    """执行分析函数并包装成统一信封，业务异常转结构化错误。"""
    try:
        return {"ok": True, "data": call()}
    except vibration.VibrationError as exc:
        return {"ok": False, "error": exc.to_dict()}
    except Exception as exc:  # 兜底：不让任何异常穿透协议层
        _stderr(f"工具执行异常：{exc!r}")
        return {"ok": False, "error": {"code": "internal_error", "message": f"工具执行异常：{exc}", "field": None}}


server = MCPServer(name="bearing-vibration", version="1.0.0", log_level="ERROR")


@server.tool()
def validate_vibration_data(
    normal_path: str,
    abnormal_path: str,
    sampling_rate: float,
    sensor_position: str | None = None,
    min_samples: int = 1024,
) -> dict[str, Any]:
    """校验正常/异常两路振动信号文件与采样率，只做数据校验，不判断故障。

    返回 checks / errors / warnings 等校验结果；校验不通过时问题放在 data.errors 里。
    """
    return _envelope(
        lambda: vibration.validate_vibration_data(
            normal_path, abnormal_path, sampling_rate, sensor_position, min_samples
        )
    )


@server.tool()
def extract_vibration_features(
    csv_path: str,
    sampling_rate: float,
    band_edges: list[float] | None = None,
    rotation_speed: float | None = None,
    envelope_band: list[float] | None = None,
    include_series: bool = False,
) -> dict[str, Any]:
    """提取单路振动信号的时域与频域特征（RMS、峰值、峭度、主频、频带能量、频谱峰）。

    只输出特征数值，不做故障判定。band_edges 为递增的频带边界，缺省按 0~奈奎斯特均分 5 段。
    rotation_speed（rpm）非空时额外做包络解调（共振带滤波 → |hilbert| → 包络谱 → BPFO/BPFI/BSF/FTF 能量与峰位对齐）。
    include_series=True 时附带降采样的波形 / 幅度谱 / 包络谱曲线与 Int16 WAV（base64），供前端绘图与试听。
    """
    return _envelope(
        lambda: vibration.extract_vibration_features(
            csv_path, sampling_rate, band_edges, rotation_speed, envelope_band, include_series
        )
    )


@server.tool()
def compare_normal_abnormal(
    normal_features: dict[str, Any],
    abnormal_features: dict[str, Any],
    rotation_speed: float | None = None,
) -> dict[str, Any]:
    """对比正常与异常特征，输出结构化诊断依据（特征变化、频带能量变化、主频迁移、证据）。

    只描述现象，不给出故障结论。rotation_speed（rpm）非空时附上轴承特征频率及频谱匹配情况。
    """
    return _envelope(
        lambda: vibration.compare_normal_abnormal(normal_features, abnormal_features, rotation_speed)
    )


def main() -> None:
    _stderr(f"启动 stdio MCP 服务，项目根目录：{PROJECT_ROOT}")
    server.run()  # 默认 transport="stdio"


if __name__ == "__main__":
    main()