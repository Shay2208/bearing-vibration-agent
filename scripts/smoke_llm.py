r"""真实模型 smoke test：走项目现有编排入口，真实调用一次模型并如实打印结果。

用法（在项目根目录执行）：

    & ".\.venv\Scripts\python.exe" scripts/smoke_llm.py

做什么：
  - 配置从环境变量读取（`app/config.py` 会在导入时读取项目根下的 `.env`，已存在的环境变量不被覆盖）；
  - 用内置样例 `data/samples/normal_1797.csv`（基线）与 `data/samples/inner_race_1797.csv`（异常），
    采样率 12000、转速 1797、测点 drive_end，跑一次完整 `orchestrator.diagnose()`；
  - 模型调用复用现有模型层 `app/agent/llm_client.py` 的真实客户端（这里只加一层记录，用于打印
    模型的原始响应耗时与原始文本），不重写模型层、不伪造任何结果。

退出码：
  0 = 真实模型调用成功（trace.llm_mode == "llm"）；
  2 = 未配置模型接口（缺 MODEL_API_BASE / MODEL_API_KEY，或 USE_LLM=never）；
  1 = 其它失败（模型调用失败降级为模板、样例缺失、流程本身报错）。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import orchestrator  # noqa: E402
from app.agent.llm_client import LLMClient, LLMError  # noqa: E402
from app.config import settings  # noqa: E402

NORMAL_SAMPLE = "normal_1797.csv"
ABNORMAL_SAMPLE = "inner_race_1797.csv"
SAMPLING_RATE = 12000.0
ROTATION_SPEED = 1797.0
SENSOR_POSITION = "drive_end"
DEVICE_TYPE = "bearing"

CONFIG_GUIDE = r"""
怎么配置环境变量（任选其一，密钥只放环境变量或本地 .env，不要写进代码）：
  1) 当前 PowerShell 会话临时生效：
       $env:MODEL_API_BASE = "https://<你的服务地址>/v1"
       $env:MODEL_API_KEY  = "<你的密钥>"
       $env:MODEL_NAME     = "gpt-4o-mini"
       & ".\.venv\Scripts\python.exe" scripts/smoke_llm.py
  2) 写入项目根目录的 .env（推荐，模板见 .env.example）：
       Copy-Item .env.example .env
       # 编辑 .env，填好 MODEL_API_BASE 与 MODEL_API_KEY 后再运行本脚本
  说明：MODEL_API_BASE 与 MODEL_API_KEY 必须同时提供；USE_LLM=never 会强制禁用模型。
        .env 已被 .gitignore 忽略，请勿把真实密钥提交进仓库。
"""


class RecordingLLMClient(LLMClient):
    """真实客户端外面只加一层记录：不改变请求与响应，只记下真实耗时和原始文本。"""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def chat(self, system: str, user: str) -> str:
        started = time.perf_counter()
        try:
            text = super().chat(system, user)
        except LLMError as exc:
            self.records.append(
                {"ok": False, "elapsed_ms": int((time.perf_counter() - started) * 1000), "error": exc.to_dict()}
            )
            raise
        self.records.append(
            {"ok": True, "elapsed_ms": int((time.perf_counter() - started) * 1000), "text": text}
        )
        return text


def _truncate(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _print_config_state() -> None:
    print("== 模型配置（只读环境变量，不打印密钥明文）==")
    print(f"MODEL_API_BASE     : {settings.model_api_base or '(未配置)'}")
    print(f"MODEL_API_KEY      : {'已配置' if settings.model_api_key else '(未配置)'}")
    print(f"MODEL_NAME         : {settings.model_name}")
    print(f"EMBEDDING_MODEL    : {settings.embedding_model}")
    print(f"USE_LLM            : {settings.use_llm}")
    print(f"MCP_MODE           : {settings.mcp_mode}")
    print(f"llm_available      : {settings.llm_available}")


def main() -> int:
    print("== 轴承振动诊断 · 真实模型 smoke test ==")
    _print_config_state()

    if not settings.llm_available:
        print()
        print("[失败] 未配置可用的模型接口：需要同时设置 MODEL_API_BASE 与 MODEL_API_KEY（USE_LLM=never 也会禁用模型）。")
        print("       本次未发起任何模型调用，因此不输出故障类型、耗时等任何结果，也不会伪造成功。")
        print(CONFIG_GUIDE)
        return 2

    normal_path = settings.samples_dir / NORMAL_SAMPLE
    abnormal_path = settings.samples_dir / ABNORMAL_SAMPLE
    for path in (normal_path, abnormal_path):
        if not path.is_file():
            print(f"\n[失败] 内置样例不存在：{path}")
            return 1

    request = {
        "normal_path": str(normal_path.resolve()),
        "abnormal_path": str(abnormal_path.resolve()),
        "sampling_rate": SAMPLING_RATE,
        "rotation_speed": ROTATION_SPEED,
        "sensor_position": SENSOR_POSITION,
        "device_type": DEVICE_TYPE,
    }

    print("\n== 发起一次真实诊断（orchestrator.diagnose → 真实 LLMClient）==")
    print(f"正常样本（基线）: {normal_path}")
    print(f"异常样本        : {abnormal_path}")
    print(f"采样率 / 转速   : {SAMPLING_RATE} Hz / {ROTATION_SPEED} rpm · 测点 {SENSOR_POSITION}")

    client = RecordingLLMClient()
    started = time.perf_counter()
    result = asyncio.run(orchestrator.diagnose(request, llm_client=client))
    wall_ms = int((time.perf_counter() - started) * 1000)

    trace = result.get("trace") or {}
    report = result.get("report") or {}
    successes = [record for record in client.records if record["ok"]]

    selected_fault_type = "-"
    if successes:
        try:
            selected_fault_type = str(
                orchestrator._parse_llm_json(successes[-1]["text"]).get("selected_fault_type") or "-"
            )
        except Exception as exc:  # 仅用于展示，编排层已独立校验过模型输出
            selected_fault_type = f"(模型输出无法解析为 JSON：{exc})"

    print("\n== 结果 ==")
    print(f"llm_mode            : {trace.get('llm_mode')}")
    print(f"llm_calls           : {trace.get('llm_calls')}")
    print(f"llm_error           : {trace.get('llm_error')}")
    print(f"真实响应耗时        : "
          + (f"{successes[-1]['elapsed_ms']} ms（第 {len(client.records)} 次模型调用）" if successes else "无（没有成功的模型响应）"))
    print(f"诊断总耗时（墙钟）  : {wall_ms} ms")
    print(f"status              : {result.get('status')}")
    print(f"selected_fault_type : {selected_fault_type}")
    print(f"报告摘要            : {_truncate(report.get('conclusion'), 300) or '(无报告)'}")

    warnings = trace.get("warnings") or []
    if warnings:
        print("trace.warnings      :")
        for warning in warnings:
            print(f"  - {_truncate(warning, 200)}")

    print()
    if trace.get("llm_mode") == "llm":
        print("[成功] 真实模型调用成功，报告结论文本来自模型。")
        return 0
    if result.get("status") == "error":
        print(f"[失败] 诊断流程在 {result.get('error')} 处停机，未产生诊断结论。")
        return 1
    print(f"[失败] 模型未被成功使用（llm_mode={trace.get('llm_mode')}），请检查上面的 llm_error 与 trace.warnings。")
    return 1


if __name__ == "__main__":
    sys.exit(main())