#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据准备脚本：下载 / 解析 CWRU 轴承数据，切窗并导出仓库内置样例。

用法示例：
    python data/prepare_data.py                          # 下载并同时生成 data/raw 与 data/samples
    python data/prepare_data.py --target samples         # 只生成仓库内置样例
    python data/prepare_data.py --files 97,105           # 只处理部分 fileId
    python data/prepare_data.py --local-mat-dir D:/cwru  # 离线：使用本地已有 .mat，跳过下载
    python data/prepare_data.py --target samples --force # 刷新内置样例（重新下载 + 重算）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402

# ---------------------------------------------------------------- 固定元数据常量
SAMPLING_RATE = 12000            # Hz：12k Drive End 数据的固定采样率
SENSOR_POSITION = "drive_end"    # 固定测点：驱动端（X{n}_DE_time）
ROTATION_SPEED = 1797            # rpm：0 hp 负载下的近似转速
DEVICE_TYPE = "bearing"
SOURCE_NAME = "CWRU Bearing Data Center"
SOURCE_URL = "https://engineering.case.edu/bearingdatacenter/download-data-file"
DATASET_HOME = "https://engineering.case.edu/bearingdatacenter/welcome"

MAT_URL_TEMPLATE = "https://engineering.case.edu/sites/default/files/{file_id}.mat"
MIN_MAT_SIZE = 100 * 1024        # 小于 100KB 视为下载不完整
DOWNLOAD_TIMEOUT = 120
USER_AGENT = "Mozilla/5.0 (compatible; bearing-vibration-agent data prep)"

DEFAULT_WINDOW_SIZE = 12000      # 12000 点 @12kHz = 1 秒
DEFAULT_WINDOW_INDEX = 0

RAW_DIR = settings.data_dir / "raw"
MAT_DIR = RAW_DIR / "mat"
SAMPLES_DIR = settings.samples_dir

# ---------------------------------------------------------------- 切窗规则（固定）
# 所有样本使用完全一致的切窗规则，保证离线评测可复现：
#   1) 采样率固定 12000 Hz（CWRU 12k Drive End 数据）；
#   2) 测点固定 drive_end，即只取 .mat 中的 X{n}_DE_time 变量；
#   3) 窗口长度固定 window_size 点（默认 12000 点 = 1 秒）；
#   4) 窗口之间不重叠：stride == window_size，第 k 个窗口起点 = k * window_size；
#   5) 时间戳从 0 开始，第 i 个采样点 t = i / sampling_rate（单位秒，%.6f）。
# 改变以上任意一条都会让样本不可比，因此不在脚本外提供随机切窗能力。

NORMAL_NOTE = "窗口为连续 12000 点不重叠片段；condition 仅用于离线评测，不作为在线诊断输入"


@dataclass(frozen=True)
class CwruFile:
    file_id: str            # CWRU 下载用的文件编号
    sample_id: str          # 输出文件名（不含扩展名）
    condition: str          # 固定取值：normal / inner_race_fault / outer_race_fault / ball_fault
    condition_label: str    # 中文标签
    de_key: str             # .mat 中的驱动端时域变量名
    source_detail: str


# 本首版固定使用：12k Drive End、0 hp / 1797 rpm、同一测点 drive_end
CWRU_FILES: tuple[CwruFile, ...] = (
    CwruFile(
        "97",
        "normal_1797",
        "normal",
        "正常",
        "X097_DE_time",
        "12k Drive End Bearing Fault Data, X097_DE_time, normal baseline (no fault), 0 hp",
    ),
    CwruFile(
        "105",
        "inner_race_1797",
        "inner_race_fault",
        "内圈故障",
        "X105_DE_time",
        "12k Drive End Bearing Fault Data, X105_DE_time, inner race fault, fault diameter 0.007 inch, 0 hp",
    ),
    CwruFile(
        "130",
        "outer_race_1797",
        "outer_race_fault",
        "外圈故障",
        "X130_DE_time",
        "12k Drive End Bearing Fault Data, X130_DE_time, outer race fault at 6 o'clock, "
        "fault diameter 0.007 inch, 0 hp",
    ),
    CwruFile(
        "118",
        "ball_1797",
        "ball_fault",
        "滚动体故障",
        "X118_DE_time",
        "12k Drive End Bearing Fault Data, X118_DE_time, ball fault, fault diameter 0.007 inch, 0 hp",
    ),
)

FILE_BY_ID = {spec.file_id: spec for spec in CWRU_FILES}

# 人为构造的损坏样本：总行数约 300（明显短于 1024 点），其中 7 行为问题值
CORRUPTED_SAMPLE_ID = "corrupted_1797"
CORRUPTED_ROWS = 300
CORRUPTED_PROBLEM_ROWS = {
    17: "",       # amplitude 为空
    58: "null",   # 非数字文本
    103: "N/A",   # 非数字文本
    149: "",      # amplitude 为空
    201: "NaN",   # NaN
    244: "nan",   # NaN
    288: "--",    # 非数字文本
}


class PrepareError(RuntimeError):
    """数据准备过程中的可预期失败（下载 / 解析 / 切窗）。"""


# ---------------------------------------------------------------- 通用工具
def write_text_atomic(path: Path, text: str) -> None:
    """先写临时文件再原子替换，避免半截文件破坏已有样例。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def build_csv(signal: np.ndarray) -> str:
    """生成 timestamp,amplitude 两列 CSV；时间戳 %.6f，幅值 8 位有效数字。"""
    lines = ["timestamp,amplitude"]
    for i, value in enumerate(np.asarray(signal, dtype=float).ravel()):
        lines.append(f"{i / SAMPLING_RATE:.6f},{value:.8g}")
    return "\n".join(lines) + "\n"


def build_metadata(sample_id: str, condition: str, condition_label: str, n_samples: int,
                   source_detail: str, note: str, source: str = SOURCE_NAME,
                   source_url: str = SOURCE_URL) -> dict:
    """元数据字段名固定，不要在别处另起炉灶。"""
    return {
        "sample_id": sample_id,
        "file": f"{sample_id}.csv",
        "device_type": DEVICE_TYPE,
        "sampling_rate": SAMPLING_RATE,
        "rotation_speed": ROTATION_SPEED,
        "sensor_position": SENSOR_POSITION,
        "condition": condition,
        "condition_label": condition_label,
        "n_samples": n_samples,
        "source": source,
        "source_url": source_url,
        "source_detail": source_detail,
        "note": note,
    }


def dump_metadata(meta: dict) -> str:
    return json.dumps(meta, ensure_ascii=False, indent=2) + "\n"


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    headers = ("fileId", "condition", "采样率", "转速", "窗口长度", "输出路径", "状态")
    keys = ("file_id", "condition", "sampling_rate", "rotation_speed", "window_size", "paths", "status")
    table = [headers]
    for row in rows:
        table.append(tuple(str(row.get(k, "-")) for k in keys))
    widths = [max(_display_width(line[i]) for line in table) for i in range(len(headers))]

    print("\n================ 汇总 ================")
    for idx, line in enumerate(table):
        print("  ".join(_pad(cell, widths[i]) for i, cell in enumerate(line)).rstrip())
        if idx == 0:
            print("  ".join("-" * w for w in widths))
    print("=====================================")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- 下载
def _download_once(spec: CwruFile, dest: Path) -> int:
    """真正执行一次下载，返回文件字节数；失败时清理临时文件后抛出异常。"""
    url = MAT_URL_TEMPLATE.format(file_id=spec.file_id)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=DOWNLOAD_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)
        size = tmp.stat().st_size
        if size <= MIN_MAT_SIZE:
            raise PrepareError(f"下载内容不完整（仅 {size} 字节，应为 >100KB）")
        os.replace(tmp, dest)
        return size
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_mat(spec: CwruFile, mat_dir: Path, force: bool) -> tuple[Path, str]:
    """下载 .mat 到本地缓存；已存在且 >100KB 直接复用。返回 (路径, 状态)。"""
    dest = mat_dir / f"{spec.file_id}.mat"
    url = MAT_URL_TEMPLATE.format(file_id=spec.file_id)

    if dest.is_file() and dest.stat().st_size > MIN_MAT_SIZE and not force:
        return dest, "缓存命中"

    mat_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in (1, 2):  # 官方站点偶发断连，重试一次即可，不做无限重试
        try:
            size = _download_once(spec, dest)
            return dest, f"已下载 {size // 1024} KB"
        except Exception as exc:
            last_error = exc
            print(f"[警告] fileId={spec.file_id} 第 {attempt} 次下载失败"
                  f"（{type(exc).__name__}: {exc}）")

    if dest.is_file() and dest.stat().st_size > MIN_MAT_SIZE:
        print(f"[警告] fileId={spec.file_id} 下载失败，改用本地缓存继续。")
        return dest, "下载失败，使用缓存"
    raise PrepareError(f"下载失败（{type(last_error).__name__}: {last_error}）；URL={url}") from last_error


# ---------------------------------------------------------------- 解析与切窗
def extract_de_signal(mat_path: Path, spec: CwruFile) -> tuple[np.ndarray, int | None]:
    """从 .mat 中取出驱动端时域信号以及（若可得的）转速。"""
    try:
        mat = loadmat(str(mat_path))
    except Exception as exc:
        raise PrepareError(f"解析 .mat 失败（{type(exc).__name__}: {exc}），文件可能损坏或不是 v5/v7 格式") from exc

    signal = None
    if spec.de_key in mat:
        signal = mat[spec.de_key]
    else:
        candidates = [key for key in mat if key.endswith("_DE_time")]
        print(f"[提示] fileId={spec.file_id} 未找到变量 {spec.de_key}，"
              f"该文件的 _DE_time 变量为：{candidates or '无'}")
        if candidates:
            signal = mat[candidates[0]]
            print(f"       已改用第一个匹配变量：{candidates[0]}")
    if signal is None:
        raise PrepareError(f"{mat_path.name} 中找不到 {spec.de_key} 或任何 _DE_time 变量")

    signal = np.asarray(signal, dtype=float).ravel()
    if signal.size == 0:
        raise PrepareError(f"{mat_path.name} 中 {spec.de_key} 为空")

    rpm = None
    for key in (f"X{int(spec.file_id):03d}RPM", f"X{int(spec.file_id):03d}_RPM", "RPM"):
        if key in mat:
            rpm = int(np.asarray(mat[key]).ravel()[0])
            break
    return signal, rpm


def slice_window(signal: np.ndarray, window_size: int, window_index: int, spec: CwruFile) -> np.ndarray:
    """按固定规则取第 window_index 个不重叠窗口。"""
    start = window_index * window_size
    end = start + window_size
    if end > signal.size:
        raise PrepareError(
            f"窗口越界：信号共 {signal.size} 点，无法取出 [第 {window_index} 个窗口，"
            f"{window_size} 点，需 {end} 点]；请调小 --window-size 或 --window-index"
        )
    segment = signal[start:end]
    if not np.all(np.isfinite(segment)):
        bad = int(np.count_nonzero(~np.isfinite(segment)))
        raise PrepareError(f"窗口内含 {bad} 个非有限值（NaN/Inf），已跳过该文件")
    if np.all(segment == 0):
        raise PrepareError("窗口内全为 0，疑似取到了无效片段，请换 --window-index")
    return segment


# ---------------------------------------------------------------- 单文件处理
def process_cwru_file(spec: CwruFile, args: argparse.Namespace, targets: list[Path]) -> dict:
    if args.local_mat_dir:
        mat_path = Path(args.local_mat_dir) / f"{spec.file_id}.mat"
        if not mat_path.is_file():
            raise PrepareError(f"本地 .mat 不存在：{mat_path}")
        mat_status = "本地文件"
    else:
        mat_path, mat_status = download_mat(spec, MAT_DIR, args.force)

    signal, rpm = extract_de_signal(mat_path, spec)
    segment = slice_window(signal, args.window_size, args.window_index, spec)

    csv_text = build_csv(segment)
    meta = build_metadata(
        spec.sample_id, spec.condition, spec.condition_label, int(segment.size), spec.source_detail, NORMAL_NOTE
    )
    outputs = []
    for target in targets:
        csv_path = target / f"{spec.sample_id}.csv"
        write_text_atomic(csv_path, csv_text)
        outputs.append(csv_path)
        # 元数据 JSON 仅随内置样例（data/samples）归档
        if target == SAMPLES_DIR:
            write_text_atomic(target / f"{spec.sample_id}.json", dump_metadata(meta))

    # 转速：.mat 里的 X{n}RPM 字段为实测值（偶为 1796），元数据 JSON 仍按本首版固定的 1797 写入
    measured_rpm = rpm if rpm is not None else ROTATION_SPEED
    if measured_rpm != ROTATION_SPEED:
        measured_rpm = f"{measured_rpm}（标称 {ROTATION_SPEED}）"
    return {
        "file_id": spec.file_id,
        "condition": spec.condition,
        "sampling_rate": SAMPLING_RATE,
        "rotation_speed": measured_rpm,
        "window_size": args.window_size,
        "paths": "; ".join(_rel(p) for p in outputs),
        "status": f"成功（{mat_status}）",
    }


# ---------------------------------------------------------------- 损坏样本
def build_corrupted_csv() -> str:
    """构造损坏样本：整体约 300 行，其中 7 行含空值 / 非数字文本 / NaN。"""
    rng = np.random.default_rng(1797)
    time_axis = np.arange(CORRUPTED_ROWS) / SAMPLING_RATE
    signal = 1.6 * np.sin(2 * np.pi * 160 * time_axis) + 0.25 * rng.standard_normal(CORRUPTED_ROWS)

    lines = ["timestamp,amplitude"]
    for i in range(CORRUPTED_ROWS):
        amplitude = CORRUPTED_PROBLEM_ROWS.get(i, f"{signal[i]:.8g}")
        lines.append(f"{i / SAMPLING_RATE:.6f},{amplitude}")
    return "\n".join(lines) + "\n"


def write_corrupted_sample(args: argparse.Namespace, targets: list[Path]) -> dict:
    csv_text = build_corrupted_csv()
    meta = build_metadata(
        CORRUPTED_SAMPLE_ID,
        "corrupted",
        "损坏样本",
        CORRUPTED_ROWS,
        "缺陷类型：amplitude 列含空值、非数字文本（null、N/A、--）与 NaN，共 "
        f"{len(CORRUPTED_PROBLEM_ROWS)} 个问题行；总行数 {CORRUPTED_ROWS}，明显短于 1024",
        "人为构造的损坏样本，用于验证数据校验与错误返回；不对应任何真实工况",
        source="手工构造",
        source_url="",
    )
    outputs = []
    for target in targets:
        csv_path = target / f"{CORRUPTED_SAMPLE_ID}.csv"
        write_text_atomic(csv_path, csv_text)
        outputs.append(csv_path)
        if target == SAMPLES_DIR:
            write_text_atomic(target / f"{CORRUPTED_SAMPLE_ID}.json", dump_metadata(meta))
    return {
        "file_id": "-",
        "condition": "corrupted",
        "sampling_rate": SAMPLING_RATE,
        "rotation_speed": "-",
        "window_size": CORRUPTED_ROWS,
        "paths": "; ".join(_rel(p) for p in outputs),
        "status": "成功（手工构造）",
    }


# ---------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载并切分 CWRU 轴承振动数据，生成 data/raw 数据与 data/samples 内置样例。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "固定切窗规则：采样率 12000 Hz、测点 drive_end、窗口长度 --window-size 点、窗口不重叠，\n"
            "第 k 个窗口起点 = k * --window-size，时间戳 t = i / 12000。\n"
            f"离线场景：--local-mat-dir 指向存放 97.mat / 105.mat / 130.mat / 118.mat 的目录。\n"
            f"数据集主页：{DATASET_HOME}"
        ),
    )
    parser.add_argument("--target", choices=("raw", "samples", "both"), default="both",
                        help="输出位置：raw=data/raw（默认 both）")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                        help=f"窗口长度（点数），默认 {DEFAULT_WINDOW_SIZE} = 1 秒")
    parser.add_argument("--window-index", type=int, default=DEFAULT_WINDOW_INDEX,
                        help=f"取第几个不重叠窗口（从 0 开始），默认 {DEFAULT_WINDOW_INDEX}")
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重新下载 .mat 并重算")
    parser.add_argument("--local-mat-dir", default=None,
                        help="本地已有 .mat 的目录，指定后跳过下载（离线场景）")
    parser.add_argument("--files", default=None,
                        help="只处理部分 fileId，逗号分隔，例如 97,105（默认全部）")
    args = parser.parse_args(argv)

    if args.window_size <= 0 or args.window_index < 0:
        parser.error("--window-size 必须为正整数，--window-index 必须 >= 0")
    if args.local_mat_dir and not Path(args.local_mat_dir).is_dir():
        parser.error(f"--local-mat-dir 目录不存在：{args.local_mat_dir}")

    if args.files:
        wanted = [item.strip() for item in args.files.replace(" ", ",").split(",") if item.strip()]
        unknown = [item for item in wanted if item not in FILE_BY_ID]
        if unknown:
            parser.error(f"未知 fileId：{', '.join(unknown)}；可选：{', '.join(FILE_BY_ID)}")
        args.specs = [FILE_BY_ID[item] for item in wanted]
    else:
        args.specs = list(CWRU_FILES)
    return args


def targets_for(args: argparse.Namespace) -> list[Path]:
    if args.target == "raw":
        return [RAW_DIR]
    if args.target == "samples":
        return [SAMPLES_DIR]
    return [RAW_DIR, SAMPLES_DIR]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = targets_for(args)

    print(f"输出目标：{', '.join(_rel(p) for p in outputs)}")
    print(f"切窗参数：window_size={args.window_size}  window_index={args.window_index}  "
          f"sampling_rate={SAMPLING_RATE}  sensor_position={SENSOR_POSITION}")
    if args.local_mat_dir:
        print(f"离线模式：使用本地 .mat 目录 {args.local_mat_dir}")
    else:
        print(f"数据源：{MAT_URL_TEMPLATE}")

    rows: list[dict] = []
    failures = 0
    for spec in args.specs:
        try:
            row = process_cwru_file(spec, args, outputs)
            print(f"[成功] fileId={spec.file_id} -> {row['paths']}")
        except PrepareError as exc:
            failures += 1
            print(f"[错误] fileId={spec.file_id}（{spec.condition}）处理失败：{exc}")
            print("       建议：先把该文件下载到本地目录，再用 "
                  f"--local-mat-dir <目录> --files {spec.file_id} 重跑；")
            print("       data/samples/ 下已有的内置样例不会被删除或覆盖，离线演示不受影响。")
            row = {
                "file_id": spec.file_id,
                "condition": spec.condition,
                "sampling_rate": SAMPLING_RATE,
                "rotation_speed": ROTATION_SPEED,
                "window_size": args.window_size,
                "paths": "-",
                "status": "失败",
            }
        rows.append(row)

    try:
        rows.append(write_corrupted_sample(args, outputs))
        print(f"[成功] {CORRUPTED_SAMPLE_ID} -> {rows[-1]['paths']}")
    except Exception as exc:
        failures += 1
        print(f"[错误] {CORRUPTED_SAMPLE_ID} 生成失败：{type(exc).__name__}: {exc}")
        rows.append({
            "file_id": "-", "condition": "corrupted", "sampling_rate": SAMPLING_RATE,
            "rotation_speed": "-", "window_size": CORRUPTED_ROWS, "paths": "-", "status": "失败",
        })

    print_summary(rows)

    if failures:
        print(f"\n共 {failures} 项失败；退出码 1。data/samples/ 下的内置样例保持原样，仍可用于离线端到端演示。")
        return 1
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())