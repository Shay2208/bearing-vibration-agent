"""轴承振动信号分析核心（纯 numpy/scipy，无网络依赖）。

对外接口：
  load_signal_csv          读取两列 / 单列振动 CSV
  validate_vibration_data  非侵入式数据校验（绝不抛异常，问题全部收集返回）
  extract_vibration_features  时域 + 频域特征提取
  compare_normal_abnormal  正常/异常特征对比（只描述现象，不做故障判定）
  shaft_frequency / bearing_fault_frequencies  纯计算函数

返回结构的字段名即接口契约，所有数值均为原生 float/int。
"""

from __future__ import annotations

import base64
import io
import math
import warnings
import wave
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------
ERROR_MISSING_COLUMN = "missing_column"
ERROR_EMPTY_DATA = "empty_data"
ERROR_INVALID_VALUE = "invalid_value"
ERROR_MISSING_VALUE = "missing_value"
ERROR_INSUFFICIENT_DATA = "insufficient_data"
ERROR_SAMPLING_RATE_MISMATCH = "sampling_rate_mismatch"
ERROR_INCOMPARABLE_LENGTH = "incomparable_length"
ERROR_FILE_NOT_FOUND = "file_not_found"
ERROR_FILE_FORMAT = "file_format"
ERROR_INVALID_ARGUMENT = "invalid_argument"


class VibrationError(Exception):
    """带错误类型的业务异常。属性：code、message、field（可为 None）。"""

    def __init__(self, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "field": self.field}


# CWRU 驱动端 6205-2RS JEM SKF 轴承几何参数（英寸 / 度）
BEARING_GEOMETRY = {
    "n_balls": 9,
    "ball_diameter": 0.3126,
    "pitch_diameter": 1.537,
    "contact_angle_deg": 0.0,
}

# 对比时使用的标量特征
_SCALAR_FEATURES = (
    "rms",
    "peak",
    "kurtosis",
    "crest_factor",
    "dominant_frequency",
    "dominant_amplitude",
)
_REQUIRED_FEATURE_KEYS = _SCALAR_FEATURES + ("band_energy", "spectrum_peaks")

# 采样率一致性容差（相对）
_SAMPLING_RATE_TOLERANCE = 0.01
# 判定"特征发生变化"的相对阈值
_CHANGE_THRESHOLD = 0.3
# 频谱峰值检测的相对 prominence 阈值
_PEAK_PROMINENCE_RATIO = 0.02
# 特征频率对齐所用的低频段上界（未提供转速时的回退值）
_LOW_BAND_MAX_HZ = 1000.0
# 特征频率带 = 最高特征频率的 _CHARACTERISTIC_BAND_ORDER 阶谐波；带内保留的谱峰数上限
_CHARACTERISTIC_BAND_ORDER = 3
_CHARACTERISTIC_PEAK_LIMIT = 20
# 主导特征频率的量级余量阈值：最高/次高低于该比值时不宣称"最突出"，改列并列候选
_DOMINANT_MARGIN_RATIO = 1.15
# 包络谱特征频率对齐：谐波阶次 / 相对容差 / 判定"无可识别成分"的能量下限
_ENVELOPE_HARMONICS = (1, 2, 3)
_ENVELOPE_TOLERANCE = 0.03
_ENVELOPE_MIN_ENERGY = 1e-9
# 包络谱能量保留位（能量可达 1e-18，6 位小数会被抹成 0）
_ENVELOPE_DIGITS = 12
# 基线检出判据：基线能量达到异常样本的该比例才算"基线中检出"
_BASELINE_DETECTED_RATIO = 0.01
# 曲线序列抽点（仅 include_series=True 时输出）：波形 min/max 分桶数、幅度谱点数、
# 包络谱裁剪上界与点数。包络谱裁剪是对齐 BPFI 3 阶（1797rpm 下约 486Hz）后留足余量。
_SERIES_WAVEFORM_BUCKETS = 600
_SERIES_SPECTRUM_POINTS = 600
_SERIES_ENVELOPE_MAX_HZ = 1000.0
_SERIES_ENVELOPE_POINTS = 400
# 音频导出：Int16 单声道 WAV，按峰值归一，留 2% 余量避免削波
_WAV_FULL_SCALE = 32767.0
_WAV_PEAK_RATIO = 0.98


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _err(code: str, message: str, field: str | None = None) -> dict:
    return {"code": code, "message": message, "field": field}


def _f(value, digits: int = 6):
    """转成原生 float（保留 digits 位小数），None / 非有限值返回 None。"""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _i(value) -> int:
    return int(value)


def _positive_number(value) -> float | None:
    """校验有限正数，失败返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _parse_float(text: str) -> tuple[bool, float | None]:
    if text == "":
        return False, None
    try:
        value = float(text)
    except ValueError:
        return False, None
    if not math.isfinite(value):
        return False, None
    return True, value


def _fmt_hz(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) < 1e-6 else f"{value:.1f}"


def _band_label(low: float, high: float) -> str:
    return f"{_fmt_hz(low)}-{_fmt_hz(high)}Hz"


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------
def _parse_header(cells: list[str]) -> tuple[str | None, int, list[dict]]:
    """返回 (模式, 列数, 错误列表)。模式：timestamp_amplitude / amplitude / None。"""
    if len(cells) == 2 and cells[0] == "timestamp" and cells[1] == "amplitude":
        return "timestamp_amplitude", 2, []
    if len(cells) == 1 and cells[0] == "amplitude":
        return "amplitude", 1, []
    if "amplitude" not in cells:
        return None, len(cells), [
            _err(ERROR_MISSING_COLUMN, "表头缺少 amplitude 列，需为 timestamp,amplitude 或 amplitude", "amplitude")
        ]
    return None, len(cells), [
        _err(ERROR_FILE_FORMAT, f"表头不合法：{'|'.join(cells)}，需为 timestamp,amplitude 或 amplitude", "header")
    ]


def _parse_rows(rows: list[str], mode: str, ncols: int) -> tuple[list[float], list[float] | None, list[dict], list[dict]]:
    values: list[float] = []
    stamps: list[float] = []
    errors: list[dict] = []
    warnings: list[dict] = []
    timestamp_broken = False

    for line_no, line in enumerate(rows, start=2):
        cells = [c.strip() for c in line.split(",")]
        if all(c == "" for c in cells):
            continue
        if len(cells) != ncols:
            errors.append(
                _err(ERROR_INVALID_VALUE, f"第 {line_no} 行有 {len(cells)} 列，应为 {ncols} 列", "row")
            )
            continue

        amp_raw = cells[-1]
        ok, amp = _parse_float(amp_raw)
        if not ok:
            if amp_raw == "":
                errors.append(_err(ERROR_MISSING_VALUE, f"第 {line_no} 行 amplitude 为空值", "amplitude"))
            else:
                errors.append(_err(ERROR_INVALID_VALUE, f"第 {line_no} 行 amplitude 不是有效数字：{amp_raw!r}", "amplitude"))
            continue
        values.append(amp)

        if mode == "timestamp_amplitude":
            t_raw = cells[0]
            ok_ts, stamp = _parse_float(t_raw)
            if ok_ts:
                stamps.append(stamp)
            elif not timestamp_broken:
                timestamp_broken = True
                warnings.append(
                    _err(ERROR_INVALID_VALUE, f"第 {line_no} 行 timestamp 不是有效数字：{t_raw!r}，已忽略时间戳", "timestamp")
                )

    if mode != "timestamp_amplitude" or timestamp_broken or len(stamps) != len(values):
        return values, None, errors, warnings
    return values, stamps, errors, warnings


def _load_signal(path) -> dict:
    """读取并解析 CSV，返回内部结构（错误与数据一起返回，不在校验维度上抛异常）。"""
    p = Path(str(path))
    if not p.is_file():
        raise VibrationError(ERROR_FILE_NOT_FOUND, f"信号文件不存在：{path}", "path")
    try:
        text = p.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise VibrationError(ERROR_FILE_FORMAT, f"信号文件无法读取：{exc}", "path")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise VibrationError(ERROR_EMPTY_DATA, "信号文件没有内容", "path")

    header_cells = [c.strip().lower() for c in lines[0].split(",")]
    mode, ncols, errors = _parse_header(header_cells)

    values: list[float] = []
    stamps: list[float] | None = None
    warnings: list[dict] = []
    if mode is not None:
        if len(lines) < 2:
            raise VibrationError(ERROR_EMPTY_DATA, "信号文件只有表头，没有数据行", "path")
        values, stamps, row_errors, warnings = _parse_rows(lines[1:], mode, ncols)
        errors = errors + row_errors
        if not values and not errors:
            raise VibrationError(ERROR_EMPTY_DATA, "信号文件没有有效数据行", "path")

    amplitude = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(stamps, dtype=np.float64) if stamps else None
    if timestamps is None and mode == "amplitude" and amplitude.size:
        # 单列模式：按行号生成时间戳（不用于推断采样率）
        timestamps = np.arange(amplitude.size, dtype=np.float64)

    inferred_rate = None
    if mode == "timestamp_amplitude" and timestamps is not None and timestamps.size >= 2:
        diffs = np.diff(timestamps)
        dt = float(np.median(diffs))
        if dt > 0:
            inferred_rate = 1.0 / dt

    return {
        "path": str(p),
        "amplitude": amplitude,
        "timestamps": timestamps,
        "n_samples": _i(amplitude.size),
        "inferred_rate": inferred_rate,
        "errors": errors,
        "warnings": warnings,
    }


def load_signal_csv(path) -> dict:
    """读取两列（timestamp,amplitude）或单列（amplitude）CSV。"""
    loaded = _load_signal(path)
    if loaded["errors"]:
        first = loaded["errors"][0]
        raise VibrationError(first["code"], first["message"], first["field"])
    return {
        "amplitude": loaded["amplitude"],
        "timestamps": loaded["timestamps"],
        "n_samples": loaded["n_samples"],
        "path": loaded["path"],
    }


# ---------------------------------------------------------------------------
# 数据校验
# ---------------------------------------------------------------------------
def validate_vibration_data(
    normal_path,
    abnormal_path,
    sampling_rate,
    sensor_position=None,
    min_samples: int = 1024,
) -> dict:
    """校验两个信号文件与采样率。绝不抛异常，所有问题收集进返回值。"""
    errors: list[dict] = []
    warnings: list[dict] = []
    loaded: dict[str, dict | None] = {}

    for key, path in (("normal", normal_path), ("abnormal", abnormal_path)):
        try:
            loaded[key] = _load_signal(path)
        except VibrationError as exc:
            loaded[key] = None
            errors.append(_err(exc.code, f"{key}：{exc.message}", exc.field))
            continue
        for item in loaded[key]["errors"]:
            errors.append(_err(item["code"], f"{key}：{item['message']}", item["field"]))
        for item in loaded[key]["warnings"]:
            warnings.append(_err(item["code"], f"{key}：{item['message']}", item["field"]))

    # 采样率合法性（入参为权威值）
    fs = _positive_number(sampling_rate)
    if fs is None:
        errors.append(_err(ERROR_INVALID_VALUE, f"采样率必须是有限正数，收到：{sampling_rate!r}", "sampling_rate"))

    # 时间戳推断采样率与入参是否一致
    rate_mismatch = False
    if fs is not None:
        for key in ("normal", "abnormal"):
            record = loaded.get(key)
            if record and record["inferred_rate"]:
                inferred = float(record["inferred_rate"])
                if abs(inferred - fs) / fs > _SAMPLING_RATE_TOLERANCE:
                    rate_mismatch = True
                    errors.append(
                        _err(
                            ERROR_SAMPLING_RATE_MISMATCH,
                            f"{key}：时间戳推断采样率约 {inferred:.1f}Hz，与入参 {_fmt_hz(fs)}Hz 不一致",
                            "sampling_rate",
                        )
                    )

    # 长度（文件本身读不出来时不再叠加长度类错误，避免错误列表噪声）
    counts = {key: (loaded[key]["n_samples"] if loaded[key] else 0) for key in ("normal", "abnormal")}
    both_usable = all(counts[key] > 0 for key in ("normal", "abnormal"))
    for key in ("normal", "abnormal"):
        if both_usable and counts[key] < int(min_samples):
            errors.append(
                _err(ERROR_INSUFFICIENT_DATA, f"{key}：仅 {counts[key]} 点，少于最少要求 {int(min_samples)} 点", "n_samples")
            )

    # 可比性
    same_length = counts["normal"] == counts["abnormal"]
    both_long_enough = counts["normal"] >= int(min_samples) and counts["abnormal"] >= int(min_samples)
    comparable = same_length or both_long_enough
    if both_usable and not same_length:
        if comparable:
            warnings.append(
                _err(ERROR_INCOMPARABLE_LENGTH, f"两文件长度不同（{counts['normal']} vs {counts['abnormal']}），均在 {int(min_samples)} 点以上，按统计量对比", "n_samples")
            )
        else:
            errors.append(
                _err(ERROR_INCOMPARABLE_LENGTH, f"两文件长度不同（{counts['normal']} vs {counts['abnormal']}）且不足 {int(min_samples)} 点，无法对比", "n_samples")
            )

    file_ok = {key: loaded.get(key) is not None for key in ("normal", "abnormal")}
    format_ok = all(file_ok.values()) and not any(e["code"] == ERROR_FILE_FORMAT for e in errors)
    columns_ok = all(file_ok.values()) and not any(e["code"] == ERROR_MISSING_COLUMN for e in errors)
    values_ok = (
        all(file_ok.values())
        and all(counts[key] > 0 for key in ("normal", "abnormal"))
        and not any(e["code"] in (ERROR_MISSING_VALUE, ERROR_INVALID_VALUE) for e in errors)
    )
    length_ok = all(counts[key] >= int(min_samples) for key in ("normal", "abnormal"))
    rate_ok = fs is not None and not rate_mismatch

    checks = {
        "format": {
            "passed": format_ok,
            "detail": "两个文件表头均为 timestamp,amplitude 或 amplitude" if format_ok else "存在无法读取或表头不合法的文件",
        },
        "columns": {
            "passed": columns_ok,
            "detail": "amplitude 列存在" if columns_ok else "缺少 amplitude 列",
        },
        "values": {
            "passed": values_ok,
            "detail": "无缺失值/非法值" if values_ok else "存在缺失值或非法值",
        },
        "length": {
            "passed": length_ok,
            "detail": f"normal={counts['normal']} 点，abnormal={counts['abnormal']} 点，最少要求 {int(min_samples)} 点",
        },
        "sampling_rate": {
            "passed": rate_ok,
            "detail": f"采样率 {_fmt_hz(fs)}Hz 有效且与时间戳推断一致" if rate_ok else "采样率非法或与文件时间戳推断不一致",
        },
        "comparability": {
            "passed": comparable,
            "detail": "两样本长度一致" if same_length else ("长度不同但均超过最少点数，可按统计量对比" if comparable else "长度不同且点数不足，无法对比"),
        },
    }
    if sensor_position:
        checks["format"]["detail"] = f"{checks['format']['detail']}（测点：{sensor_position}）"

    blocking = [e for e in errors if e["code"] != ERROR_INCOMPARABLE_LENGTH]
    return {
        "valid": len(blocking) == 0,
        "comparable": comparable,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "normal": {
            "path": str(normal_path),
            "n_samples": _i(counts["normal"]),
            "sampling_rate": _f(fs, 6) if fs is not None else None,
        },
        "abnormal": {
            "path": str(abnormal_path),
            "n_samples": _i(counts["abnormal"]),
            "sampling_rate": _f(fs, 6) if fs is not None else None,
        },
    }


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------
def _resolve_band_edges(band_edges, fs: float) -> list[float]:
    if band_edges is None:
        nyquist = fs / 2.0
        return [nyquist * i / 5.0 for i in range(6)]
    edges: list[float] = []
    pairs = all(isinstance(item, (list, tuple)) and len(item) == 2 for item in band_edges)
    if pairs:
        for low, high in band_edges:
            if not edges:
                edges.append(float(low))
            edges.append(float(high))
    else:
        edges = [float(v) for v in band_edges]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise VibrationError(ERROR_INVALID_ARGUMENT, f"band_edges 必须是递增的频带边界：{band_edges!r}", "band_edges")
    return edges


def _band_energy(magnitude: np.ndarray, freqs: np.ndarray, edges: list[float]) -> dict:
    result: dict[str, float] = {}
    last = len(edges) - 2
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        mask = (freqs >= low) & (freqs <= high if index == last else freqs < high)
        result[_band_label(low, high)] = _f(float(np.sum(magnitude[mask] ** 2)))
    return result


def _envelope_magnitude(band_signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """包络谱：|hilbert(x)| 去均值后加汉宁窗 rFFT，归一方式与幅值谱一致。"""
    n = int(band_signal.size)
    envelope_axis = np.abs(sp_signal.hilbert(band_signal))
    envelope_axis = envelope_axis - float(np.mean(envelope_axis))
    window = sp_signal.windows.hann(n, sym=False)
    magnitude = np.abs(np.fft.rfft(envelope_axis * window)) * (2.0 / n)
    magnitude[0] /= 2.0
    if n % 2 == 0:
        magnitude[-1] /= 2.0
    return magnitude, np.fft.rfftfreq(n, d=1.0 / fs)


def _local_maxima(magnitude: np.ndarray, freqs: np.ndarray) -> list[int]:
    """与幅值谱相同的局部极大值检测（排除直流）。"""
    max_amp = float(np.max(magnitude)) if magnitude.size else 0.0
    if max_amp <= 0:
        return []
    indices, _ = sp_signal.find_peaks(magnitude, prominence=max_amp * _PEAK_PROMINENCE_RATIO)
    return [int(i) for i in indices if freqs[i] > 0]


def _bandpass(x: np.ndarray, fs: float, envelope_band) -> tuple[np.ndarray, list[float], str]:
    """返回 (带通后信号, 实际带宽, band_rule 说明)。范围越界时退化为不做带通。"""
    nyquist = fs / 2.0
    if envelope_band is not None:
        try:
            band = [float(v) for v in envelope_band]
        except (TypeError, ValueError):
            raise VibrationError(ERROR_INVALID_ARGUMENT, f"envelope_band 必须是两个数字：[lo, hi]，收到：{envelope_band!r}", "envelope_band")
        if len(band) != 2 or not all(math.isfinite(v) for v in band) or band[0] <= 0 or band[1] <= band[0]:
            raise VibrationError(ERROR_INVALID_ARGUMENT, f"envelope_band 必须是递增的正数区间：[lo, hi]，收到：{envelope_band!r}", "envelope_band")
        low, high = band
        rule = f"使用入参 envelope_band=[{_fmt_hz(low)}, {_fmt_hz(high)}]Hz 作为共振带"
    else:
        low, high = 0.33 * nyquist, 0.83 * nyquist
        rule = "取奈奎斯特频率的 0.33~0.83 倍作为共振带（12kHz 采样下约 2~5kHz）"

    if low <= 0 or high >= nyquist:
        return x, [0.0, float(nyquist)], f"{rule}；该范围超出 (0, Nyquist)，退化为不做带通"

    b, a = sp_signal.butter(4, [low / nyquist, high / nyquist], btype="band")
    return sp_signal.filtfilt(b, a, x), [_f(low, 4), _f(high, 4)], rule


def _envelope_analysis(x: np.ndarray, fs: float, n: int, rotation_speed, envelope_band) -> dict:
    """包络解调 + 特征频率能量/峰位对齐（只输出频率证据，不做故障判定）。"""
    faults = bearing_fault_frequencies(rotation_speed)
    band_signal, band_hz, band_rule = _bandpass(x, fs, envelope_band)
    magnitude, freqs = _envelope_magnitude(band_signal, fs)
    resolution = fs / n

    characteristic_frequencies = {
        "shaft": _f(faults["shaft_frequency"], 4),
        "ftf": _f(faults["ftf"], 4),
        "bsf": _f(faults["bsf"], 4),
        # 滚动体缺陷的知识库判据落在 2×BSF，单列出来避免与 1×BSF 口径混淆
        "bsf_2x": _f(2.0 * faults["bsf"], 4),
        "bpfo": _f(faults["bpfo"], 4),
        "bpfi": _f(faults["bpfi"], 4),
    }

    def window_sum(target: float) -> float:
        tolerance = max(_ENVELOPE_TOLERANCE * target, resolution)
        mask = (freqs >= target - tolerance) & (freqs <= target + tolerance)
        return float(np.sum(magnitude[mask] ** 2))

    candidate_indices = _local_maxima(magnitude, freqs)
    characteristic_energy: dict[str, float] = {}
    characteristic_peaks: list[dict] = []
    for label in ("ftf", "bsf", "bpfo", "bpfi"):
        base = float(characteristic_frequencies[label])
        characteristic_energy[label] = _f(sum(window_sum(base * h) for h in _ENVELOPE_HARMONICS), _ENVELOPE_DIGITS)
        for order in _ENVELOPE_HARMONICS:
            target = base * order
            tolerance = max(_ENVELOPE_TOLERANCE * target, resolution)
            inside = [i for i in candidate_indices if abs(float(freqs[i]) - target) <= tolerance]
            if not inside:
                continue
            index = max(inside, key=lambda i: magnitude[i])
            characteristic_peaks.append(
                {
                    "label": label.upper(),
                    "harmonic_order": _i(order),
                    "frequency": _f(target, 4),
                    "nearest_peak": _f(float(freqs[index]), 4),
                    "deviation_pct": _f(abs(float(freqs[index]) - target) / target * 100.0, 2),
                    "peak_amplitude": _f(float(magnitude[index]), _ENVELOPE_DIGITS),
                }
            )

    by_amplitude = sorted(candidate_indices, key=lambda i: magnitude[i], reverse=True)
    envelope_peaks = [
        {"frequency": _f(float(freqs[i]), 4), "amplitude": _f(float(magnitude[i]), _ENVELOPE_DIGITS)}
        for i in sorted(by_amplitude[:10], key=lambda i: freqs[i])
    ]

    return {
        "band_hz": band_hz,
        "band_rule": band_rule,
        "characteristic_frequencies": characteristic_frequencies,
        "characteristic_energy": characteristic_energy,
        "characteristic_peaks": characteristic_peaks,
        "envelope_peaks": envelope_peaks,
    }


# ---------------------------------------------------------------------------
# 曲线序列抽点与音频（仅 include_series=True 时输出，供前端手绘 SVG 与试听）
# ---------------------------------------------------------------------------
def _bucket_bounds(n: int, count: int) -> list[tuple[int, int]]:
    """把 n 个采样点均分成 count 个非空桶（count <= n）。"""
    return [((i * n) // count, ((i + 1) * n) // count) for i in range(count)]


def _pick_series(magnitude: np.ndarray, freqs: np.ndarray, points: int) -> dict:
    """谱曲线抽点：每桶取桶内最大幅值对应的点（峰值保序），避免特征频率峰被抽掉。"""
    n = int(magnitude.size)
    points = min(int(points), n)
    if points <= 0:
        return {"points": 0, "frequency": [], "amplitude": []}
    axes: list[float] = []
    values: list[float] = []
    for start, stop in _bucket_bounds(n, points):
        index = start + int(np.argmax(magnitude[start:stop]))
        axes.append(_f(float(freqs[index]), 4))
        values.append(_f(float(magnitude[index]), 6))
    return {"points": points, "frequency": axes, "amplitude": values}


def _min_max_series(x: np.ndarray, buckets: int) -> dict:
    """时域波形 min/max 分桶降采样：保留冲击包络形状，桶数不超过采样点数。"""
    n = int(x.size)
    buckets = min(int(buckets), n)
    if buckets <= 0:
        return {"buckets": 0, "min": [], "max": []}
    lows: list[float] = []
    highs: list[float] = []
    for start, stop in _bucket_bounds(n, buckets):
        segment = x[start:stop]
        lows.append(_f(float(np.min(segment)), 6))
        highs.append(_f(float(np.max(segment)), 6))
    return {"buckets": buckets, "min": lows, "max": highs}


def _envelope_series(x: np.ndarray, fs: float, envelope_band) -> dict:
    """包络谱曲线：复用包络谱算法，裁到 _SERIES_ENVELOPE_MAX_HZ 内再抽点。"""
    band_signal, _, _ = _bandpass(x, fs, envelope_band)
    magnitude, freqs = _envelope_magnitude(band_signal, fs)
    keep = freqs <= _SERIES_ENVELOPE_MAX_HZ
    return _pick_series(magnitude[keep], freqs[keep], _SERIES_ENVELOPE_POINTS)


def _wav_bytes(x: np.ndarray, fs: float) -> bytes:
    """去均值波形 → Int16 单声道 WAV；按全局峰值归一（保留 2% 余量）避免削波。"""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    scale = (_WAV_PEAK_RATIO * _WAV_FULL_SCALE / peak) if peak > 0 else 1.0
    pcm = np.clip(np.round(x * scale), -_WAV_FULL_SCALE - 1.0, _WAV_FULL_SCALE).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(round(fs)))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def extract_vibration_features(
    csv_path, sampling_rate, band_edges=None, rotation_speed=None, envelope_band=None, include_series=False
) -> dict:
    """提取时域与频域特征。输入非法时抛 VibrationError，由 MCP 层转成结构化错误。

    传入 rotation_speed 时额外做包络解调分析（共振带 → |hilbert| → 包络谱 → 特征频率对齐）。
    include_series=True 时额外输出 series 块：降采样的时域波形 / 幅度谱 / 包络谱曲线，
    以及按峰值归一的 Int16 WAV（base64）。曲线是为前端手绘 SVG 抽的点，不参与任何判据计算。
    """
    fs = _positive_number(sampling_rate)
    if fs is None:
        raise VibrationError(ERROR_INVALID_ARGUMENT, f"采样率必须是有限正数，收到：{sampling_rate!r}", "sampling_rate")

    signal_data = load_signal_csv(csv_path)
    x = np.asarray(signal_data["amplitude"], dtype=np.float64)
    n = int(x.size)
    if n < 2:
        raise VibrationError(ERROR_INSUFFICIENT_DATA, f"有效数据点不足（{n}），无法计算频谱", "amplitude")

    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kurtosis = float(sp_stats.kurtosis(x, fisher=False, bias=False))
    if not math.isfinite(kurtosis):
        kurtosis = 0.0
    crest_factor = peak / rms if rms > 0 else 0.0

    # 加汉宁窗 rFFT，单边幅值谱（2/N 归一，直流与奈奎斯特不乘 2）
    window = sp_signal.windows.hann(n, sym=False)
    spectrum = np.fft.rfft(x * window)
    magnitude = np.abs(spectrum) * (2.0 / n)
    magnitude[0] /= 2.0
    if n % 2 == 0:
        magnitude[-1] /= 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    dominant_index = int(np.argmax(magnitude))
    dominant_frequency = float(freqs[dominant_index])
    dominant_amplitude = float(magnitude[dominant_index])

    edges = _resolve_band_edges(band_edges, fs)
    band_energy = _band_energy(magnitude, freqs, edges)

    peaks: list[dict] = []
    max_amp = float(np.max(magnitude)) if magnitude.size else 0.0
    if max_amp > 0:
        indices, _ = sp_signal.find_peaks(magnitude, prominence=max_amp * _PEAK_PROMINENCE_RATIO)
        indices = [int(i) for i in indices if freqs[i] > 0]
        by_amplitude = sorted(indices, key=lambda i: magnitude[i], reverse=True)

        # 特征带（最高特征频率的 3 阶谐波，1797rpm 下约 486Hz）内改用带内最大幅值作
        # prominence 基准：高通全谱基准会被高频共振峰抬得很高，BPFO/BPFI/BSF 的谐波峰
        # 排不进前 5 名，fault_frequency_matches 实测为空。带内基准 + 幅值前 N 才覆盖得到。
        band_high = _LOW_BAND_MAX_HZ
        if rotation_speed is not None:
            faults = bearing_fault_frequencies(rotation_speed)
            band_high = _CHARACTERISTIC_BAND_ORDER * max(
                faults["bpfi"], faults["bpfo"], faults["bsf"], faults["ftf"]
            )
        band_high = min(band_high, float(freqs[-1]))
        in_band = freqs <= band_high
        band_max = float(np.max(magnitude[in_band])) if bool(in_band.any()) else 0.0
        band_peaks: list[int] = []
        if band_max > 0:
            local, _ = sp_signal.find_peaks(magnitude, prominence=band_max * _PEAK_PROMINENCE_RATIO)
            band_peaks = [int(i) for i in local if freqs[i] > 0 and freqs[i] <= band_high]
        band_peaks.sort(key=lambda i: magnitude[i], reverse=True)

        selected = sorted(
            set(by_amplitude[:5] + band_peaks[:_CHARACTERISTIC_PEAK_LIMIT]),
            key=lambda i: freqs[i],
        )
        peaks = [
            {"frequency": _f(float(freqs[i]), 4), "amplitude": _f(float(magnitude[i]))}
            for i in selected
        ]

    envelope = _envelope_analysis(x, fs, n, rotation_speed, envelope_band) if rotation_speed is not None else None

    series = None
    if include_series:
        series = {
            "sampling_rate": _f(fs, 6),
            "duration_seconds": _f(n / fs, 6),
            "waveform": _min_max_series(x, _SERIES_WAVEFORM_BUCKETS),
            "spectrum": _pick_series(magnitude, freqs, _SERIES_SPECTRUM_POINTS),
            "envelope": _envelope_series(x, fs, envelope_band) if rotation_speed is not None else None,
            "audio": {
                "format": "wav",
                "encoding": "pcm_s16le",
                "sampling_rate": _f(fs, 6),
                "duration_seconds": _f(n / fs, 6),
                "normalized": True,
                "wav_base64": base64.b64encode(_wav_bytes(x, fs)).decode("ascii"),
            },
        }

    return {
        "sampling_rate": _f(fs, 6),
        "n_samples": _i(n),
        "duration_seconds": _f(n / fs, 6),
        "rms": _f(rms),
        "peak": _f(peak),
        "kurtosis": _f(kurtosis),
        "crest_factor": _f(crest_factor),
        "dominant_frequency": _f(dominant_frequency, 4),
        "dominant_amplitude": _f(dominant_amplitude),
        "band_energy": band_energy,
        "spectrum_peaks": peaks,
        "envelope": envelope,
        "series": series,
    }


# ---------------------------------------------------------------------------
# 轴承特征频率
# ---------------------------------------------------------------------------
def shaft_frequency(rotation_speed) -> float:
    """转频（Hz）= 转速(rpm)/60。"""
    rpm = _positive_number(rotation_speed)
    if rpm is None:
        raise VibrationError(ERROR_INVALID_ARGUMENT, f"转速必须是有限正数，收到：{rotation_speed!r}", "rotation_speed")
    return rpm / 60.0


def bearing_fault_frequencies(
    rotation_speed,
    n_balls: int = BEARING_GEOMETRY["n_balls"],
    ball_diameter: float = BEARING_GEOMETRY["ball_diameter"],
    pitch_diameter: float = BEARING_GEOMETRY["pitch_diameter"],
    contact_angle_deg: float = BEARING_GEOMETRY["contact_angle_deg"],
) -> dict:
    """按 CWRU 6205-2RS JEM SKF 几何参数计算 BPFO/BPFI/BSF/FTF。"""
    fr = shaft_frequency(rotation_speed)
    d_over_d = float(ball_diameter) / float(pitch_diameter)
    cos_a = math.cos(math.radians(float(contact_angle_deg)))
    ratio = d_over_d * cos_a
    half_ball_count = float(n_balls) / 2.0

    bpfo = half_ball_count * fr * (1.0 - ratio)
    bpfi = half_ball_count * fr * (1.0 + ratio)
    bsf = float(pitch_diameter) / (2.0 * float(ball_diameter)) * fr * (1.0 - ratio**2)
    ftf = fr / 2.0 * (1.0 - ratio)

    return {
        "rotation_speed": _f(float(rotation_speed), 6),
        "shaft_frequency": _f(fr, 4),
        "bpfo": _f(bpfo, 4),
        "bpfi": _f(bpfi, 4),
        "bsf": _f(bsf, 4),
        "ftf": _f(ftf, 4),
        "geometry": {
            "n_balls": _i(n_balls),
            "ball_diameter": _f(ball_diameter, 6),
            "pitch_diameter": _f(pitch_diameter, 6),
            "contact_angle_deg": _f(contact_angle_deg, 6),
        },
    }


def _match_fault_frequencies(abnormal_features: dict, fault_frequencies: dict) -> list[dict]:
    """把 4 个特征频率及其 2/3 次谐波与异常样本频谱峰对齐。"""
    peaks = abnormal_features.get("spectrum_peaks") or []
    if not peaks:
        return []
    peak_freqs = np.asarray([float(p["frequency"]) for p in peaks], dtype=np.float64)
    peak_amps = np.asarray([float(p["amplitude"]) for p in peaks], dtype=np.float64)

    fs = _positive_number(abnormal_features.get("sampling_rate")) or 0.0
    n = int(abnormal_features.get("n_samples") or 0)
    resolution = fs / n if fs > 0 and n > 0 else 1.0

    matches: list[dict] = []
    for label in ("BPFO", "BPFI", "BSF", "FTF"):
        base = float(fault_frequencies[label.lower()])
        for order in (1, 2, 3):
            target = base * order
            tolerance = max(0.02 * target, resolution)
            distances = np.abs(peak_freqs - target)
            index = int(np.argmin(distances))
            if float(distances[index]) > tolerance:
                continue
            matches.append(
                {
                    "label": label,
                    "frequency": _f(target, 4),
                    "nearest_peak": _f(float(peak_freqs[index]), 4),
                    "deviation_pct": _f(float(distances[index]) / target * 100.0, 2),
                    "peak_amplitude": _f(float(peak_amps[index])),
                    "harmonic_order": _i(order),
                }
            )
    return matches


# ---------------------------------------------------------------------------
# 正常 / 异常对比
# ---------------------------------------------------------------------------
def _require_features(features, name: str) -> dict:
    if not isinstance(features, dict):
        raise VibrationError(ERROR_INVALID_ARGUMENT, f"{name} 必须是特征字典，收到：{type(features).__name__}", name)
    for key in _REQUIRED_FEATURE_KEYS:
        if key not in features:
            raise VibrationError(ERROR_MISSING_VALUE, f"{name} 缺少字段：{key}", key)
    return features


def _change_block(normal_value, abnormal_value, with_ratio: bool = True) -> dict:
    """ratio 为倍数 abnormal/normal；normal 为 0（或无意义特征）时置 None。"""
    normal = float(normal_value)
    abnormal = float(abnormal_value)
    delta = abnormal - normal
    ratio = None
    if with_ratio and normal != 0.0:
        candidate = abnormal / normal
        if math.isfinite(candidate):
            ratio = _f(candidate, 4)
    return {"normal": _f(normal, 4), "abnormal": _f(abnormal, 4), "delta": _f(delta, 4), "ratio": ratio}


def _trend(ratio: float | None) -> str:
    if ratio is None:
        return "变化"
    return "上升" if ratio >= 1.0 else "下降"


def _percent(ratio: float | None) -> str:
    """相对变化百分比 = (ratio - 1) * 100。"""
    return f"{abs(ratio - 1.0) * 100:.0f}%" if ratio is not None else "未知"


def _build_evidence(feature_changes, band_energy_changes, frequency_change, envelope_evidence=()) -> list[str]:
    evidence: list[str] = []

    rms = feature_changes["rms"]
    evidence.append(f"RMS 由 {rms['normal']:.4f} 变为 {rms['abnormal']:.4f}，{_trend(rms['ratio'])} {_percent(rms['ratio'])}")

    kurt = feature_changes["kurtosis"]
    # 峭度变化同样按显著性阈值措辞：3% 的波动不足以宣称"冲击成分增强"
    if kurt["ratio"] is not None and abs(float(kurt["ratio"]) - 1.0) >= _CHANGE_THRESHOLD:
        kurt_trend = "冲击成分增强" if kurt["ratio"] >= 1.0 else "冲击成分减弱"
    else:
        kurt_trend = "变化不显著"
    evidence.append(f"峭度由 {kurt['normal']:.2f} 变为 {kurt['abnormal']:.2f}，{kurt_trend}")

    # 包络谱证据紧跟趋势特征，避免被 6 条上限截掉
    evidence.extend(envelope_evidence)

    # 取能量增量最明显的频带（比相对变化更稳健，避免极小基线的百分比失真）
    ranked_bands = [
        (label, block) for label, block in band_energy_changes.items() if block["ratio"] is not None
    ]
    if ranked_bands:
        label, block = max(ranked_bands, key=lambda item: abs(item[1]["delta"]))
        if block["normal"] == 0.0 and block["abnormal"] > 0:
            # 正常样本该频带近似为 0，倍数/百分比会失真，只报绝对量
            evidence.append(f"{label} 频带能量由接近 0 变为 {block['abnormal']:.6f}（正常样本该频带无明显成分）")
        else:
            evidence.append(
                f"{label} 频带能量由 {block['normal']:.6f} 变为 {block['abnormal']:.6f}，{_trend(block['ratio'])} {_percent(block['ratio'])}"
            )

    shift = frequency_change["shift_hz"]
    if frequency_change["normal_dominant"] != frequency_change["abnormal_dominant"]:
        evidence.append(
            f"主频由 {frequency_change['normal_dominant']:.1f}Hz 迁移至 {frequency_change['abnormal_dominant']:.1f}Hz，偏移 {shift:+.1f}Hz"
        )
    else:
        evidence.append(f"主频保持 {frequency_change['normal_dominant']:.1f}Hz 未迁移")

    peak = feature_changes["peak"]
    if peak["ratio"] is not None and abs(peak["ratio"] - 1.0) >= _CHANGE_THRESHOLD:
        evidence.append(f"峰值由 {peak['normal']:.4f} 变为 {peak['abnormal']:.4f}，{_trend(peak['ratio'])} {_percent(peak['ratio'])}")

    crest = feature_changes["crest_factor"]
    if crest["ratio"] is not None and abs(crest["ratio"] - 1.0) >= _CHANGE_THRESHOLD:
        crest_trend = "瞬时冲击相对水平升高" if crest["ratio"] >= 1.0 else "瞬时冲击相对水平降低"
        evidence.append(f"波峰因子由 {crest['normal']:.2f} 变为 {crest['abnormal']:.2f}，{crest_trend}")

    if len(evidence) < 3:
        dom = feature_changes["dominant_amplitude"]
        evidence.append(
            f"主频幅值由 {dom['normal']:.4f} 变为 {dom['abnormal']:.4f}，整体振动水平未见明显单点突变"
        )
    return evidence[:6]


_CHARACTERISTIC_LABELS = ("bpfo", "bpfi", "bsf", "ftf")


def _characteristic_comparison(normal: dict, abnormal: dict) -> tuple[dict | None, dict | None, list[str]]:
    """对比包络谱特征频率能量，返回 (changes, dominant, evidence)。任一侧无 envelope 时返回 (None, None, [])。"""
    envelope_normal = normal.get("envelope") or None
    envelope_abnormal = abnormal.get("envelope") or None
    if not envelope_normal or not envelope_abnormal:
        return None, None, []

    normal_energy_map = envelope_normal.get("characteristic_energy") or {}
    abnormal_energy_map = envelope_abnormal.get("characteristic_energy") or {}
    frequency_map = envelope_abnormal.get("characteristic_frequencies") or {}

    changes: dict[str, dict] = {}
    for label in _CHARACTERISTIC_LABELS:
        normal_energy = float(normal_energy_map.get(label, 0.0))
        abnormal_energy = float(abnormal_energy_map.get(label, 0.0))
        # 基线未检出该成分时倍数无意义，只报绝对量
        detected = normal_energy >= _BASELINE_DETECTED_RATIO * abnormal_energy
        changes[label] = {
            "normal": _f(normal_energy, _ENVELOPE_DIGITS),
            "abnormal": _f(abnormal_energy, _ENVELOPE_DIGITS),
            "delta": _f(abnormal_energy - normal_energy, _ENVELOPE_DIGITS),
            "ratio": _f(abnormal_energy / normal_energy, 4) if (detected and normal_energy != 0.0) else None,
            "detected_in_baseline": bool(detected),
        }

    ranked = sorted(
        _CHARACTERISTIC_LABELS, key=lambda item: float(abnormal_energy_map.get(item, 0.0)), reverse=True
    )
    best_label = ranked[0]
    best_energy = float(abnormal_energy_map.get(best_label, 0.0))
    dominant = None
    evidence: list[str] = []
    if best_energy < _ENVELOPE_MIN_ENERGY:
        # 各特征频率处都没有可识别的包络成分，不强行挑一个
        return changes, None, evidence

    block = changes[best_label]
    target = float(frequency_map.get(best_label, 0.0))
    baseline_note = "，且基线中未检出该成分" if not block["detected_in_baseline"] else f"，基线对应能量 {block['normal']:.6f}"

    # 量级余量判据：最高与次高能量差在噪声量级时不给排他性主导项，改为并列候选。
    # 7% 量级的差距不足以支撑"最突出"的表述，也不该把检索查询压向单一条目。
    second_label = ranked[1]
    second_energy = float(abnormal_energy_map.get(second_label, 0.0))
    margin = best_energy / second_energy if second_energy > 0 else None
    tied: list[dict] = []
    if margin is not None and margin < _DOMINANT_MARGIN_RATIO:
        for label in (best_label, second_label):
            family = changes[label]
            tied.append(
                {
                    "label": label.upper(),
                    "frequency_hz": _f(float(frequency_map.get(label, 0.0)), 4),
                    "abnormal_energy": family["abnormal"],
                    "ratio": family["ratio"],
                    "detected_in_baseline": family["detected_in_baseline"],
                }
            )
        summary = (
            f"异常样本包络谱中 {tied[0]['label']}（{tied[0]['frequency_hz']:.1f}Hz）能量 {best_energy:.3e} "
            f"与 {tied[1]['label']}（{tied[1]['frequency_hz']:.1f}Hz）能量 {second_energy:.3e} 接近"
            f"（相差 {(margin - 1.0) * 100:.1f}%，低于量级余量阈值 {(_DOMINANT_MARGIN_RATIO - 1) * 100:.0f}%），"
            f"不足以区分主次"
        )
    else:
        summary = (
            f"异常样本包络谱中 {best_label.upper()}（{target:.1f}Hz）及其 2、3 阶谐波频带能量为各特征频率中最高"
            f"（{best_energy:.4f}）{baseline_note}"
        )

    dominant = {
        "label": best_label.upper(),
        "frequency_hz": _f(target, 4),
        "abnormal_energy": _f(best_energy),
        "normal_energy": block["normal"],
        "ratio": block["ratio"],
        "detected_in_baseline": block["detected_in_baseline"],
        "ambiguous": bool(tied),
        "margin_ratio": _f(margin, 4) if margin is not None else None,
        "tied_candidates": tied,
        "reason": summary,
    }
    evidence.append(summary)

    peaks = envelope_abnormal.get("characteristic_peaks") or []
    if peaks:
        top = max(peaks, key=lambda item: float(item.get("peak_amplitude") or 0.0))
        evidence.append(
            f"包络谱中 {top['label']}（{float(top['frequency']):.1f}Hz）第 {top['harmonic_order']} 阶谐波处出现明显峰"
            f"（{float(top['nearest_peak']):.1f}Hz，偏差 {float(top['deviation_pct']):.2f}%）"
        )
    return changes, dominant, evidence


def compare_normal_abnormal(normal_features, abnormal_features, rotation_speed=None) -> dict:
    """对比正常/异常特征，输出结构化诊断依据（只描述现象，不做故障判定）。"""
    normal = _require_features(normal_features, "normal_features")
    abnormal = _require_features(abnormal_features, "abnormal_features")

    feature_changes = {
        key: _change_block(normal[key], abnormal[key], with_ratio=(key != "dominant_frequency"))
        for key in _SCALAR_FEATURES
    }

    band_energy_changes: dict[str, dict] = {}
    normal_bands = normal.get("band_energy") or {}
    abnormal_bands = abnormal.get("band_energy") or {}
    for label in list(normal_bands.keys()) + [k for k in abnormal_bands if k not in normal_bands]:
        block = _change_block(normal_bands.get(label, 0.0), abnormal_bands.get(label, 0.0))
        band_energy_changes[label] = {"normal": block["normal"], "abnormal": block["abnormal"], "delta": block["delta"], "ratio": block["ratio"]}

    frequency_change = {
        "normal_dominant": _f(normal["dominant_frequency"], 4),
        "abnormal_dominant": _f(abnormal["dominant_frequency"], 4),
        "shift_hz": _f(float(abnormal["dominant_frequency"]) - float(normal["dominant_frequency"]), 4),
    }

    changed_features = [
        key
        for key, block in feature_changes.items()
        if block["ratio"] is not None and abs(block["ratio"] - 1.0) >= _CHANGE_THRESHOLD
    ]

    characteristic_energy_changes, dominant_characteristic, envelope_evidence = _characteristic_comparison(normal, abnormal)

    evidence = _build_evidence(feature_changes, band_energy_changes, frequency_change, envelope_evidence)

    fault_frequencies = None
    fault_frequency_matches = None
    if rotation_speed is not None:
        fault_frequencies = bearing_fault_frequencies(rotation_speed)
        fault_frequency_matches = _match_fault_frequencies(abnormal, fault_frequencies)

    return {
        "feature_changes": feature_changes,
        "band_energy_changes": band_energy_changes,
        "frequency_change": frequency_change,
        "changed_features": changed_features,
        "evidence": evidence,
        "fault_frequencies": fault_frequencies,
        "fault_frequency_matches": fault_frequency_matches,
        "characteristic_energy_changes": characteristic_energy_changes,
        "dominant_characteristic": dominant_characteristic,
    }