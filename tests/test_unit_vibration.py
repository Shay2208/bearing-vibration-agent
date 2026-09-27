"""SubTask 9.1：振动分析单元测试。

覆盖：正常数据读取、缺失值处理、数据长度不一致、采样率不一致/不可比较、
特征计算、频谱峰值检测。全部针对 app/mcp/vibration.py 的真实实现，
期望值由测试侧独立用 numpy/scipy 重算得到，不依赖被测代码自身。
"""

from __future__ import annotations

import csv
import re

import numpy as np
import pytest
from scipy import signal as sp_signal
from scipy import stats as sp_stats

from app.mcp import vibration
from tests.helpers import SAMPLES, sample_path, write_signal_csv

FS = 12000.0
EXPECTED_BANDS = ["0-1200Hz", "1200-2400Hz", "2400-3600Hz", "3600-4800Hz", "4800-6000Hz"]


# ---------------------------------------------------------------------------
# 测试侧独立重算（与被测实现无关的第二套算法）
# ---------------------------------------------------------------------------
def read_amplitude(path) -> np.ndarray:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    return np.asarray([float(row[-1]) for row in rows[1:] if row and row[-1].strip()], dtype=np.float64)


def reference_spectrum(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """加汉宁窗的单边幅值谱（2/N 归一，直流与奈奎斯特不乘 2）。"""
    n = int(values.size)
    window = sp_signal.windows.hann(n, sym=False)
    magnitude = np.abs(np.fft.rfft(values * window)) * (2.0 / n)
    magnitude[0] /= 2.0
    if n % 2 == 0:
        magnitude[-1] /= 2.0
    return magnitude, np.fft.rfftfreq(n, d=1.0 / fs)


# ---------------------------------------------------------------------------
# 1. 正常数据读取
# ---------------------------------------------------------------------------
def test_normal_sample_can_be_loaded():
    data = vibration.load_signal_csv(sample_path("normal_1797"))

    assert data["n_samples"] == 12000
    assert data["amplitude"].shape == (12000,)
    assert data["timestamps"] is not None and data["timestamps"].shape == (12000,)
    assert np.isfinite(data["amplitude"]).all()
    # 样例 CSV 的时间戳保留 6 位小数（0.000083），推断采样率约 12048Hz，仍在 1% 容差内
    step = float(np.median(np.diff(data["timestamps"])))
    assert 1.0 / step == pytest.approx(FS, rel=0.01)


def test_single_column_format_is_supported(tmp_path):
    """单列 amplitude 格式也应能读取（表头只有 amplitude 时按行号生成时间戳）。"""
    values = np.linspace(-1.0, 1.0, 2048)
    path = write_signal_csv(tmp_path / "single_column.csv", values, timestamps=False)

    data = vibration.load_signal_csv(path)

    assert data["n_samples"] == 2048
    assert data["amplitude"][:3] == pytest.approx(values[:3], rel=1e-6)


def test_valid_pair_passes_all_checks():
    result = vibration.validate_vibration_data(sample_path("normal_1797"), sample_path("inner_race_1797"), FS, "drive_end")

    assert result["valid"] is True
    assert result["comparable"] is True
    assert result["errors"] == []
    assert set(result["checks"]) == {"format", "columns", "values", "length", "sampling_rate", "comparability"}
    assert all(item["passed"] for item in result["checks"].values())
    assert result["normal"]["n_samples"] == result["abnormal"]["n_samples"] == 12000


# ---------------------------------------------------------------------------
# 2. 缺失值处理（corrupted_1797.csv 含空值 / null / N/A / NaN / --）
# ---------------------------------------------------------------------------
def test_corrupted_sample_reports_missing_value():
    corrupted = str(SAMPLES / "corrupted_1797.csv")
    result = vibration.validate_vibration_data(sample_path("normal_1797"), corrupted, FS, "drive_end")

    assert result["valid"] is False
    assert result["comparable"] is False
    assert result["checks"]["values"]["passed"] is False

    codes = [item["code"] for item in result["errors"]]
    assert codes[0] == vibration.ERROR_MISSING_VALUE
    assert vibration.ERROR_MISSING_VALUE in codes
    assert vibration.ERROR_INVALID_VALUE in codes
    assert vibration.ERROR_INSUFFICIENT_DATA in codes
    assert vibration.ERROR_INCOMPARABLE_LENGTH in codes

    first = result["errors"][0]
    assert first["field"] == "amplitude"
    assert "第 19 行 amplitude 为空值" in first["message"]
    # 300 行里 7 行有问题，有效点只有 293 个
    assert result["abnormal"]["n_samples"] == 293


def test_corrupted_sample_raises_on_load():
    with pytest.raises(vibration.VibrationError) as excinfo:
        vibration.load_signal_csv(str(SAMPLES / "corrupted_1797.csv"))

    assert excinfo.value.code == vibration.ERROR_MISSING_VALUE
    assert excinfo.value.field == "amplitude"


# ---------------------------------------------------------------------------
# 3. 数据长度不一致
# ---------------------------------------------------------------------------
def test_length_mismatch_over_min_samples_is_comparable_with_warning(tmp_path):
    long_path = write_signal_csv(tmp_path / "long.csv", np.linspace(-1.0, 1.0, 12000))
    medium_path = write_signal_csv(tmp_path / "medium.csv", np.linspace(-1.0, 1.0, 8000))

    result = vibration.validate_vibration_data(long_path, medium_path, FS)

    assert result["normal"]["n_samples"] == 12000
    assert result["abnormal"]["n_samples"] == 8000
    assert result["comparable"] is True
    assert result["valid"] is True
    assert vibration.ERROR_INCOMPARABLE_LENGTH in [item["code"] for item in result["warnings"]]
    assert result["errors"] == []


def test_length_mismatch_below_min_samples_is_not_comparable(tmp_path):
    long_path = write_signal_csv(tmp_path / "long.csv", np.linspace(-1.0, 1.0, 12000))
    short_path = write_signal_csv(tmp_path / "short.csv", np.linspace(-1.0, 1.0, 300))

    result = vibration.validate_vibration_data(long_path, short_path, FS)

    assert result["comparable"] is False
    assert result["valid"] is False
    assert result["checks"]["comparability"]["passed"] is False
    assert result["checks"]["length"]["passed"] is False
    codes = [item["code"] for item in result["errors"]]
    assert vibration.ERROR_INCOMPARABLE_LENGTH in codes
    assert vibration.ERROR_INSUFFICIENT_DATA in codes


# ---------------------------------------------------------------------------
# 4. 采样率不一致 / 不可比较
# ---------------------------------------------------------------------------
def test_sampling_rate_mismatch_is_rejected(tmp_path):
    # 时间戳按 6000Hz 生成，但入参声称 12000Hz
    fs6000_path = write_signal_csv(
        tmp_path / "fs6000.csv", np.sin(np.linspace(0, 200 * np.pi, 2400)), sampling_rate=6000.0
    )

    result = vibration.validate_vibration_data(fs6000_path, sample_path("normal_1797"), FS)

    mismatches = [item for item in result["errors"] if item["code"] == vibration.ERROR_SAMPLING_RATE_MISMATCH]
    assert len(mismatches) == 1
    assert mismatches[0]["field"] == "sampling_rate"
    assert "6000" in mismatches[0]["message"] and "12000" in mismatches[0]["message"]
    assert result["checks"]["sampling_rate"]["passed"] is False
    assert result["valid"] is False


def test_two_different_sampling_rates_are_both_flagged(tmp_path):
    fs6000_path = write_signal_csv(
        tmp_path / "fs6000.csv", np.sin(np.linspace(0, 200 * np.pi, 2400)), sampling_rate=6000.0
    )

    result = vibration.validate_vibration_data(fs6000_path, sample_path("normal_1797"), 9000.0)

    mismatches = [item for item in result["errors"] if item["code"] == vibration.ERROR_SAMPLING_RATE_MISMATCH]
    assert len(mismatches) == 2
    assert [item["message"].split("：")[0] for item in mismatches] == ["normal", "abnormal"]
    for item in mismatches:
        assert "与入参 9000Hz 不一致" in item["message"]
    # 两个文件各自推断出的采样率（样例时间戳 6 位小数，实际推断值约 12048Hz）
    inferred = [float(re.search(r"约 ([\d.]+)Hz", item["message"]).group(1)) for item in mismatches]
    assert inferred[0] == pytest.approx(6000.0, rel=0.01)
    assert inferred[1] == pytest.approx(12000.0, rel=0.01)
    assert result["valid"] is False


def test_invalid_sampling_rate_is_rejected():
    result = vibration.validate_vibration_data(sample_path("normal_1797"), sample_path("normal_1797"), 0)

    assert result["valid"] is False
    assert result["errors"][0]["code"] == vibration.ERROR_INVALID_VALUE
    assert result["errors"][0]["field"] == "sampling_rate"


# ---------------------------------------------------------------------------
# 5. 特征计算
# ---------------------------------------------------------------------------
def test_feature_extraction_matches_independent_recomputation():
    features = vibration.extract_vibration_features(sample_path("normal_1797"), FS)

    for key in (
        "sampling_rate",
        "n_samples",
        "duration_seconds",
        "rms",
        "peak",
        "kurtosis",
        "crest_factor",
        "dominant_frequency",
        "dominant_amplitude",
        "band_energy",
        "spectrum_peaks",
    ):
        assert key in features

    assert features["sampling_rate"] == pytest.approx(FS)
    assert features["n_samples"] == 12000
    assert features["duration_seconds"] == pytest.approx(1.0)

    raw = read_amplitude(sample_path("normal_1797"))
    centered = raw - float(np.mean(raw))

    assert features["rms"] == pytest.approx(float(np.sqrt(np.mean(centered**2))), rel=1e-4)
    assert features["peak"] == pytest.approx(float(np.max(np.abs(centered))), rel=1e-4)
    assert features["kurtosis"] == pytest.approx(
        float(sp_stats.kurtosis(centered, fisher=False, bias=False)), rel=1e-4
    )
    assert features["crest_factor"] == pytest.approx(features["peak"] / features["rms"], rel=1e-3)

    magnitude, freqs = reference_spectrum(centered, FS)
    assert features["dominant_frequency"] == pytest.approx(float(freqs[int(np.argmax(magnitude))]), abs=1e-3)
    assert features["dominant_amplitude"] == pytest.approx(float(np.max(magnitude)), rel=1e-4)

    # 频带能量：默认按奈奎斯特均分 5 段，完全覆盖 0~Nyquist，总和等于全谱能量
    band_energy = features["band_energy"]
    assert list(band_energy) == EXPECTED_BANDS
    assert all(value >= 0 for value in band_energy.values())
    assert sum(band_energy.values()) == pytest.approx(float(np.sum(magnitude**2)), rel=1e-4)


def test_spectrum_peaks_are_real_local_maxima_on_frequency_grid():
    features = vibration.extract_vibration_features(sample_path("normal_1797"), FS)
    peaks = features["spectrum_peaks"]

    # 全谱幅值前 5 + 特征带内幅值前 _CHARACTERISTIC_PEAK_LIMIT（两者取并集）
    assert 0 < len(peaks) <= 5 + vibration._CHARACTERISTIC_PEAK_LIMIT
    frequencies = [peak["frequency"] for peak in peaks]
    assert frequencies == sorted(frequencies)
    assert len(set(frequencies)) == len(frequencies)

    raw = read_amplitude(sample_path("normal_1797"))
    magnitude, freqs = reference_spectrum(raw - float(np.mean(raw)), FS)
    resolution = FS / raw.size

    for peak in peaks:
        assert set(peak) == {"frequency", "amplitude"}
        assert peak["amplitude"] > 0
        # 峰值必须落在频率栅格上（间隔 = fs/N）
        assert peak["frequency"] / resolution == pytest.approx(round(peak["frequency"] / resolution), abs=1e-6)
        index = int(round(peak["frequency"] * raw.size / FS))
        assert abs(float(freqs[index]) - peak["frequency"]) < 1e-3
        assert peak["amplitude"] == pytest.approx(float(magnitude[index]), rel=1e-3, abs=1e-6)
        # 且确实是局部极大值
        assert magnitude[index] >= magnitude[index - 1]
        assert magnitude[index] >= magnitude[index + 1]


def test_envelope_analysis_is_optional_and_complete():
    plain = vibration.extract_vibration_features(sample_path("normal_1797"), FS)
    assert plain["envelope"] is None

    with_envelope = vibration.extract_vibration_features(sample_path("normal_1797"), FS, rotation_speed=1797)

    envelope = with_envelope["envelope"]
    assert envelope is not None
    assert set(envelope["characteristic_frequencies"]) == {"shaft", "ftf", "bsf", "bsf_2x", "bpfo", "bpfi"}
    assert set(envelope["characteristic_energy"]) == {"ftf", "bsf", "bpfo", "bpfi"}
    assert set(envelope["band_hz"]) and envelope["band_rule"]
    fault_frequencies = vibration.bearing_fault_frequencies(1797)
    assert envelope["characteristic_frequencies"]["bpfi"] == pytest.approx(fault_frequencies["bpfi"], abs=1e-3)
    assert envelope["characteristic_frequencies"]["shaft"] == pytest.approx(1797 / 60.0, abs=1e-3)
    # 滚动体缺陷的判据落在 2×BSF，对外单列一档，避免与 1×BSF 口径混淆
    assert envelope["characteristic_frequencies"]["bsf_2x"] == pytest.approx(
        2 * fault_frequencies["bsf"], abs=1e-3
    )


def test_bearing_fault_frequencies_follow_cwru_6205_formulas():
    faults = vibration.bearing_fault_frequencies(1797)

    fr = 1797 / 60.0
    geometry = vibration.BEARING_GEOMETRY
    ratio = geometry["ball_diameter"] / geometry["pitch_diameter"] * np.cos(np.radians(geometry["contact_angle_deg"]))
    half = geometry["n_balls"] / 2.0

    assert faults["shaft_frequency"] == pytest.approx(fr, abs=1e-3)
    assert faults["bpfo"] == pytest.approx(half * fr * (1 - ratio), abs=1e-3)
    assert faults["bpfi"] == pytest.approx(half * fr * (1 + ratio), abs=1e-3)
    assert faults["bsf"] == pytest.approx(geometry["pitch_diameter"] / (2 * geometry["ball_diameter"]) * fr * (1 - ratio**2), abs=1e-3)
    assert faults["ftf"] == pytest.approx(fr / 2.0 * (1 - ratio), abs=1e-3)


# ---------------------------------------------------------------------------
# 6. 频谱峰值检测（合成正弦信号）
# ---------------------------------------------------------------------------
def test_spectrum_peak_detection_on_synthetic_sine(tmp_path):
    # 162.2Hz 为主分量（对应 1797rpm 下 1×BPFI 的 162.186Hz），另加 2× 谐波与一个低频分量
    n = 12000
    t = np.arange(n) / FS
    synthetic = (
        1.0 * np.sin(2 * np.pi * 162.2 * t)
        + 0.6 * np.sin(2 * np.pi * 324.4 * t)
        + 0.3 * np.sin(2 * np.pi * 30.0 * t)
    )
    path = write_signal_csv(tmp_path / "sine.csv", synthetic)

    features = vibration.extract_vibration_features(path, FS)

    assert features["dominant_frequency"] == pytest.approx(162.2, abs=1.5)
    assert features["dominant_amplitude"] > 0.4

    detected = [peak["frequency"] for peak in features["spectrum_peaks"]]
    assert any(abs(freq - 162.2) <= 1.5 for freq in detected)
    assert any(abs(freq - 324.4) <= 2.0 for freq in detected)

    # 1 秒样本的频率分辨率就是 fs/N = 1Hz：162.2Hz 只能落在 162.0Hz 栅格上
    resolution = FS / n
    assert resolution == pytest.approx(1.0)
    assert features["dominant_frequency"] == pytest.approx(162.0, abs=1e-6)
    assert vibration.bearing_fault_frequencies(1797)["bpfi"] == pytest.approx(162.186, abs=0.01)