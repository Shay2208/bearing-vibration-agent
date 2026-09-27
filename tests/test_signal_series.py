"""原始信号曲线序列与波形音频测试。

覆盖：抽点形状与峰值保真、包络谱裁剪范围、WAV 头与归一化、序列不影响标量特征、
端到端响应不含 PCM 字节、音频懒加载端点命中与 404 分支。
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave

import httpx
import numpy as np
import pytest

from app.config import settings
from app.main import app
from app.mcp import vibration
from tests.helpers import INNER_RACE, NORMAL, CORRUPTED, diagnose_local, sample_path
from tests.test_unit_vibration import read_amplitude, reference_spectrum

BASE_URL = "http://testserver"


def call(method: str, url: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def extract(stem: str, **kwargs) -> dict:
    return vibration.extract_vibration_features(
        sample_path(stem), 12000, rotation_speed=1797, include_series=True, **kwargs
    )


def test_series_is_opt_in():
    assert vibration.extract_vibration_features(sample_path(INNER_RACE), 12000)["series"] is None


def test_waveform_buckets_keep_min_max_envelope():
    series = extract(INNER_RACE)["series"]
    waveform = series["waveform"]
    # 序列取自去均值后的分析信号（与频谱、音频同一路），对比值同样先去均值
    centered = read_amplitude(sample_path(INNER_RACE))
    centered = centered - centered.mean()

    assert waveform["buckets"] == 600
    assert len(waveform["min"]) == len(waveform["max"]) == 600
    # min/max 分桶必须真的包住原始波形：抽点后的极值不超过整段极值，且波动幅度量级一致
    assert max(waveform["max"]) <= float(np.max(centered)) + 1e-6
    assert min(waveform["min"]) >= float(np.min(centered)) - 1e-6
    assert max(waveform["max"]) - min(waveform["min"]) >= 0.5 * (float(np.max(centered)) - float(np.min(centered)))


def test_spectrum_sampling_preserves_peak_bins():
    series = extract(INNER_RACE)["series"]
    spectrum = series["spectrum"]
    magnitude, freqs = reference_spectrum(read_amplitude(sample_path(INNER_RACE)), 12000.0)

    assert spectrum["points"] == 600
    assert len(spectrum["frequency"]) == len(spectrum["amplitude"]) == 600
    assert spectrum["frequency"] == sorted(spectrum["frequency"])
    # 峰值保序：全谱最大幅值与其频率都必须出现在抽点结果里
    assert max(spectrum["amplitude"]) == pytest.approx(float(np.max(magnitude)), abs=1e-6)
    assert float(freqs[int(np.argmax(magnitude))]) in spectrum["frequency"]


def test_envelope_series_is_cropped_to_1khz():
    series = extract(INNER_RACE)["series"]
    envelope = series["envelope"]

    assert envelope["points"] == 400
    assert max(envelope["frequency"]) <= 1000.0
    assert max(envelope["amplitude"]) > 0


def test_wav_is_mono_int16_normalized():
    audio = extract(INNER_RACE)["series"]["audio"]
    raw = base64.b64decode(audio["wav_base64"])

    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    with wave.open(io.BytesIO(raw)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 12000
        assert handle.getnframes() == 12000
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    # 峰值归一：最大幅度接近满量程但不削波
    assert 0.9 * 32767.0 <= float(np.max(np.abs(pcm))) <= 32767.0


def test_series_does_not_change_scalar_features():
    with_series = extract(INNER_RACE)
    without = vibration.extract_vibration_features(sample_path(INNER_RACE), 12000, rotation_speed=1797)

    assert with_series["series"] is not None
    for key in ("rms", "peak", "kurtosis", "crest_factor", "dominant_frequency", "band_energy", "spectrum_peaks"):
        assert with_series[key] == without[key]


def test_diagnose_response_carries_series_but_no_pcm(local_backend):
    result = diagnose_local(NORMAL, INNER_RACE)
    series = result["features"]["normal"]["series"]

    assert result["trace"]["tool_calls"] == 4
    assert series["spectrum"]["points"] == 600
    for slot in ("normal", "abnormal"):
        audio = result["features"][slot]["series"]["audio"]
        assert audio["available"] is True
        assert "wav_base64" not in audio
    assert "wav_base64" not in json.dumps(result)


def test_audio_endpoint_serves_wav_and_handles_missing(local_backend):
    response = call(
        "POST",
        "/api/diagnose",
        data={
            "sampling_rate": 12000,
            "rotation_speed": 1797,
            "normal_sample": NORMAL,
            "abnormal_sample": INNER_RACE,
        },
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    audio = call("GET", f"/api/sessions/{session_id}/audio/normal.wav")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"

    assert call("GET", f"/api/sessions/{session_id}/audio/side.wav").status_code == 404
    assert call("GET", "/api/sessions/no-such-session/audio/normal.wav").status_code == 404
    assert call("GET", "/api/sessions/../audio/normal.wav").status_code == 404


def test_corrupted_sample_yields_no_series_or_audio(local_backend):
    result = diagnose_local(NORMAL, CORRUPTED)

    assert result["status"] == "error"
    assert result["trace"]["tool_calls"] == 1
    assert result["features"] is None
    assert not list(settings.audio_dir.glob(f"{result['session_id']}_*.wav"))
