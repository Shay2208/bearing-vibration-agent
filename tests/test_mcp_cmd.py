"""MCP_SERVER_CMD 解析与 stdio 启动：JSON 数组、含空格路径、非法配置的结构化错误。

对应适配层改造（`app/mcp/adapter.py` 的命令解析）：
  1. JSON 数组形式能解析成正确的 argv；
  2. 带引号、含空格的路径能解析成正确的 argv（这里用项目根下真实含空格的 venv 解释器路径）；
  3. 真实走 mcp:stdio 后端跑通一次工具调用（标 stdio，每次调用新建子进程，约 2~3 秒）；
  4. 非法 MCP_SERVER_CMD 返回结构化错误信封，不抛裸异常。

不依赖任何 API Key；不修改 tests/helpers.py 与 tests/conftest.py。
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, settings
from app.mcp import adapter
from tests.helpers import backend_mode, sample_path

MODULE_CMD = "app.mcp.server"
SERVER_SCRIPT = PROJECT_ROOT / "app" / "mcp" / "server.py"
INVALID_CODE = "invalid_server_command"


@contextmanager
def server_command(value: str):
    """临时设置 settings.mcp_server_cmd。

    settings 是 frozen dataclass，普通赋值会抛 FrozenInstanceError，
    这里与 tests/helpers.backend_mode 一样用 object.__setattr__ 绕过，退出时恢复原值。
    """
    previous = settings.mcp_server_cmd
    object.__setattr__(settings, "mcp_server_cmd", value)
    try:
        yield value
    finally:
        object.__setattr__(settings, "mcp_server_cmd", previous)


# ---------------------------------------------------------------------------
# 1. JSON 数组形式
# ---------------------------------------------------------------------------
def test_json_array_command_is_parsed_into_argv():
    expected = [r"C:\Program Files\Python\python.exe", "-m", MODULE_CMD]
    raw = json.dumps(expected)  # 例如 ["C:\\Program Files\\Python\\python.exe", "-m", "app.mcp.server"]

    assert raw.startswith("[") and "\\\\" in raw  # JSON 里反斜杠是转义的
    assert adapter.parse_server_command(raw) == expected

    # 路径含空格，但必须仍然只有 3 个 argv 元素
    argv = adapter.parse_server_command(raw)
    assert len(argv) == 3
    assert argv[0] == r"C:\Program Files\Python\python.exe"
    assert argv[1:] == ["-m", MODULE_CMD]


def test_json_array_command_is_used_as_is_by_server_parameters():
    # 用真实解释器路径（项目根含空格），验证 argv → StdioServerParameters 的映射与 cwd
    raw = json.dumps([sys.executable, "-m", MODULE_CMD])
    assert " " in sys.executable

    with server_command(raw):
        params = adapter._server_parameters()

    assert params.command == sys.executable
    assert list(params.args) == ["-m", MODULE_CMD]
    assert params.cwd == str(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 2. 带引号、含空格的路径（用真实含空格的 venv 解释器路径）
# ---------------------------------------------------------------------------
def test_quoted_command_with_space_path_is_parsed_into_argv():
    executable = sys.executable
    assert " " in executable, f"本用例依赖含空格的项目路径，当前解释器路径不含空格：{executable}"
    assert "\\" in executable  # Windows 路径反斜杠，用于验证不会被当转义符吃掉

    raw = f'"{executable}" -m {MODULE_CMD}'
    argv = adapter.parse_server_command(raw)

    assert argv == [executable, "-m", MODULE_CMD]
    assert len(argv) == 3
    assert Path(argv[0]).is_file()
    # 含空格的路径没有被切成两段
    assert " " in argv[0]

    with server_command(raw):
        params = adapter._server_parameters()
    assert params.command == executable
    assert list(params.args) == ["-m", MODULE_CMD]
    assert params.cwd == str(PROJECT_ROOT)


def test_single_quoted_command_with_space_path_is_parsed_into_argv():
    executable = sys.executable
    raw = f"'{executable}' -m {MODULE_CMD}"

    assert adapter.parse_server_command(raw) == [executable, "-m", MODULE_CMD]


def test_empty_command_falls_back_to_current_interpreter_and_server_script():
    """空值回退到原有默认行为，且不改变 local 后端与 describe_backend() 的语义。"""
    with backend_mode("local"):
        assert adapter.describe_backend() == "local"
        with server_command(""):
            params = adapter._server_parameters()

    assert params.command == sys.executable
    assert list(params.args) == [str(SERVER_SCRIPT)]
    assert params.cwd == str(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 3. 真实跑通一次：引号包含空格的 venv python + -m app.mcp.server
# ---------------------------------------------------------------------------
@pytest.mark.stdio
def test_quoted_command_with_space_path_runs_stdio_backend():
    raw = f'"{sys.executable}" -m {MODULE_CMD}'
    csv_path = sample_path("normal_1797")

    with server_command(raw):
        with backend_mode("stdio"):
            assert adapter.describe_backend() == "mcp:stdio"
            envelope = asyncio.run(adapter.extract_vibration_features(csv_path, 12000, rotation_speed=1797))

    assert envelope["ok"] is True, envelope
    assert envelope["backend"] == "mcp:stdio"
    assert envelope["tool"] == "extract_vibration_features"

    with backend_mode("local"):
        local_envelope = asyncio.run(adapter.extract_vibration_features(csv_path, 12000, rotation_speed=1797))

    assert local_envelope["ok"] is True
    assert envelope["data"]["rms"] == pytest.approx(local_envelope["data"]["rms"], rel=1e-9)
    assert envelope["data"]["dominant_frequency"] == local_envelope["data"]["dominant_frequency"]


# ---------------------------------------------------------------------------
# 4. 非法 MCP_SERVER_CMD → 结构化错误信封（不抛裸异常）
# ---------------------------------------------------------------------------
MISSING_EXECUTABLE = str(PROJECT_ROOT / "no_such_dir" / "definitely_not_an_executable.exe")

INVALID_COMMANDS = [
    pytest.param(f'"{MISSING_EXECUTABLE}" -m {MODULE_CMD}', id="missing-executable"),
    pytest.param("[" + f'"{MISSING_EXECUTABLE}", "-m"' + ",]", id="broken-json-array"),
    pytest.param(f'"{MISSING_EXECUTABLE} -m {MODULE_CMD}', id="unbalanced-quote"),
    pytest.param("[]", id="empty-json-array"),
]


@pytest.mark.parametrize("raw", INVALID_COMMANDS)
def test_invalid_server_command_returns_structured_error(raw):
    with server_command(raw):
        with backend_mode("stdio"):
            envelope = asyncio.run(adapter.extract_vibration_features(sample_path("normal_1797"), 12000))

    assert envelope["ok"] is False
    assert envelope["tool"] == "extract_vibration_features"
    assert envelope["backend"] == "mcp:stdio"
    assert set(envelope["error"]) == {"code", "message", "field"}
    assert envelope["error"]["code"] == INVALID_CODE
    assert envelope["error"]["message"].strip()
    assert "MCP 服务启动命令无效" in envelope["error"]["message"]


def test_parse_server_command_raises_on_invalid_input():
    """解析层对非法输入抛 ServerCommandError（ValueError 子类），不返回半成品命令。"""
    for raw in ("[]", f'"{MISSING_EXECUTABLE} -m {MODULE_CMD}', "  "):
        with pytest.raises(adapter.ServerCommandError):
            adapter.parse_server_command(raw)