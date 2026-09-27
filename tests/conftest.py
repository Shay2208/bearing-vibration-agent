"""pytest 公共 fixture。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# `settings` 是 app.config 导入时构建的单例，且该模块用 setdefault 读取项目根 .env，
# 因此必须在任何 app.* 导入之前把模型接口置空（置空串而非删除，setdefault 才不会用
# .env 里的真实值把它补回来）。这样测试恒定运行在「离线 + 模板报告 + 纯 BM25」环境，
# 不依赖开发者本机的 .env 与真实密钥。需要模型/向量的用例都通过注入假客户端实现。
os.environ["MODEL_API_BASE"] = ""
os.environ["MODEL_API_KEY"] = ""

# 会话落盘改为每次测试运行的临时库：既不污染仓库 data/sessions.db，
# 也让「重启不丢」「会话列表」这类用例不受上一次运行的历史数据影响。
os.environ["SESSIONS_DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="bearing-sessions-")) / "sessions.db")

# 波形音频同理落到临时目录，避免诊断用例把 WAV 写进仓库 data/audio。
os.environ["AUDIO_DIR"] = str(Path(tempfile.mkdtemp(prefix="bearing-audio-")) / "audio")

import pytest  # noqa: E402

from tests.helpers import backend_mode  # noqa: E402


@pytest.fixture
def local_backend():
    """把 MCP 后端切到 local（进程内调用），避免常规测试反复启动 stdio 子进程。"""
    with backend_mode("local"):
        yield "local"


@pytest.fixture
def stdio_backend():
    """把 MCP 后端切到 mcp:stdio（真实 MCP 服务子进程）。"""
    with backend_mode("stdio"):
        yield "mcp:stdio"