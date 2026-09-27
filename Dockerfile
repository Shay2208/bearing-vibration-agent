# 工业轴承振动故障诊断助手 Agent —— 运行镜像
#
# 构建：docker build -t bearing-diagnosis .
# 运行（8000 端口暴露，密钥只通过环境变量注入，绝不写进镜像）：
#   docker run --rm -p 8000:8000 \
#     -e MODEL_API_BASE=... -e MODEL_API_KEY=... \
#     -v "${PWD}/data/chroma:/app/data/chroma" \
#     bearing-diagnosis
# 不传任何环境变量也能启动、能离线演示：知识检索自动降级 BM25，报告走模板化生成。

FROM python:3.12-slim

# numpy / scipy / openai / mcp / uvicorn[standard] 在 cp312 上都有 manylinux wheel，
# slim 镜像无需额外装 gcc / g++ / libgfortran 等编译工具链。
# 若将来 requirements.txt 引入只能源码编译的依赖，再在此处补 apt-get 安装构建依赖。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MCP_MODE=local \
    LLM_TIMEOUT=20 \
    LLM_MAX_RETRIES=1 \
    MAX_UPLOAD_BYTES=20971520

WORKDIR /app

# requirements.txt 由项目维护，本镜像只引用、不修改
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 只复制运行所需内容：应用代码（含 app/rag/knowledge 知识库与 app/web 演示页面）与内置样例
# （data/samples 是离线演示的必需资源）。.venv / .env / data/chroma / __pycache__ 见 .dockerignore。
COPY app/ ./app/
COPY data/samples/ ./data/samples/

# 向量索引持久化目录：构建镜像时不带索引（因此镜像里没有它），
# 运行时用 -v 挂载宿主机目录即可复用；未挂载时 Docker 会创建一个空卷，
# 代码侧对索引缺失有降级处理（向量不可用 → BM25），所以应用仍能正常启动。
VOLUME ["/app/data/chroma"]

EXPOSE 8000

# exec form：uvicorn 作为 PID 1 直接接收信号，CTRL+C / docker stop 能优雅退出。
# MCP_MODE 默认 local，容器内也可显式设为 stdio 验证真实 MCP 协议链路。
# stdio 后端使用同一解释器启动 app/mcp/server.py（sys.executable 存在、cwd=/app）。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
