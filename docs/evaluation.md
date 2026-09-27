# 工业设备故障诊断 Agent 工作台 · 测试与评测记录

本文记录仓库现有实现（`app/**`、`data/samples/**`）在本地真实跑出的测试与评测结果：**138 个测试用例全部通过**（125 条原有测试零回退 + 判据条款化与条款级核对新增 13 条），7 个端到端场景逐条给出 `status`、候选、来源引用、工具调用次数、耗时与检索方式，并如实记录 1 个确凿的误判案例（本轮已修正为正确候选 `ball_fault`、置信度 low）与各项未验证内容。**所有数字均来自真实运行，不含推算值，也不对 5 个内置样例之外的样本做任何准确率外推。**

## 一、测试文件与覆盖范围

测试代码全部放在 `tests/` 下，只调用现有实现，不修改 `app/**` 与 `data/**`。

| 文件 | 覆盖内容 | 用例数 |
|---|---|---|
| `tests/helpers.py` | 公共工具：样例路径、临时 CSV 构造、`backend_mode()` 切换 MCP 后端、`diagnose_local()`、`TimeoutLLMClient`、`fake_embedder` | — |
| `tests/conftest.py` | `local_backend` / `stdio_backend` 两个 fixture；启动即把 `SESSIONS_DB_PATH` 与 `AUDIO_DIR` 指向临时目录，避免测试写仓库的 `data/sessions.db` 与 `data/audio` | — |
| `tests/test_unit_vibration.py` | 数据读取、缺失值、长度不一致、采样率不一致、特征计算与独立复算、频谱峰值检测、特征频率公式 | 15 |
| `tests/test_unit_adapter_rag_report.py` | MCP 错误信封与双后端一致性、RAG 空结果、报告字段完整性 | 9 |
| `tests/test_e2e_diagnose.py` | `orchestrator.diagnose()` 的 7 个端到端场景 | 7 |
| `tests/test_api.py` | `/health`、`/api/samples`、`/api/diagnose` 的正常与异常路径，以及向量库/编排改造后新增的接口契约、模型等待期间的健康检查 | 14 |
| `tests/test_mcp_cmd.py` | `MCP_SERVER_CMD` 的解析（JSON 数组、含空格引号路径、空值回退、非法配置）与一次真实 stdio 调用 | 11 |
| `tests/test_rag_vector.py` | 知识切分（块数、块大小、重叠、稳定块 ID）、Chroma 持久化写入与 `--status`、混合检索打分与降级 | 7 |
| `tests/test_graph_flow.py` | LangGraph 图结构与节点访问顺序、三条降级路径、候选外类型拦截、错误路径契约、并列族采信守卫（第 10 节） | 22 |
| `tests/test_sse_ask.py` | SSE 事件顺序与统一信封 9 字段、工具失败事件、证据不足分支、模型回退、断开安全、旧接口兼容、受控追问（越权 / 404 / 无依据拦截）、会话内自由问答（事实直答 / 模型回答校验 / 降级 / 错误码） | 22 |
| `tests/test_session_persistence.py` | 会话与事件落盘后重开仍在、会话列表与详情接口、详情 404、追问与自由问答消息落盘、对话历史轮数窗口、容量淘汰连带删消息 | 9 |
| `tests/test_signal_series.py` | 曲线序列 opt-in（默认不返回）、波形分桶保住 min/max、幅度谱峰值保序抽点、包络谱裁到 1kHz、WAV 单声道 16bit 归一化、序列不改变标量特征、响应不含 `wav_base64`、音频端点 200 与三类 404、损坏样本无序列无音频 | 9 |
| `tests/test_rag_criteria.py` | 条款词表守卫（全部条款的 `metric` / `op` 落在受控词表、条款 id 不重复）、对比事实缺失时排序不变、三态判定与比较符守卫、判据因子与状态边界、`alignment_score` 重排（检索分高但判据冲突的条目下沉）、条款级证据进 `evidence`、三场景条款状态（内圈 6/6、外圈滚动体条目冲突沉底、滚动体 5/5）、条款证据进 trace / 提示词 / 报告与 markdown、冲突不采信并追加人工复核 | 13 |
| `pytest.ini` | `testpaths=tests`、`pythonpath=.`、注册 `stdio` mark | — |

合计 15 + 9 + 7 + 14 + 11 + 7 + 22 + 22 + 9 + 9 + 13 = **138**。测试规模演进过程：**38 → 49 → 77 → 79 → 80 → 92 → 102 → 111 → 116 → 125 → 138**。

后端策略：常规用例统一通过 `local_backend` fixture 切到进程内后端（单次诊断约 0.08~0.12 秒）；标了 `@pytest.mark.stdio` 的用例真实走 `mcp:stdio`（每次工具调用新建一次 stdio 连接与子进程），它们**在默认 `pytest` 运行中通过**，可用 `-m stdio` 单独筛选。

## 二、运行方式与原始输出

解释器用仓库自带虚拟环境，工作目录为项目根 `bearing-vibration-agent`。

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

原始输出（判据条款化与条款级核对接入后最近一次全量运行，2026-09-26）：

```
138 passed in 20.86s
```

此前记录：125 passed in 24.10s（原始曲线与波形音频接入后）、116 passed in 18.45s（滚动体并列族采信修正后）、111 passed in 19.07s（滚动体误判修正与谱峰扩围后）、111 passed in 18.86s（会话持久化改造后）、102 passed in 20.10s（三栏工作台改造后）、92 passed in 24.54s（SSE 与追问接入后）、80 passed in 21.23s（改造前基线），原有用例零回退。耗时排前几位的用例仍是 `mcp:stdio` 后端（每次调用新建一次子进程，单条 2~3 秒），合计占掉整轮时间的一大半；`test_mcp_cmd.py::test_quoted_command_with_space_path_runs_stdio_backend` 真实走通一次 stdio（约 2.1 秒）。测试环境为 win32 / Python 3.12.10 / pytest 9.1.1。

## 三、端到端评测（真实数据）

7 个场景都调用真实的 `orchestrator.diagnose()`。场景 1~5、7 用内置样例（fs=12000、1797rpm、`drive_end`），场景 6 通过 `monkeypatch` 把 `RAG_SCORE_THRESHOLD` 抬到 `0.99` 模拟知识库未命中。

| 场景 | status | 候选 top1 | score | 是否引用来源 | 与 CWRU 公开标签 | tool_calls | elapsed_ms | llm_mode | retrieval |
|---|---|---|---|---|---|---|---|---|---|
| 1 正常 vs 正常 | `insufficient_evidence` | — | — | 无（`candidates`/`sources`/`conclusion_sources` 均为空） | —（同为正常，期望证据不足） | 4 | 103 | `template` | `bm25` |
| 2 正常 vs 内圈 | `ok` | `inner_race_fault` | 0.4453 | sources=2（内圈/外圈两条候选，同名来源）、报告引用 1 条（CWRU Bearing Data Center） | 一致 | 4 | 109 | `template` | `bm25` |
| 3 正常 vs 外圈 | `ok` | `outer_race_fault` | 0.4412 | sources=1，报告引用 1 条（CWRU Bearing Data Center） | 一致 | 4 | 97 | `template` | `bm25` |
| 4 正常 vs 滚动体 | `ok` | `ball_fault` | 0.6283 | sources=1（仅 `ball_fault` 一条候选，报告引用 1 条） | 一致 | 4 | 105 | `template` | `bm25` |
| 5 正常 vs 损坏样本 | `error` | — | — | 无 | —（手工构造样本，无真实工况） | 1 | 23 | `template` | — |
| 6 知识库未命中（阈值 0.99） | `insufficient_evidence` | — | — | 无 | —（人为抬高阈值） | 4 | 98 | `template` | `bm25` |
| 7 正常 vs 内圈（模型超时） | `ok` | `inner_race_fault` | 0.4453 | sources=1，报告引用 1 条（CWRU Bearing Data Center） | 一致 | 4 | 106 | `template_fallback` | `bm25` |

补充说明（同为实测）：

- **场景 5 的错误载荷**：`error.code = missing_value`，消息为「数据校验未通过（valid=False，comparable=False），流程在特征提取前停止：abnormal：第 19 行 amplitude 为空值；abnormal：第 60 行 amplitude 不是有效数字：'null'；abnormal：第 105 行 amplitude 不是有效数字：'N/A'」。`report` 与 `report_markdown` 均为 `null`，`candidates` 为空，`tool_calls` 只有 1（校验阶段就停机）。
- **场景 1 的来源残留（已修复）**：此前正常对正常时 `sources` 字段会残留 1 条（`inner_race_fault` 的来源），原因是在检索完成后就赋值 `out["sources"]`，而「无显著变化」拦截在其后才清空 `candidates`。现在 `evidence_check_node` 在拦截时会把候选与来源一并清空，实测正常对正常返回 `status=insufficient_evidence`、`candidates==[]`、`sources==[]`、`report["conclusion_sources"]==[]`、报告正文含「无法确认」。回归测试为 `tests/test_api.py::test_normal_vs_normal_returns_no_sources`，已通过。
- **场景 6 的验证方式**：把阈值抬到 `0.99` 后 `candidates` 为空、`sources` 为空、`status=insufficient_evidence`、报告结论为「无法确认」，模型调用被跳过（`llm_calls=0`）。另一条独立验证见第四节「RAG 空结果」用例：用与知识库无关的查询词也能得到同样的空候选结果。
- **场景 7 的降级**：注入的假客户端抛 `LLMError("timeout", ...)`，`trace.llm_mode` 为 `template_fallback`、`trace.llm_calls=1`、`trace.llm_error={"code":"timeout","message":"模拟模型调用超时"}`，`status` 仍为 `ok`、报告结构与长度完整（`report_markdown` 非空），说明模型失败不会中断主流程。
- **场景 4 的并列族采信（本轮修正）**：`comparison.dominant_characteristic` 给出 `label=BPFI`、`ambiguous=true`、`margin_ratio=1.0706` 与 `tied_candidates`（BPFI 4.20470e-4 @162.186Hz / BSF 3.92749e-4 @70.5838Hz），过阈值候选只有 1 条 `ball_fault`（0.6283，`frequency_family=BSF`）。编排层 `_tied_family_resolution()` 判定首位候选声明的族 BSF 落在并列族（BPFI、BSF）内，据此采信该候选并写入 `trace.tied_resolution = {"fault_type": "ball_fault", "frequency_family": "BSF", "tied_labels": ["BPFI", "BSF"], "margin_ratio": 1.0706}`；`trace.warnings` 出现「包络谱特征频率族并列（BPFI、BSF，最高/次高能量比 1.071 低于量级余量阈值），但检索首位候选 ball_fault 声明的特征频率族 BSF 落在并列族内，据此采信该候选并下调置信度；并列本身仍提示需人工复核，不代表主导族已确定。」。最终 `status=ok`、结论「最可能故障类型：滚动轴承滚动体（钢球）故障（ball_fault）。」、置信等级 **low**（理由写明主导特征频率族不可判定、本次结论系并列证据下的采信），`llm_calls=0`（模板模式）。报告另新增一条现场复核建议：「现场层面：包络谱中 BPFI、BSF 特征频率族能量并列、无单一主导族，请分别核对各并列族（含 2×BSF 与转频边带）的谐波能量与周期稳定性，确认是否属能量分散型损伤表现后再决定是否拆检。」
- **场景 2 的复核告警（连带观察）**：谱峰按特征频率族聚合进查询后，内圈样本前两名候选（`inner_race_fault` 0.4453 / `outer_race_fault` 0.3986）分差为 **0.0467**，低于 `CANDIDATE_GAP_THRESHOLD=0.05`，因此 `trace.warnings` 会追加「前两名候选分数接近…单凭检索分数不足以区分，已追加频率复核」；本轮条款核对后第二名 `outer_race_fault` 为 4 命中 / 2 未命中（partial，factor 0.8，alignment_score 0.3189），候选 top1 与最终结论均未改变。

### 检索候选与来源（真实输出）

| 场景 | RAG 候选（fault_type, score） | 图谱证据要点 |
|---|---|---|
| 2 正常 vs 内圈 | `inner_race_fault` 0.4453（条款 6/6 命中，alignment 0.4453）、`outer_race_fault` 0.3986（4 命中 / 2 未命中，alignment 0.3189） | 包络谱 BPFI 能量 0.01513（基线未检出），RMS 上升 295%，峭度 2.87 → 5.63 |
| 3 正常 vs 外圈 | `outer_race_fault` 0.4412（6/6 命中）、`ball_fault` 0.3974（1 命中 / 3 未命中，conflict，alignment 0.2186 沉到末位） | 包络谱 BPFO 能量 0.2494（基线未检出），RMS 上升 823%，峭度 2.87 → 7.62 |
| 4 正常 vs 滚动体 | `ball_fault` 0.6283（`frequency_family=BSF`），过阈值候选仅此 1 条 | 包络谱 BPFI 4.20470e-4、BSF 3.92749e-4 并列（差 7.06%），首位候选的族落在并列族内故采信，详见第五节 |
| 1 / 6 | 空 | 场景 1 被「无显著变化」拦截；场景 6 被阈值拦截 |

## 四、单元测试要点与实测结果

### 数据校验（9.1 第 1~4 项）

| 用例 | 断言要点 | 实测结果 |
|---|---|---|
| `test_normal_sample_can_be_loaded` | 12000 点、单列/双列两种格式可读；由时间戳中位数反推采样率约 12000Hz | 通过。样例时间戳只保留 6 位小数，反推值约 12048Hz，因此断言用 `rel=0.01` |
| `test_single_column_format_is_supported` | 只有 `amplitude` 单列的 CSV 也能读 | 通过 |
| `test_corrupted_sample_reports_missing_value` | `codes[0]=="missing_value"`、`values.passed is False` | 通过。损坏样本 300 行中 7 行为空/`null`/`N/A`/`nan`/`--` |
| `test_corrupted_sample_raises_on_load` | 直接调 `vibration.extract_vibration_features` 抛 `VibrationError` | 通过 |
| `test_length_mismatch_over_min_samples_is_comparable_with_warning` | 12000 vs 8000 → 可比 + `warning` | 通过 |
| `test_length_mismatch_below_min_samples_is_not_comparable` | 12000 vs 300 → `insufficient_data` | 通过 |
| `test_sampling_rate_mismatch_is_rejected` | 6000Hz 时间戳 vs 入参 12000 → `sampling_rate_mismatch` | 通过 |
| `test_two_different_sampling_rates_are_both_flagged` | 两个样本各按自己的时间戳被标记，消息里推断值约 6000Hz 与约 12000Hz | 通过 |
| `test_invalid_sampling_rate_is_rejected` | `sampling_rate=0` → `invalid_argument` | 通过 |

### 特征计算与频谱峰值（9.1 第 5~6 项）

`test_feature_extraction_matches_independent_recomputation` 在测试侧用 `numpy/scipy` **独立重算**全部标量特征（RMS、峰值、峭度、波峰因子、主频、主频幅值）、5 个频带能量与频谱峰值列表，再与工具输出逐项比对；`test_spectrum_peaks_are_real_local_maxima_on_frequency_grid` 校验峰列表里每个频点都落在 `k·fs/N` 网格上；`test_envelope_analysis_is_optional_and_complete` 校验传/不传 `rotation_speed` 时 `envelope` 字段的有无与内部字段齐全；`test_bearing_fault_frequencies_follow_cwru_6205_formulas` 用 CWRU 6205-2RS JEM SKF 的几何公式复算 BPFO/BPFI/BSF/FTF；`test_spectrum_peak_detection_on_synthetic_sine` 用合成正弦信号验证主频检测。全部通过。

实测特征频率（1797rpm）：转频 29.95Hz、FTF 11.9293Hz、BSF 70.5838Hz、BPFO 107.364Hz、BPFI 162.186Hz。实测特征值（去均值口径）：内圈 RMS 0.288906 / 峭度 5.633704 / 主频 3587Hz；外圈 RMS 0.6756 / 峭度 7.623 / 主频 3445Hz；滚动体 RMS 0.1374 / 峭度 2.965 / 主频 3356Hz。

### MCP 错误返回与双后端一致性（9.2 第 7 项）

| 用例 | 断言要点 | 实测结果 |
|---|---|---|
| `test_bad_sampling_rate_error_envelope_is_identical_across_backends` | 两后端信封键集合、`code`、`field` 全等；数字归一后消息相等 | 通过。**存在一处真实差异**：`local` 的原始消息是「采样率必须是有限正数，收到：0」，`stdio` 是「…收到：0.0」——MCP 工具签名把 `sampling_rate` 声明为 `float`，整型 `0` 跨协议边界被强转成 `0.0` |
| `test_missing_file_error_envelope_is_identical_across_backends` | `code=file_not_found`，消息逐字相等 | 通过 |
| `test_compare_missing_field_error_is_structured` | 缺 `rms` 字段 → `code=missing_value`、`field="rms"` | 通过 |
| `test_stdio_backend_success_matches_local_backend`（`@pytest.mark.stdio`） | 两后端的 `rms`/`peak`/`kurtosis`/`crest_factor`/主频/主频幅值/频带能量/频谱峰/包络特征频率能量逐字段一致 | 通过 |

### RAG 空结果与报告完整性（9.2 第 8~9 项）

`test_rag_unrelated_query_returns_no_candidate`（无关查询 → 候选为空、`sources` 为空、`insufficient_evidence=True`）、`test_rag_high_threshold_returns_no_candidate`（阈值 0.99）、`test_rag_relevant_query_still_hits_candidate`（反例：相关查询仍命中 `inner_race_fault` 且 `score>=0.35`）、`test_rag_empty_result_marks_insufficient_evidence_without_guessing`（候选为空时不返回任何猜测的故障类型）、`test_report_fields_and_markdown_are_complete`（报告 11 个小节标题齐全、字段齐全、`report_markdown` 长度 > 1000 字符）全部通过。报告 11 个小节为：分析对象与输入信息、数据质量检查、正常与异常特征对比表、频谱或频带变化说明、RAG 检索到的故障候选、最终结论或无法确认说明、结论对应的振动证据、结论对应的知识来源、置信说明、建议复核的数据或现场检查项目、项目原型的适用边界。

### 接口层

| 用例 | 期望 | 实测 |
|---|---|---|
| `GET /health` | 200 | 200。默认配置下 `mcp_backend="local"`；`llm_available=false`、`retrieval_mode="bm25"`、`sample_count=5` |
| `GET /api/samples` | 200 且 5 条，不暴露服务器绝对路径 | 200，`ball_1797 / corrupted_1797 / inner_race_1797 / normal_1797 / outer_race_1797` |
| `POST /api/diagnose`（内置样例） | 200 + `status=ok` | 200、`ok`、`inner_race_fault` 0.4453、`tool_calls=4`、`sources=2`、`report_markdown` 6686 字符（含第 5 节条款核对表） |
| 缺 `normal_sample`/`abnormal_sample` | 400 | 400，`detail.code="missing_input"` |
| 未知样例 id | 400 | 400，`detail.code="sample_not_found"`，消息里列出全部可用 id |
| `sampling_rate=0` | 422 | 422，`detail.code="invalid_argument"`，消息含 `sampling_rate` |
| 损坏样本 | 200 但 `status=error` | 200、`error`、`missing_value`、`tool_calls=1`、`report=None` |

### 本次改造后的验证记录

这一节按改造引入的三个测试文件，逐条记录验证结论与真实输出要点。**这些用例不依赖任何 API Key**，向量相关路径用 `tests/helpers.py` 里离线注入的假 embedding 驱动。

**向量库与混合检索（`tests/test_rag_vector.py`，7 条）**

| 用例 | 验证结论与真实输出要点 |
|---|---|
| 切分块数与块大小 | 7 个知识条目切成 **14 个文本块**，每块正文最大 **499** 字符（≤ `RAG_CHUNK_SIZE=500`） |
| 相邻块重叠 | 相邻块重叠实测 **79** 字符（约等于 `RAG_CHUNK_OVERLAP=80`，断言用 `approx(80, abs=2)`） |
| 稳定块 ID | 块 ID 匹配 `^[a-z_]+#chunk-\d{3}$`，形如 `inner_race_fault#chunk-000`，重复切分结果一致 |
| 块元数据 | 每个块携带 `fault_type` / `title` / `source` / `source_url` / `locator` / `document` / `chunk_id` |
| Chroma 持久化写入与 `--status` | 用假 embedding 写入后集合可被 `--status` 读出条目与块 ID；幂等 `upsert` 不产生重复块 |
| 混合检索打分 | `final_score = 0.6 * vector_score + 0.4 * bm25_score`，两个分量与融合结果均落在 `[0, 1]` |
| 降级路径 | 注入 embedder 但索引目录不存在时 `retrieval_mode="bm25"`、`degraded=True` 且给出 `degraded_reason` |

**LangGraph 编排（`tests/test_graph_flow.py`，22 条）**

| 用例分组 | 验证结论与真实输出要点 |
|---|---|
| 图结构与节点顺序 | 正常路径节点访问顺序实测为 `["validate_node", "mcp_analysis_node", "rag_retrieval_node", "evidence_check_node", "llm_node", "report_node"]`，与 `trace.nodes` 一致 |
| 输入非法（8 个参数化用例） | `validate_node` 直接结束，`status=error`、`report=None`、`tool_calls=0`，不进入后续节点 |
| 模型超时降级 | 注入抛 `LLMError("timeout")` 的假客户端 → `llm_mode="template_fallback"`、`llm_calls=1`、`status=ok`，报告保留对比数据与 RAG 候选 |
| 返回非 JSON 降级 | 假客户端返回纯文本 → `llm_mode="template_fallback"`、`status=ok`，报告字段完整 |
| 候选外类型拦截 | 假客户端返回知识库不存在的 `gear_fault` → `llm_mode="template_fallback"`，最终结论**不含** `gear_fault` |
| 合法 JSON | 返回 `{"selected_fault_type","summary","reasoning","review_suggestions"}` → `llm_mode="llm"`、`llm_calls=1` |
| 错误路径契约 | 工具失败时直接结束，`report=None`、13 个响应键齐全 |

**接口层新增契约（`tests/test_api.py`，4 条）**

| 用例 | 验证结论与真实输出要点 |
|---|---|
| `test_health_reports_retrieval_mode_and_llm_availability` | `retrieval_mode ∈ {hybrid, bm25}`；当前无 Key 下稳定为 `bm25`，`llm_available=False` |
| `test_diagnose_trace_backend_matches_health` | 诊断响应 `trace.backend` 与 `/health` 的 `mcp_backend` 一致，`trace.retrieval` 与 `/health` 的 `retrieval_mode` 一致 |
| `test_normal_vs_normal_returns_no_sources` | 正常对正常：`status=insufficient_evidence`、`sources==[]`、`candidates==[]`、`report["conclusion_sources"]==[]`、`report["knowledge_candidates"]==[]` |
| `test_diagnose_still_works_when_chroma_unavailable` | 注入 embedder 但 Chroma 目录不存在时 `store.retrieval_mode="bm25"`、`degraded=True`；接口仍返回 200 与正确候选 `inner_race_fault` |

**SSE 事件流、受控追问与会话内自由问答（`tests/test_sse_ask.py`，22 条）**

| 用例 | 验证结论与真实输出要点 |
|---|---|
| `test_sse_normal_path_event_sequence` | 正常路径事件顺序：首事件 `diagnosis_started`、末事件 `diagnosis_completed`；6 个节点 started/completed 齐全且顺序与 `trace.nodes` 一致；4 次工具调用 started/completed 成对且顺序正确；分支事件 `payload.branch=llm_node`；完成事件 `payload.result` 携带完整响应（`inner_race_fault` 0.4453、`tool_calls=4`）；每条事件都含统一信封 9 字段且 `elapsed_ms` 为非负整数 |
| `test_sse_tool_failure_emits_failed_events` | 损坏样本：`mcp_analysis_node` 以 `failed` 状态结束并携带错误摘要；只有 1 次工具 started，不进入检索节点；末事件 `diagnosis_failed`，`error.code=missing_value`、`tool_calls=1` |
| `test_sse_insufficient_evidence_branch` | 正常 vs 正常：分支事件 `branch=report_node`、`status=skipped`，摘要含「无显著变化」；不出现 `llm_node` 事件；完成事件 `status=insufficient_evidence`，候选与来源为空 |
| `test_sse_llm_timeout_falls_back_to_template` | 注入超时假客户端：`llm_node` 发 `warning(status=fallback)` 且 `payload.llm_error.code=timeout`；`llm_mode=template_fallback`，已算出的候选与报告完整保留 |
| `test_sse_client_disconnect_is_safe` | 消费 3 个事件后主动断开：后台诊断照常跑完并保存会话（`result.status=ok`），事件轨迹完整保留 |
| `test_legacy_diagnose_stays_compatible_and_saves_session` | 旧 `/api/diagnose`：原有 13 键齐全、结论不变（`inner_race_fault` 0.4453、`tool_calls=4`）；新增 `session_id` 且会话可追问，`request_summary` 记录输入来源 |
| `test_sse_missing_input_returns_plain_400` | 流开始前输入缺失：返回普通 400 `missing_input` JSON，不产生半截事件流 |
| `test_ask_answers_from_session_context` | 追问「为什么判断为这个故障类型？」：200、`question_key=why_fault`、答案含 `inner_race_fault`、`facts_used` 非空、`context_scope` 说明事实范围 |
| `test_ask_all_seven_questions_are_allowed` | 7 个固定问题逐个追问全部 200，`question_key` 与后端定义一一对应 |
| `test_ask_rejects_out_of_scope_question` | 清单外问题：400 `question_not_allowed`，`detail.allowed_questions` 返回全部 7 条 |
| `test_ask_unknown_session_returns_404` | 未知会话 id：404 `session_not_found` |
| `test_ask_without_factual_basis_returns_no_support` | 错误会话问「为什么判断为这个故障」：`no_support=true`、答案固定为「当前诊断记录无法支持该结论。」、`facts_used=[]` |
| `test_chat_answers_candidate_question_from_session_facts` | 未配置模型（`available=False`）时问「候选故障有哪些」：`mode=deterministic`、答案含 `inner_race_fault` 与分数、`facts_used` 非空、`context_scope` 说明事实范围 |
| `test_chat_input_topic_uses_recorded_condition` | 问输入条件（转速/采样率/工况）：回答直接引用会话记录的请求摘要，不重新计算 |
| `test_chat_controlled_question_falls_back_to_deterministic_answer` | 问 7 个受控问题之一且模型不可用：走确定性直答，与受控追问答案等价 |
| `test_chat_accepts_model_answer_inside_session_scope` | 注入假模型（`ScriptedLLMClient`）：回答只在会话事实内 → `mode=llm`、`facts_used` 非空、`model_error=null`；提示词里含候选与来源摘要 |
| `test_chat_rejects_fault_type_outside_candidates` | 假模型回答里出现知识库已知但不在候选内的故障类型（如 `unbalance`）→ `answer_rejected`、整段作废、降级为确定性回答 |
| `test_chat_rejects_url_outside_session_sources` | 假模型回答里出现会话来源之外的 URL → 同样 `answer_rejected` 并降级 |
| `test_chat_model_timeout_falls_back` | 注入超时假模型：`model_error.code=llm_timeout`、`mode=deterministic`、回答仍基于会话事实 |
| `test_chat_error_session_answers_from_error_facts` | 错误会话（`missing_value`）提问：摘要含错误码与中文原因，能回答「为什么失败」 |
| `test_chat_empty_question_returns_400` | 空问题：400 `empty_question` + 中文原因 |
| `test_chat_unknown_session_returns_404` | 未知会话：404 `session_not_found` + 中文原因 |

## 五、失败案例与修正：滚动体样本曾被误判为内圈故障

场景 4 是保留的真实失败案例：`normal_1797` vs `ball_1797`（CWRU fileId 118，0.007 inch 故障直径，公开标签 `ball_fault`）。它经历了三个阶段：**①修正前**系统输出 `inner_race_fault`（score 0.4614），与真正的内圈样本得分完全相同；**②上一轮**加入量级余量判据后降级为「无法确认」，但检索首位仍是内圈条目；**③本轮**（判据表达 + 检索排序 + 并列族采信）检索首位修正为 `ball_fault`（score 0.6283），结论 `ball_fault`、置信度 low。回归测试为 `tests/test_e2e_diagnose.py::test_normal_vs_ball_fault_tie_broken_by_top_candidate`（断言 `status=ok`、top1 `ball_fault`、`frequency_family=BSF`、置信 low 与 `trace.tied_resolution`）。

### 确切数字（实测）

| 项 | 数值 |
|---|---|
| CWRU 公开标签 | `ball_fault`（fileId 118，0.007 inch 故障直径） |
| ①修正前候选 top1 | `inner_race_fault`，score **0.4614**（与场景 2 的内圈样本得分完全相同） |
| ②上一轮结果 | `status=insufficient_evidence`，`candidates`/`sources` 为空，结论以「无法确认」开头；检索首位仍是 `inner_race_fault`（滚动体条目未登顶） |
| ③本轮结果 | `status=ok`，候选 top1 **`ball_fault` 0.6283**（`frequency_family=BSF`，标题「滚动轴承滚动体（钢球）故障」），结论 `ball_fault`，置信度 **low** |
| 本轮过阈值候选数 | **1 条**（`ball_fault`），缺少交叉印证 |
| `trace.tied_resolution`（本轮） | `{"fault_type": "ball_fault", "frequency_family": "BSF", "tied_labels": ["BPFI", "BSF"], "margin_ratio": 1.0706}` |
| 包络谱 BPFI 族能量（1/2/3 阶） | **0.000420470293**（4.20470e-4） |
| 包络谱 BSF 族能量（1/2/3 阶） | **0.000392748873**（3.92749e-4） |
| BPFI / BSF 比值 | **1.0705830669563756**（BPFI 仅高约 **7.06%**） |
| 包络谱 BPFO 族能量 | 0.000351333806 |
| 包络谱 FTF 族能量 | 0.000150947743 |
| 基线（正常样本）对应能量 | BPFI 4.6789e-7、BSF 2.80984e-7、BPFO 3.18644e-7、FTF 1.06747e-7 |
| `dominant_characteristic`（本轮） | `label=BPFI`、`ambiguous=true`、`margin_ratio=1.0706`、`tied_candidates=[BPFI@162.186Hz 4.20470e-4, BSF@70.5838Hz 3.92749e-4]` |
| 滚动体包络谱最大峰 | 162.0Hz（幅值 9.638174e-3）；次高在 45.0Hz（1.0580502e-2）；BSF 理论位置 70.58Hz 附近为 69.0Hz（7.5422732e-3） |
| `fault_frequency_matches`（本轮） | **3 条**：BPFI 1/2/3 阶，对应最近峰 162.0 / 329.0 / 481.0Hz（修正前仅 1 条） |
| `spectrum_peaks` 数量（本轮） | 24~25 个（修正前最多 10 个） |
| 时域 | RMS 0.1374（基线 0.0732，约 1.878 倍）、峰值 0.5357、峭度 2.965、波峰因子 3.898 |

### 根因

1. **包络谱上 BPFI 族能量压过了 BSF 族。** 该样本 BPFI(1/2/3 阶) 累加能量 4.20470e-4，BSF(1/2/3 阶) 为 3.92749e-4，前者高 7.06%。`dominant_characteristic` 取各特征频率族中的最大值，于是输出 BPFI，`reason` 也写成「包络谱中 BPFI（162.2Hz）及其 2、3 阶谐波频带能量为各特征频率中最高」，并作为证据进入检索查询。
2. **检索查询文本因此被 BM25 引向内圈。** `build_query` 把上一步的结论转成中文文本，出现「BPFI 最突出」「谱峰与 BPFI 对齐」等措辞，`inner_race_fault` 知识条目与这些词高度重合，得分 0.4614——恰好与真正的内圈样本得分相同，说明这条查询对两个不同故障给出了几乎一致的文本特征。
3. **`bsf` 的口径没有把「2×BSF 为主」表达出来。** 对外输出的 `characteristic_frequencies["bsf"]` 是 **1×BSF = 70.5838Hz**，而 `ball_fault` 知识条目的判据是「频谱在 **2×BSF**（=141.1676Hz）附近出现峰值」。查询文本里 BSF 族只以「含 2、3 阶谐波的包络能量」这种笼统表述出现，没有「2×BSF 处峰值最高」这一关键区分点，因此知识条目里最有辨识度的判据没有被查询命中。
4. **（本轮新增认知）知识库条目的判据表达与特征频率族元数据缺失。** 滚动体条目原来的判据侧重「2×BSF 附近出现峰值」，没有把「包络能量分散、无单一主导族」这一与并列现象自洽的表述写出来，条目本身也没有声明自己属于哪个特征频率族（`frequency_family`），导致并列时无法用「首位候选声明的族是否落在并列族内」这一规则去打破并列。补齐判据表达并声明族元数据后，检索首位由内圈条目变为 `ball_fault`（0.6283）。

**需要更正一处常见表述**：说「`bsf` 只计算球自转频率、未计算 2×BSF」在**能量口径上并不成立**。`vibration._ENVELOPE_HARMONICS = (1, 2, 3)`，包络能量桶 `characteristic_energy["bsf"]` 已经把 1×BSF(70.58Hz)、2×BSF(141.17Hz)、3×BSF(211.75Hz) 一起求和；`_match_fault_frequencies` 也按 (1,2,3) 阶逐阶去对齐频谱峰。**真正的问题在第 3 点**：对外暴露的特征频率与知识库判据之间的口径没有对齐（一个给 1×BSF，一个按 2×BSF 判），查询文本没有把「2×BSF 为主」传递出去；再加上第 1 点 7.06% 的能量差本身就落在噪声量级，`dominant_characteristic` 的「取最大值」规则对此没有余量。（第 1、3 点按上一节四项修正处理，第 4 点按本轮判据表达 + 检索排序 + 并列族采信处理。）

### 已实施的修正（四项）

| # | 修正 | 落点 |
|---|---|---|
| a | **量级余量判据 + 并列族采信**：最高/次高能量比低于 `_DOMINANT_MARGIN_RATIO = 1.15`（即差 < 15%）时不宣称「最突出」，改输出 `tied_candidates` 并列候选并置 `ambiguous=True`；本轮在此基础上新增 `frequency_family` 元数据机制与并列族采信规则——滚动体条目补「包络能量分散、无单一主导族」判据并声明 `frequency_family: BSF`（内圈/外圈/保持架分别声明 `BPFI`/`BPFO`/`FTF`），检索候选透出该字段，编排层 **仅当检索首位候选声明的族落在并列族内** 才采信该候选出结论，族不匹配或条目未声明族时仍按证据不足处理 | 知识条目 `app/rag/knowledge/*.md` 的判据与 front-matter；`app/rag/store.py` 透出 `frequency_family`；`app/mcp/vibration.py::_characteristic_comparison`；`app/agent/graph.py::_tied_family_resolution`（`_ambiguous_dominant` 保留为兜底）；`app/agent/report.py` 在并列采信时强制把置信度下调为 low 并写明并列依据与复核建议 |
| b | **BSF 族口径对齐**：对外同时给出 1×BSF 与 2×BSF，新增 `characteristic_frequencies["bsf_2x"]`，避免「知识库按 2×BSF 判、接口只给 1×BSF」的口径错位 | `app/mcp/vibration.py::_envelope_analysis` |
| c | **检索查询按族聚合**：`fault_frequency_matches` 改为按特征频率族合并成一条（同族多阶只出现一次），并列时把两个族一并写入查询，不再用「最突出」这类单点措辞 | `app/rag/retriever.py::build_query` |
| d | **扩大谱峰保留范围**：除全谱幅值前 5 外，额外保留「特征带（最高特征频率的 3 阶谐波）内幅值前 `_CHARACTERISTIC_PEAK_LIMIT = 20`」，两者取并集，谱峰上限由 10 提到 25 | `app/mcp/vibration.py::extract_vibration_features` |

**修正前后对比（同为实测）**：

| 场景 | ①修正前 | ②上一轮修正后 | ③本轮修正后 |
|---|---|---|---|
| 4 正常 vs 滚动体 | `ok`，误判 `inner_race_fault` 0.4614 | `insufficient_evidence`，结论「无法确认」+ BPFI/BSF 并列证据（检索首位仍是 `inner_race_fault` 0.432） | `ok`，`ball_fault` **0.6283**（`frequency_family=BSF`），置信 low，按并列族采信 |
| 3 正常 vs 外圈 | `fault_frequency_matches` 实测为空列表 | 6 条（BPFO 1/2 阶、BPFI 1/2/3 阶、BSF 3 阶），结论仍为 `outer_race_fault` 0.4415 | 不变 |
| 2 正常 vs 内圈 | `inner_race_fault` 0.4614 | `inner_race_fault` 0.4463，top1 与结论不变（新增一条候选分差复核告警） | 不变 |
| 1 正常 vs 正常 | `insufficient_evidence`，4 次工具调用 | 不变 | 不变 |

### 仍未解决

- **并列本身仍然存在，不代表主导族已确定**：BPFI 与 BSF 包络能量只差 7.06%，本轮解决的是「检索首位不再落到内圈条目」，结论来自「知识库条目声明的特征频率族与并列族一致」这一采信规则，**不代表主导族已确定**，所以置信度只有 low，报告必须提示人工复核并列族。
- **采信规则依赖知识库条目的 `frequency_family` 元数据**：条目未声明该字段时不会采信，仍判证据不足（这是防回归守卫）。
- **谱峰对齐证据客观上仍偏向内圈**：原始频谱峰只与 BPFI 1/2/3 阶对齐（162 / 329 / 481Hz），而 2×BSF = 141.17Hz 最近峰为 148Hz（偏差 4.84%），超出 2% 容差；这一条未动。
- **弱冲击信噪比低、共振带按奈奎斯特频率固定比例（0.33~0.83）选取（未用谱峭度 / kurtogram 自适应选带）这两个根因未动**，本次未做。
- **ball 场景过阈值候选只有 1 条**，缺少交叉印证；内圈 / 外圈场景仍会触发「前两名候选分数接近」复核告警（既有行为，未变）。
- **混合检索（hybrid）路径本轮仍未验收**。

## 六、调用次数、响应时间与后端性能

模型调用与工具调用次数直接取自 `trace`：

| 场景 | `llm_mode` | `llm_calls` | `tool_calls` | `elapsed_ms`（local 后端） |
|---|---|---|---|---|
| 1 正常 vs 正常 | `template` | 0 | 4 | 98 |
| 2 正常 vs 内圈 | `template` | 0 | 4 | 88 |
| 3 正常 vs 外圈 | `template` | 0 | 4 | 84 |
| 4 正常 vs 滚动体 | `template` | 0 | 4 | 89 |
| 5 正常 vs 损坏样本 | `template` | 0 | 1 | 27 |
| 6 知识库未命中 | `template` | 0 | 4 | 96 |
| 7 模型超时 | `template_fallback` | 1 | 4 | 116 |

- **模型调用次数**：本测试套件在 `tests/conftest.py` 中主动清空模型环境变量，因此场景 1~6 走模板报告，场景 7 使用假客户端验证超时降级。真实模型路径另由 `scripts/smoke_llm.py` 验证：2026-09-25 本地配置下 `llm_mode="llm"`、`llm_calls=1`、`status=ok`，模型调用约 9.4 秒。
- **工具调用次数**：正常路径固定 4 次（校验 + 正常特征 + 异常特征 + 对比），损坏样本只 1 次即停机，远低于 `MAX_TOOL_CALLS=8`。

两个后端的真实性能（同一次「正常 vs 内圈」诊断）：

| 后端 | 单次完整诊断 `elapsed_ms` | 单次工具调用墙钟时间 | 说明 |
|---|---|---|---|
| `local`（进程内） | 84~116 | 毫秒级 | 测试默认后端 |
| `mcp:stdio` | **8346**（约 8.3 秒；历史实测另有 10773 / 11202 / 11223，随负载波动） | **2~3s** | 每次工具调用新建一次 stdio 连接 + 子进程；4 次调用即约 8~12 秒 |

结论：`local` 与 `mcp:stdio` 两次诊断的**结论、候选、分数、来源完全一致**（同为 `inner_race_fault` 0.4453、`sources=2`），差异只在耗时。演示时可切 `local` 提升响应速度，`stdio` 保留用于验证真实 MCP 协议链路。

## 七、已知局限与未验证项

### 2026-09-26 展示体验复测

- 本轮完整测试：80 passed in 21.23s；新增慢模型等待期间健康检查可响应的回归测试。
- 浏览器真实模型路径：外圈样例 status=ok、llm_mode=llm、tool_calls=4、llm_calls=1，最终复测 elapsed_ms=10272。
- 1440 px 桌面与 390 px 手机实际操作通过；下载报告与接口内容一致，演示场景清除旧上传，实际节点轨迹可展开。
- 注入 HTTP 503 验证页面错误原因可见、上一份结论被清除；此项为模拟错误，不代表真实服务故障统计。
- 手机图表横向滚动，频谱坐标标签位于 SVG 画布内；无页面横向溢出或运行时错误。

### 2026-09-26 Agent 工作台改造复测（SSE + 追问）

- 本轮完整测试：**92 passed in 24.54s**（80 条原有用例零回退 + 12 条 SSE/追问新增用例）。
- 桌面 1440px 浏览器实测（`local` 后端、模型可用、检索 BM25）：SSE 时间线实时更新——6 个节点徽标「等待 → 执行中 → 已完成」逐个翻转、4 次工具调用明细（含一句话摘要与 T+毫秒耗时）、绿色分支条「证据充分 → 调用模型生成诊断」。
- 「Agent 决策依据」卡片 6 部分渲染正确：候选类型与相似度、关键数值变化、主要频率证据、RAG 命中数与来源、置信度原因、降级与复核提示。
- 受控追问实测：诊断完成后追问框启用并显示会话 id 前 8 位；选择「为什么判断为这个故障类型？」返回答案并渲染 `facts_used` 标签（`candidates[0]`、`comparison.evidence`、`comparison.dominant_characteristic`、`trace.llm_mode`）。
- 11 小节报告、频带与频谱 SVG 图表、数据质量、知识来源均正常渲染；浏览器控制台无 JS 运行时错误。
- 390px 移动端实测：无页面横向溢出，表格在容器内横向滚动，时间线与追问区纵向堆叠，图表标签位于 SVG 画布内。
- 节点「跳过」与「回退」徽标语义由用例覆盖（`test_sse_insufficient_evidence_branch`、`test_sse_llm_timeout_falls_back_to_template`），本轮浏览器演示走的是证据充分 + 真实模型路径，未在浏览器中人为触发这两种徽标。
- **如实说明**：前端「SSE 意外中断 → 自动降级同步接口」的降级路径未在浏览器中人为触发验证（本轮 SSE 连接全部正常完成）；断开安全本身由 `test_sse_client_disconnect_is_safe` 覆盖，时间线重建逻辑经代码走查。另：验收时浏览器曾缓存旧版页面，需带 cache-bust 参数刷新后复测，服务端返回的为新版本。

### 2026-09-26 结果区卡片目录改版复测

- 结果区由「顺序堆叠」改为「卡片目录 + 点击切换」：8 张卡片（执行过程 / Agent 决策依据 / 关键证据 / 信号分析 / 知识检索与来源 / 数据质量与复核 / 完整报告 / 诊断追问（受控）），同一时刻只显示所选卡片内容；状态横幅与结论摘要常驻在目录之上。注释性文本（输入区说明句、状态行、文件说明、黄色安全提示等）已移除，免责声明降级为页脚一行小字。
- 桌面 1440px 实测：诊断开始后默认停在「执行过程」卡片，6 个节点徽标最终全部「已完成」；卡片元信息（已完成 N/6 节点、候选条数、RMS 倍数、报告模式等）随结果填充；逐卡点击切换正常，仅显示被点中的内容。
- 390px 移动端实测：卡片每行 2 个（scrollWidth 372 = clientWidth 372），切换、图表与追问可用，无横向溢出。
- 本轮回归：**92 passed in 19.05s**（80 条原有用例零回退 + 12 条 SSE/追问用例）；本次只改前端静态页，接口契约与后端未变。
- 控制台如实记录：页面加载后控制台为空、无 JS 运行时错误（无 Uncaught/SyntaxError）；一次完整诊断后固定出现 1 条 `net::ERR_ABORTED` 网络层条目（指向 `/api/diagnose/stream`）。插桩验证页面 JS 未调用 `AbortController.abort` 或 `ReadableStreamDefaultReader.cancel`，服务端日志该请求 `200 OK` 完成、结果完整渲染，判定为流式响应在浏览器网络栈的收尾分类，改版前后均如实存在；成因未进一步定位，如实保留。

### 2026-09-26 三栏工作台与自由问答验收

- 本轮完整测试：**102 passed in 20.10s**（92 条原有用例零回退 + 10 条会话内自由问答新增用例）。
- 三栏布局（Chrome 无头浏览器 1440×1000 实际渲染，经 DevTools 协议读取计算样式）：`.shell` 计算列宽 **250px / 798px / 320px**；左右两栏 `position: sticky`（`top` 12px、`max-height` 881px），左栏「结果目录」8 张卡片纵向排列、下方为「会话窗口」列表，右栏为「与模型对话」；`scrollWidth − clientWidth = 0`，无横向溢出。全页截图核对：左（目录 + 会话窗口）／中（分析输入、状态横幅、结论摘要、执行时间线）／右（对话）同屏可见。
- 桌面宽度完整流程实测（同一会话内连续执行）：点「正常 vs 外圈故障」→ 状态横幅「分析完成 置信等级：低」→ 左栏会话窗口新增 1 条（`normal_1797 vs outer_race_1797`，卡片元信息全部填充）；卡片切换后仅显示被点中的视图（点「关键证据」时可见面板仅 `evidence`）；右栏提问「候选故障有哪些？」返回模型回答（候选 `outer_race_fault` 0.4415、证据描述与来源链接），标记「模型生成（已通过会话事实校验）」。
- 窄窗（387px，内置浏览器）复测：三栏降级为单列、卡片目录转横向滚动、对话区落到页面底部；两次诊断生成 2 个会话窗口，切换会话时结论、时间线、对话记录按会话重放；无横向溢出。
- 控制台：桌面宽度下 **0 条 JS 异常、0 条网络请求失败**。验收中发现控制台 error 级条目 1 条，起因为页面未定义图标导致的 `favicon.ico` 404；已改为内联 data-URI SVG 图标（不引入 CDN、不新增文件），修复后复测 error 条目为 0。
- **如实说明（一）**：桌面宽度 3 次诊断中有 1 次报告走模板回退（页面标记「模板生成」，即 `llm_mode=template_fallback`，报告字段完整、数据与候选不变）；同轮另 2 次及接口直连均为 `llm`（8.4s、报告 4117 字）。回退属既有设计（模型调用失败 → 模板），该次模型调用失败的具体原因未进一步定位。
- **如实说明（二）**：内置浏览器窗口宽度固定为 387px，无法拉宽，故桌面宽度验收由无头 Chrome + DevTools 协议完成，不是人工交互式操作；内置浏览器路径下的功能验收在 387px 完成。

### 2026-09-26 会话持久化与刷新恢复验收

- 本轮完整测试：**111 passed in 18.86s**（102 条原有测试零回退 + `tests/test_session_persistence.py` 9 条新增）。
- 落盘存储：会话与对话消息写入 SQLite（默认 `data/sessions.db`；测试在 `conftest.py` 里把 `SESSIONS_DB_PATH` 指向临时目录隔离，不写仓库数据）。新增用例覆盖「重开 `SessionStore` 后会话、事件轨迹与结果仍在」「会话列表接口返回摘要与 `message_count`」「详情接口返回完整记录」「未知会话 id 返回 404」「追问与自由问答各落一条消息」「对话历史轮数窗口跟随 `CHAT_HISTORY_TURNS`」「超过 50 个会话时淘汰最旧会话及其消息」。
- 浏览器实测（1440px 无头浏览器；服务端 `local` 后端、模型可用）：跑一次「正常 vs 内圈故障」→ 状态横幅「分析完成」、6 个节点时间线完成、左栏新增会话窗口；右栏提问一次（真实模型，`mode=llm`）后**刷新页面**，左栏会话列表、状态横幅、6 个时间线节点与右栏 4 条对话消息（`chat-user` / `chat-assistant` 各 2）全部恢复，并在恢复后的会话里继续提问成功。
- **进程重启不丢**：停掉 uvicorn 再启动，`GET /api/sessions` 仍返回该会话（`status=ok`）；详情实测 `events=23`、`messages=4`、`status=ok`，消息顺序为 `['chat:user','chat:assistant','chat:user','chat:assistant']`。
- 无横向溢出（`horizontalOverflow: false`）；控制台无 JS 运行时错误、无 404，仅保留既有的 `net::ERR_ABORTED /api/diagnose/stream` 流式收尾条目（见上一小节，非本次引入）。
- **如实说明**：多轮上下文只在提示词层面生效，未做「对话轮数很多时回答质量」的人工评测；SQLite 为单机文件，仅由 `threading.Lock` 保证单进程内串行，未做多副本与并发写入压测。

### 2026-09-26 原始曲线与波形音频验收

- 本轮完整测试：**125 passed in 24.10s**（116 条原有测试零回退 + `tests/test_signal_series.py` 9 条新增）。
- **曲线序列（降采样、真实抽点结果）**：`include_series` 默认 `false` 时 `features` 里没有 `series` 键（旧契约逐字不变）；编排层传 `true` 后实测 `waveform` 600 个分桶、`spectrum` 600 点、`envelope` 400 点且最大频率恰为 **1000.0Hz**。幅度谱抽点保峰实测：`max(spectrum.amplitude)` 与整条全谱的幅值最大值相等（四舍五入到 6 位小数内），全谱 argmax 对应的频点也出现在抽点频率里——说明特征频率的窄峰没有被等间隔抽点抹掉。序列取自**去均值后**的分析信号，与 RMS / 峭度同口径。
- **序列不污染标量特征**：加 `include_series` 前后，`rms` / `peak` / `kurtosis` / `crest_factor` / `dominant_frequency` / `band_energy` / `spectrum_peaks` 逐字段完全相等（有专门用例断言）。
- **WAV 实测**：单路 24044 字节，文件头为 `RIFF` / `WAVE`，1 声道、每样本 2 字节、12000Hz、12000 帧，峰值落在满量程 `[0.9×32767, 32767]` 区间（按全局峰值归一化，未削波）。
- **体积口径**：含 `series` 的 `features` JSON 约 **64KB**，一次完整诊断响应约 **89KB**；PCM 不在其中——响应体与 SQLite 里都不含 `wav_base64`（有用例断言整份响应 JSON 搜不到该字段）。
- **音频端点实测**（服务重启后打真实接口）：`status=ok`、`tool_calls=4`；`series` 键为 `audio / duration_seconds / envelope / sampling_rate / spectrum / waveform`，`audio.available=true` 且响应体无 PCM；`GET /api/sessions/{id}/audio/normal.wav` 与 `.../abnormal.wav` 均 **200**、`Content-Type: audio/wav`、24044 字节；`side.wav`（非法 slot）、未知 `session_id`、含 `../` 的路径均 **404 `audio_not_found`**。
- **降级路径**：损坏样本在数据校验阶段就停机，`features` 里既没有 `series` 也没有音频，`tool_calls` 仍为 **1**；音频落盘失败时只在 `trace.warnings` 追加一条提示，诊断照常出结论，前端播放器显示「无可播放音频」。
- **浏览器实测**（`http://127.0.0.1:8000/`，一键演示「正常 vs 内圈故障」后点开「信号分析」卡片）：四段内容（原始信号时域波形 / 原始波形音频 / 幅度谱 / 包络谱）齐全；页面共 6 个 `svg.chart`，其中本轮新增 3 张共 8 条 `polyline`（波形两路各上下包络 = 4 条，幅度谱与包络谱各 2 条，点数 600×6 + 400×2）；2 个 `<audio>` 的 `src` 指向正确的会话与 slot；**控制台 0 报错、网络无 4xx/5xx**。
- **如实说明（一）**：波形/幅度谱/包络谱都是给前端画图用的**降采样可视化**，不是完整数据，也不参与任何特征计算与打分；音频同理，只用于辅助听冲击与调制形态。
- **如实说明（二）**：极窄视口（`clientWidth` 约 284px）下左栏导航卡片的行内容仍会超出容器宽度（`scrollWidth` 约 441px），这是**既有现象、非本轮引入**（信号分析面板自身的 `scrollWidth` 仅约 89px）；本轮顺手把音频卡片网格从写死两列改为 `minmax(180px, 1fr)` 自适应，避免新增内容在窄屏被压扁。

### 2026-09-26 曲线区间查看（拖拽放大与复位）验收

- **交互实现**：三张曲线图（时域波形 / 幅度谱 / 包络谱）各自维护一个视图区间，在绘图区内横向拖拽即框选放大，双击图内或点图右上角「重置视图」回到全段。**纯前端重绘已有序列**：不发新请求、不改后端接口、不动任何特征数值；放大后纵轴按当前区间内的点重算，横轴刻度按区间跨度自适应小数位（跨度 ≥20 取整、≥2 保留 1 位、更窄保留 2 位）。
- **实测（无头 Chrome，`http://127.0.0.1:8000/`，一键演示「正常 vs 外圈故障」后点开「信号分析」卡片）**：3 张图按顺序为波形 / 幅度谱 / 包络谱，初始都没有「重置视图」按钮。包络谱拖拽 5%→25% 区间后横轴刻度由 `0 / 204 / 408 / 611 / 815 / 1019 Hz` 变为 `51 / 92 / 132 / 173 / 214 / 255 Hz`，折线点数由 4790 降到 970；幅度谱拖拽 40%→70% 后刻度由 `0 / 1223 / … / 6117` 变为 `2447 / 2814 / 3181 / 3548 / 3915 / 4282`；波形拖拽 10%→30% 后刻度由 `0 / 122 / … / 611` 变为 `61 / 86 / 110 / 134 / 159 / 183`，**区间内的细节被摊开**。
- **复位**：波形双击图内后横轴刻度逐字回到初始 `0 / 122 / 244 / 367 / 489 / 611`、「重置视图」按钮消失、折线点数回到 7186（与放大前完全一致）；幅度谱点「重置视图」后刻度逐字回到 `0 / 1223 / 2447 / 3670 / 4894 / 6117`、按钮消失。放大其中一张不影响另外两张（包络谱的放大状态在波形复位后仍然保持），三张图各自独立维护区间。
- **回归与边界**：全量 `pytest -q` 仍为 **125 passed**（本轮只改前端模板，无用例变更）；全新加载页 + 跑一次完整诊断后控制台 **0 条消息、0 条 error、无未捕获异常**，网络无 4xx/5xx；1440px 视口下无横向溢出，两个音频播放器照常保留。
- **如实说明**：拖拽只在**鼠标指针**下生效（触摸指针直接忽略），避免与窄屏下 `chart-scroll` 的横向滚动抢手势；框选区间下限为 12px 像素宽度，更窄视为点击而不缩放；放大后若区间内没有抽点（极窄区间落在两个分桶之间）会显示「该区间无数据点」，此时仍可用「重置视图」返回。放大只是把已抽样的可视化序列摊开，**不改变任何判据与结论，也不会因此获得更细的原始数据**。

### 2026-09-26 判据条款化与条款级核对验收

- 本轮完整测试：**138 passed in 20.86s**（125 条原有测试零回退 + `tests/test_rag_criteria.py` 13 条新增）。
- **知识侧**：7 个条目共 **24 条具名判据条款**（内圈 6 / 外圈 6 / 滚动体 5 / 保持架 3 / 正常 2 / 不对中 1 / 不平衡 1），每条含 `id` / `claim` / `metric` / `op` / `value`（可选 `family` / `order`）。词表守卫用例逐条断言所有条款的 `metric` / `op` 都落在 **15 个指标 × 6 个比较符**的受控词表内、条款 `id` 全局不重复；词表外的 `metric`（如 `magic_score`）或比较符（如 `≈`）一律记 `unknown`，不判「不成立」。
- **核对引擎实测（三态与因子）**：内圈 top1 `inner_race_fault` score 0.4453、条款 **6/6 命中**（factor 1.0，alignment_score 0.4453，排序不被改写）；外圈 top1 `outer_race_fault` **6/6 命中**（0.4412）；同场景 `ball_fault` **1 命中 / 3 未命中**（`conflict`，factor 0.55，0.3974 → **0.2186 沉到末位**）；滚动体 top1 `ball_fault` **5/5 命中**（0.6283，其中 `ball-dispersed` 与 `ball-tie-bsf` 两条判据命中）。三个场景 top1 与 CWRU 真实标签一致。
- **阈值口径未变**：过滤仍按检索分 `score`，`alignment_score` 只用于重排；`test_evaluate_candidates_sinks_entry_contradicted_by_data` 直接构造「0.6283 但判据冲突」与「0.4453 但判据一致」两条候选，验证后者被排到前面。
- **落点链路**：条款证据写入 `trace.criteria_summary`（每候选的 `state` / 命中 / 未命中 / 未核对 / 已核对 / `factor` / `score` / `alignment_score`）与 `trace.criteria_conflict`（首位候选冲突时记未命中条款清单，本批三场景均无冲突故为 `null`）；报告第 5 节新增**条款核对表**（命中 / 未命中 / 未核对 + 期望值与实测值 + locator），模板路径实测 `report_markdown` 6686 字符、正文含「判据条款核对」与条款 id；冲突候选不采信、置信度强制 low 并追加人工复核建议（用外圈样本上的滚动体条目验证）。
- **如实说明**：条款仅 24 条、词表固定，覆盖不了全部工况；`unknown` 不罚分，指标缺失时核对强度会变弱；本轮查询文本接入异常特征后检索分发生小幅位移（内圈 0.4463 → 0.4453、外圈 0.4415 → 0.4412、第二名 0.3987 → 0.3986），候选集合与结论未变；条款核对只证明「条目写的可核对条件与本次数据一致」，不是物理层面的故障确认，也不构成准确率结论。
- **浏览器端到端复测（真实模型路径，2026-09-26 19:10）**：页面刷新后先自动恢复历史会话（此时报告卡 7397 字符、含条款），随后点「开始分析」跑「正常 vs 外圈故障」，左栏会话窗口由 **9 增至 10**，新会话 `8213756d0b07`。逐项核对：横幅「分析完成 · 置信等级低」；数据质量 6 项全部通过；特征对比表含 RMS / 峰值 / 峭度 / 波峰因子等行与频带能量（2400-3600Hz 由接近 0 升至 0.316）；**「完整报告」卡片正文 7243 字符，含「判据条款核对」与条款 id `outer-bpfo-peaks`，11 个二级标题齐全**；「知识检索与来源」候选行含判据核对（`outer_race_fault` 0.4412、6 命中；`ball_fault` 0.3974 → alignment **0.2186**、1 命中 / 3 未命中 / 1 未核对）；硬刷新后报告卡仍为 7243 字符且含条款（恢复路径一致）；接口 `/api/sessions` 按新→旧排序，最新会话 `8213756d0b07` 的 `report_markdown` 7243 字符、含条款。控制台仅 1 条既有的 `net::ERR_ABORTED /api/diagnose/stream`（流式响应收尾分类，非本轮引入），无 4xx/5xx。
- **如实说明（一次无效测量被修正）**：首轮浏览器验收曾把页面**自动恢复的历史会话**（06:45 生成的旧格式报告 4845 字符，候选无 criteria 字段）误当作「本次点击产生的新结果」，据此得出「完整报告卡片缺条款」的错误结论；复测改用「会话窗口数量 9→10 + 报告卡含条款 id」双判据定位新结果后，同一检查项通过。该现象提示：演示时页面加载会自动恢复最新历史会话，验收前需先确认当前展示结果的会话归属。

### 保留的验证边界

以下内容如实记录，其中「未验证」是明确的负面结论，不代表已通过。

- **真实 embedding 未作为线上默认路径验证**：当前公开演示显式关闭 `ENABLE_HYBRID_RETRIEVAL`，即使配置了 embedding 也使用 BM25；向量/混合检索路径由 `tests/helpers.py` 里的假 embedding 覆盖写入、元数据、打分公式、归一化与降级机制。真实 embedding 的语义召回质量未评估，需完成独立评测后再开启。
- **真实模型成功路径已验证但稳定性仍未评估**：2026-09-25 `scripts/smoke_llm.py` 在本地配置下跑通 `llm_mode="llm"`；超时、非 JSON 和候选外故障类型仍由假客户端覆盖，长期延迟、失败率和成本尚未统计。
- **Docker 未构建（未验证）**：仓库提供 `Dockerfile` 与 `.dockerignore`，但本机未安装 Docker（`docker --version` 报命令不存在），**镜像既未真实构建、也未启动容器验证 `/health`**，只做了静态检查。
- **会话持久化未做多副本与压力验证（部分未验证）**：会话与消息落盘在单机 SQLite（`data/sessions.db`），服务重启不丢；但多副本部署不共享，也未做并发写入压力测试与超大量会话下的列表查询性能验证。多轮上下文只带最近 `CHAT_HISTORY_TURNS`（默认 3）轮、单条截断 400 字符，更早对话不进入提示词。
- **波形音频旁挂文件是单机本地盘的（部分未验证）**：WAV 落在 `AUDIO_DIR`（默认 `data/audio`），多副本不共享；按 `AUDIO_MAX_FILES`（默认 200）淘汰最旧的旁挂文件，被淘汰后旧会话再播放会拿到 404（页面降级为「无可播放音频」，诊断结果不受影响）。未做「大量并发诊断同时写盘」的压测，也未测跨平台（非 Windows）下的文件写入行为。
- **曲线与音频是降采样可视化，不参与判据**：波形为 600 桶 min/max 包络、幅度谱 600 点、包络谱裁到 0~1000Hz 共 400 点，虽然抽点规则保证特征频率峰被保留，但**不是完整数据**；音频按全局峰值归一化、且是去均值信号的重放。三者都不参与特征计算与打分，也没有做「曲线/音频是否改善人工判读效率」的量化评估。
- **频率分辨率 = fs/N**：1 秒样本（N=12000、fs=12000）分辨率恰为 **1Hz**，理论 BPFI 162.186Hz 实际落在 **162.0Hz**，报告里 0.11% 的偏差即来自这里。BSF 70.5838Hz、BPFO 107.364Hz、FTF 11.9293Hz 同理都存在约 1Hz 的量化误差。
- **原始幅值谱峰位对齐已扩围，但仍受 1Hz 分辨率限制**：`spectrum_peaks` 现保留「全谱幅值前 5 + 特征带（最高特征频率的 3 阶谐波）内幅值前 20」，最多 25 个局部极大值（修正前为「全谱前 5 + 1kHz 以下前 5」共 10 个）。修正前外圈样本这 10 个峰全部落在 538~3552Hz、1kHz 以下一个峰都没有，`fault_frequency_matches` 实测为空列表；扩围后外圈样本实测 6 条匹配（BPFO 1/2 阶、BPFI 1/2/3 阶、BSF 3 阶）。但特征带内仍是 1Hz 栅格，理论 162.186Hz 只能落在 162.0Hz，谱峰对齐这类证据的量化误差依旧存在。
- **`bsf` 口径已对齐，滚动体条目已在场景 4 登顶（但并列未消除）**：对外同时给出 1×BSF（`characteristic_frequencies["bsf"]` = 70.5838Hz）与 2×BSF（`["bsf_2x"]` = 141.1676Hz），与 `ball_fault` 知识条目判据口径一致；本轮进一步靠知识库判据表达与 `frequency_family` 元数据把 `ball_fault` 顶到检索首位（0.6283）。但 BPFI 与 BSF 包络能量仍只差 7.06%（并列），结论由「首位候选声明的族落在并列族内」这一采信规则给出、置信度只有 low，不代表主导族已确定（见第五节）。
- **并列族采信依赖知识库 `frequency_family` 元数据**：条目未声明该字段时不采信、仍判证据不足；ball 场景过阈值候选只有 1 条、缺少交叉印证；谱峰对齐证据仍偏向内圈（2×BSF 最近峰偏差 4.84% 超出 2% 容差）；内圈 / 外圈场景仍会触发前两名候选分差复核告警；混合检索（hybrid）路径本轮仍未验收。
- **工况单一**：只覆盖 12k 驱动端 / 0 hp / 1797rpm 一种工况，变转速、变负载未验证。
- **样本量极小**：每个状态只有 1 秒（12000 点），实验台单点加速度计数据，与现场噪声和传递路径差异很大。**5 个内置样例不足以支撑任何准确率结论，本文不给出、也不外推准确率。**
- **共振带固定比例选取**：按奈奎斯特频率 0.33~0.83 倍约定，未用谱峭度 / kurtogram 自适应选带。
- 输出属工程辅助建议，不能替代现场复测、拆检与专业人员判断。

外部资料：数据集说明见 `docs/dataset.md` 与 [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)；架构与失败处理见 `docs/architecture.md`；示例报告见 `docs/example_report.md`；滚动体故障判据见仓库内 `app/rag/knowledge/ball_fault.md`。
