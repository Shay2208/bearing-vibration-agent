# 设计问答

围绕「工业设备故障诊断 Agent 工作台」的设计问答，覆盖架构取舍、数值口径与已知边界。所有数字都来自 `docs/evaluation.md` 的真实实测记录或仓库内实测，**没有推算值、没有未经验证的准确率**；文末单列「未验证项」，如实说明。

## 1. 为什么把 MCP 和 RAG 分成两层，不合成一个「知识+计算」模块？

因为两者的契约完全不同，混在一起就没法分别替换、分别测试。

- **MCP 层**的契约是「数据进、结构化证据出」，纯 `numpy/scipy` 计算，无网络依赖，同一输入必然得到同一输出。它的输出是数值事实：RMS、峭度、频带能量、包络谱特征频率能量、谱峰与特征频率的对齐偏差。这一层不知道任何故障名称。
- **RAG 层**的契约是「结构化证据进、故障候选与来源出」。它不做任何信号计算，只负责把设备信息和对比证据拼成中文查询、打分排序、按阈值过滤，返回候选 `fault_type`、命中特征、来源与 `score`。

分开之后有三点好处：一是**算法可以独立验证**——测试里用 `numpy/scipy` 独立重算全部标量特征与工具输出逐项比对；二是**知识库可以独立演进**——加一条知识条目不需要碰信号算法；三是**职责可审计**——出了问题能立刻分清是「算错了」还是「查错了」。本次那个滚动体误判的修正正好用到了这一点：信号侧修「能量并列不选优」，知识库侧补「包络能量分散、无单一主导族」判据并声明 `frequency_family: BSF`，两层各自演进、合起来把检索首位从内圈条目翻成 `ball_fault`（0.6283，置信 low）。

## 2. 为什么不让大模型直接看波形 / 直接判断故障？

三个理由，按重要性排序：

1. **不可复现**。同一条波形问两次模型可以是两个答案，而工程诊断需要「同一输入必然同一输出」。工具层是确定性的：`vibration.py` 里没有任何随机性，测试里 `test_feature_extraction_matches_independent_recomputation` 就是对这一点的强约束。
2. **模型不能保证数值正确**。故障频率对不对、频带能量涨了多少、谱峰偏了几赫兹，这些必须算出来、不只是「看起来像」。模型很容易生成一个「听起来对」但数值错误的解释，而报告里的数字一旦被改写，整份报告就失效了。
3. **可追溯性**。每条结论要能追到具体数据事实与具体知识条目来源。让模型直接看波形，就丢掉了「依据」这一环。

所以设计里模型被严格限制在「在给定候选集合内选一个，并组织中文表达」。`SYSTEM_PROMPT` 明确写了「严禁引入候选列表之外的故障类型，严禁编造数据、来源或引用」。而且模型就算选了一个候选外的类型，整条模型结果也会被丢弃、回到模板报告（`orchestrator._run_llm` 里的 `allowed` 校验）。

## 3. 将来要换成用户提供的正式 MCP 函数，代价有多大？

设计目标就是**只改一个模块**：`app/mcp/adapter.py`。

适配层对上只暴露 3 个固定契约函数（`validate_vibration_data` / `extract_vibration_features` / `compare_normal_abnormal`），每次调用都返回同一个信封：

```
成功 {"ok": true,  "tool": <工具名>, "backend": <后端>, "data": {...}}
失败 {"ok": false, "tool": <工具名>, "backend": <后端>,
      "error": {"code": "...", "message": "...", "field": null}}
```

编排层只读信封的 `ok` / `data` / `error`，**完全不关心 `backend` 字段**。要接正式 MCP 函数，只需要：改 `_server_parameters()`（启动命令 / 传输方式）或 `_call_mcp()`（协议交互细节），或在 `describe_backend()` 里加一个后端分支。**Agent 编排层、报告层、接口层零改动**，契约与信封结构保持不变。

这个「可替换」不是空话：仓库里已经存在两条后端实现（进程内 `local` 与协议 `mcp:stdio`），它们返回同一种信封、被同一套上层代码消费，实测同一次诊断在两个后端下逐字段一致——这就是这份契约可行性的证据。

## 4. 故障类型是怎么被约束在 RAG 候选范围内的？

三层约束，从内到外：

1. **提示词层**：`SYSTEM_PROMPT` 要求 `selected_fault_type` 只能从给定的候选 `fault_type` 或 `unconfirmed` 中选，并明确列出允许取值：`_build_user_prompt` 会追加一行「允许的 selected_fault_type 取值：<候选列表>、unconfirmed」。
2. **校验层（真正兜底的这层）**：`_run_llm` 里用 `allowed = {候选 fault_type} | {"unconfirmed"}` 做集合校验，`selected` 不在其中就整条丢弃、`llm_mode` 退回 `template_fallback`、并往 `trace.warnings` 里写明「模型输出了候选外的故障类型（…）」。**校验不依赖模型是否听话**。
3. **报告层**：`build_report` 只接受候选集合里的故障类型，没有候选时输出「无法确认」，不会新增知识库里没有的类型。

还有一条更上游的约束：候选本身是**阈值过滤后的结果**（`RAG_SCORE_THRESHOLD` 默认 0.35，`RAG_TOP_K` 默认 5）。如果阈值过滤后候选为空，流程根本不进模型。所以最坏情况下，模型能选的范围就是「5 条候选以内 + unconfirmed」。

## 5. RAG 检索不到候选时怎么处理？

**直接降级为「证据不足」，绝不硬凑结论。** 具体行为：

- 候选为空（或 `rag.insufficient_evidence` 为真）→ Agent 置 `status="insufficient_evidence"`、**跳过模型调用**、报告结论写「无法确认」、`sources` 为空。
- HTTP 状态码**仍然是 200**——这不是 HTTP 层面的错误，而是业务层面的「没有可信证据」，因此它通过响应体 `status` 字段表达，页面上会显示黄色横幅「无法确认 / 证据不足」，并原样展示 `trace.warnings`。
- 验证过两条独立路径：把 `RAG_SCORE_THRESHOLD` 抬到 `0.99`（场景 6）与用与知识库无关的查询词（单测 `test_rag_unrelated_query_returns_no_candidate`），都能得到空候选 + `insufficient_evidence`；`test_rag_empty_result_marks_insufficient_evidence_without_guessing` 专门断言此时不返回任何猜测的故障类型。
- 报告里也不留白：即使无法确认，仍会输出完整的数据事实与检索过程展示，只是明确标注「不作为诊断结论」。

对工业场景来说，「今天证据不够、请补充数据」是一个合法且必要的输出，比给一个 51% 把握的故障名安全得多。

## 6. 包络解调为什么是必要的？不做行不行？

**不做就没有区分度。** 这是这个项目里最硬的一条物理事实：

CWRU 这些样本的特征频率都落在低频段——转频 29.95Hz、FTF 11.929Hz、BSF 70.584Hz、BPFO 107.364Hz、BPFI 162.186Hz。但轴承局部缺陷的冲击能量主要被激励到轴承与结构的高频共振带上（12kHz 采样下约 2~5kHz）。**直接在原始幅值谱上取这些低频点的能量，异常与正常的能量比全部小于 1**——也就是异常样本在这些频点上反而不比正常样本高。

做法是：先在共振带上带通滤波（默认取奈奎斯特频率的 0.33~0.83 倍，4 阶巴特沃斯 + `filtfilt` 零相位滤波），对带通信号取 `|hilbert(x)|` 得到包络并去均值，再对包络做 rFFT 得到包络谱，最后在 FTF/BSF/BPFO/BPFI 的 1、2、3 阶谐波处按 ±3% 窗口累加幅值平方。

换到包络谱之后区分度立刻出来：内圈样本 BPFI 处能量 1.51e-2（基线未检出）、外圈样本 BPFO 处 2.49e-1，都成为各特征频率中最高，结论正确。

也要说清局限：共振带是按奈奎斯特频率**固定比例**选的，没用谱峭度 / kurtogram 自适应选带；带选偏了包络谱信噪比就下降。滚动体样本的误判就与弱冲击信噪比低有关。

## 7. 最大工具调用次数和超时、重试是怎么定的？

**工具调用上限 `MAX_TOOL_CALLS` 默认 8**。依据是正常路径**固定消耗 4 次**（数据校验、正常特征、异常特征、对比），留一倍余量给未来的步骤扩展。实现上是「**调用前检查**」：`call_tool` 在每次调用前判断 `trace["tool_calls"] >= limit`，超了就停止并返回 `error.code=max_tool_calls_exceeded`，且**不产生任何诊断结论**。这个约束是可以被验证的：把上限调成 2，就会在第 3 次调用（异常样本特征提取）前停下。测试里 `test_llm_timeout_falls_back_to_template_report` 等场景的 `tool_calls` 都实测为 4，远低于上限。

**模型超时 `LLM_TIMEOUT` 默认 20 秒、`LLM_MAX_RETRIES` 默认 1**（即最多尝试 2 次），同时 SDK 层设 `max_retries=0`，避免重试次数被 SDK 再叠一层。适配层的单次 MCP 调用超时也是 60 秒（`_MCP_TIMEOUT_SECONDS`，含连接+调用+断开）。

设计原则是「**失败必须降级，不能中断**」：模型超时、调用失败、返回不是合法 JSON、返回候选外的故障类型，这四种情况都统一降级为模板报告，把原因写进 `trace.warnings`，`status` 仍然是 `ok`。场景 7 用注入的假客户端抛 `LLMError("timeout")` 验证过：`llm_mode=template_fallback`、`llm_error={"code":"timeout",...}`、报告结构与长度完整。

## 8. 为什么报告里要刻意区分「数据事实 / 知识依据 / 推断结论」？

因为这三种信息的**可信度和责任归属完全不同**，混着写就无法审计：

- **数据事实**（`report.data_facts`）：来自 MCP 工具的确定性计算，是「算出来的」，可以复现、可以第三方复算。例如「RMS 由 0.0732 变为 0.2889，上升 295%」。
- **知识依据**（`report.knowledge_basis`）：来自知识库条目，是「查到的」，每条都挂来源与 locator 定位。例如内圈条目的判据「包络谱上 BPFI、2×BPFI 附近幅值成组抬升」。
- **推断结论**：把前两者对齐后得出的判断，是「推出来的」，可能错，需要现场复核。

分开写还带来一个实际好处：**评审者能一眼看出结论的强度**。如果数据事实很硬、知识依据只有一条、两者对得也不算紧，那置信等级就该是 low，报告也会这么写（示例报告第 9 节：仅检索到 1 条候选，缺少交叉印证，置信等级 low）。这也是为什么模型只被允许改写「结论」和「推断」部分的措辞，`build_report` 会**确定性**地组装结构与数值，保证换模型不会改变事实部分。

## 9. 双后端（`local` / `mcp:stdio`）的一致性是靠什么保证的？验证过吗？

靠**共用同一套信号实现 + 共用一个返回信封**。

- 数据来源唯一：`mcp:stdio` 后端启动的 `app/mcp/server.py` 只是把 `app/mcp/vibration.py` 的同一批函数注册成 MCP 工具，两边**调用的是同一份算法代码**，不存在「两份实现」漂移的可能。
- 契约唯一：适配层不论走哪条路径都返回同一个信封结构，编排层只读 `ok` / `data` / `error`，对 `backend` 无感。

验证过，而且发现了一处真实差异：`test_stdio_backend_success_matches_local_backend`（标 `@pytest.mark.stdio`，真实起子进程）断言两后端的 `rms`/`peak`/`kurtosis`/`crest_factor`/主频/主频幅值/频带能量/频谱峰/包络特征频率能量**逐字段一致**并全部通过。另一处 `test_bad_sampling_rate_error_envelope_is_identical_across_backends` 也通过，但记录了一处真实差异：`local` 的原始消息是「采样率必须是有限正数，收到：0」，`stdio` 是「…收到：0.0」——因为 MCP 工具签名把 `sampling_rate` 声明为 `float`，整型 `0` 跨协议边界被强转成了 `0.0`。这让测试断言改为「键集合、`code`、`field` 全等 + 数字归一后消息相等」，而不是逐字相等。**这类差异只有在真的跑了两条路径之后才会暴露。**

端到端层面同样一致：同一「正常 vs 内圈」用例，两个后端都是 `inner_race_fault`、score 0.4453、`sources=2`，差异只在耗时（`local` 97~109ms，`mcp:stdio` 实测 8346ms 约 8.3 秒）。

## 10. 没有 API Key 时系统怎么工作？为什么这样设计？

不提供环境配置时没有模型 Key，系统设计上**把两个可选的网络依赖都做成了可降级项**：

- **Embedding 不可用**（`settings.embedding_available=false`）→ `KnowledgeStore` 初始化时直接落到 **BM25** 分支，检索照常工作。`/health` 会返回真实的 `retrieval_mode="bm25"`，不隐瞒。
- **模型不可用**（`settings.llm_available=false`）→ `_run_llm` 一进来就检查 `client.available`，为假则直接写 `trace.llm_mode="template"`、返回 None，**不发起任何网络请求**，报告由 `report.py` 模板化生成。

这样设计有三个好处：一是**演示和测试可离线**（内置样例 + BM25 + 模板报告全在本地完成，断网也能跑完整链路）；二是**关键约束不依赖外部服务**（故障类型只能来自候选这条硬约束，在模板模式下同样成立）；三是**状态透明**（不可用不是静默兜底，`/health`、`trace.llm_mode`、`trace.warnings` 都会写明原因）。模板报告的字段结构与接真实模型时逐字段一致，模型只影响 `conclusion` 与 `inferences` 的措辞。

需要提醒的公平性问题：阈值 `0.35` 是给归一化余弦相似度设计的量级，而 BM25 分量是词权重点积乘 `_BM25_SCALE=2.5` 放大后的分数，两者在混合检索里各自归一到 `[0,1]` 再按 `0.6 / 0.4` 加权融合，但**阈值对两条分量的含义并不严格等价**，不能把 BM25 下的表现直接当成向量检索的表现。混合检索的融合公式、归一化与降级路径见第 17 问。

## 11. 这个原型的局限有哪些？

按重要性列：

1. **数据面**：只覆盖 12k 驱动端 / 0 hp / 1797rpm **一种工况**；每个样本只有 **1 秒（12000 点）**，实验台单点加速度计，与现场噪声和传递路径差异很大；5 个内置样例**不足以支撑任何准确率结论**。
2. **算法面**：共振带按奈奎斯特频率**固定比例**（0.33~0.83）选取，未使用谱峭度 / kurtogram 自适应选带；不做严重程度分级，不做剩余寿命预测。
3. **证据面**：`spectrum_peaks` 早期只保留「全谱前 5 + 1kHz 以下前 5」共最多 10 个峰，谱峰对齐这类证据在 1 秒数据下会退化（外圈样本这 10 个峰全落在 538~3552Hz，1kHz 以下一个峰都没有，`fault_frequency_matches` 实测为空列表）。**已扩围**为「全谱前 5 + 特征带内前 20」共最多 25 个，外圈样本现可得到 6 条匹配；但特征带内仍是 1Hz 栅格，量化误差依旧。
4. **已知缺陷**：滚动体样本的误判已修正——滚动体条目补了「包络能量分散、无单一主导族」判据并声明 `frequency_family: BSF`，检索首位变为 `ball_fault`（0.6283、置信 low），编排层按「并列时首位候选声明的族落在并列族内才采信」给出结论。但**并列没有被消除**（BPFI 与 BSF 只差 7.06%）、结论**不代表主导族已确定**，采信依赖条目 `frequency_family` 元数据、ball 场景只有 1 条过阈值候选，见第 13 问。`bsf` 对外口径已与知识库对齐（同时给出 1×BSF 与 2×BSF）。
5. **链路面**：演示项目不做 MCP 连接池，`stdio` 每次调用都重建连接，单次诊断要 8~12 秒；`MCP_SERVER_CMD` 含空格路径的问题已修复（解析规则见 `docs/architecture.md` 第五节）。
6. **定位面**：输出是工程辅助建议，**不能替代现场复测、拆检与专业人员判断**。

## 12. 如果给你更多时间，你会按什么优先级改？

1. **已完成的那个可解释误判修正**（第 13 问）：给 `dominant_characteristic` 加**量级余量判据**（`_DOMINANT_MARGIN_RATIO = 1.15`）与并列候选；给滚动体条目补「包络能量分散、无单一主导族」判据并声明 `frequency_family: BSF`（内圈 / 外圈 / 保持架分别 `BPFI` / `BPFO` / `FTF`）；检索候选透出 `frequency_family`；编排层 `_tied_family_resolution()` 只在「检索首位候选声明的族落在并列族内」时采信该候选，并在报告里把置信度下调为 low；`build_query` 按特征频率族聚合、并列时两族一并写入；谱峰保留范围从「全谱前 5 + 1kHz 以下前 5」扩到「全谱前 5 + 特征带内前 20」。**这一步已把滚动体场景的检索首位从内圈条目换成 `ball_fault`（0.6283，置信 low）**；下一步要把并列真正消除，需要自适应选带改善弱冲击信噪比，而不是继续调阈值。
2. **把共振带选带从固定比例换成自适应**（谱峭度 / kurtogram），直接改善滚动体这类弱冲击的包络谱信噪比，这是比调阈值更根本的改进。
3. **验证未验证的分支**：配好 Key 真实跑通向量的检索分支与真实模型，重新标定两条检索分支的阈值，并实测真实模型的超时率与措辞质量。
4. **扩数据与工况**：在 1hp/2hp/3hp、1730/1750/1770rpm 上做不重叠切窗，做按文件（而非按窗口）划分的评估，同时把时长从 1 秒拉到多秒以缓解 1Hz 的分辨率量化误差。
5. **工程化**：MCP 侧加连接池或常驻会话、补严重程度分级与趋势对比。`MCP_SERVER_CMD` 的解析（已改成 JSON 数组 / 引号切分，支持含空格路径）与「正常对正常 `sources` 残留」（已随 `evidence_check_node` 清空而修复）两项已完成，不再列入待办。

## 13. 【真实失败案例】正常样本 vs 滚动体样本曾被误判为内圈故障（已修正为 `ball_fault`，置信 low，并列局限保留）

**现象**：用例 `normal_1797` vs `ball_1797`（CWRU fileId 118，0.007 inch 故障直径，公开标签 `ball_fault`），**修正前**系统输出 **`inner_race_fault`，score 0.4614**——与真正的内圈样本得分**完全相同**，说明这条查询对两个不同故障给出了几乎一致的文本特征。**上一轮**加入量级余量判据后，该场景不再输出单一故障类型（`status=insufficient_evidence`、结论「无法确认」，`dominant_characteristic` 给出 `ambiguous=true` 与 BPFI/BSF `tied_candidates`），但检索首位仍是内圈条目。**本轮（判据表达 + 检索排序 + 并列族采信）**检索首位修正为 `ball_fault`（score **0.6283**、`frequency_family=BSF`），结论 `ball_fault`，置信度 **low**。回归测试为 `tests/test_e2e_diagnose.py::test_normal_vs_ball_fault_tie_broken_by_top_candidate`（断言 `status=ok`、top1 `ball_fault`、`frequency_family=BSF`、置信 low 与 `trace.tied_resolution`）。

**确切数字（实测）**：

| 项 | 数值 |
|---|---|
| ①修正前候选 top1 | `inner_race_fault` 0.4614（与内圈样本得分相同） |
| ②上一轮结果 | `insufficient_evidence`，「无法确认」；检索首位仍是 `inner_race_fault` 0.432 |
| ③本轮结果 | `status=ok`，候选 top1 **`ball_fault` 0.6283**（`frequency_family=BSF`），结论 `ball_fault`，置信度 **low** |
| 本轮过阈值候选数 | **1 条**（`ball_fault`），缺少交叉印证 |
| `trace.tied_resolution` | `{"fault_type": "ball_fault", "frequency_family": "BSF", "tied_labels": ["BPFI", "BSF"], "margin_ratio": 1.0706}` |
| 包络谱 BPFI 族能量（1/2/3 阶累加） | **4.20470e-4** |
| 包络谱 BSF 族能量（1/2/3 阶累加） | **3.92749e-4** |
| BPFI / BSF 比值 | **1.0706**，即 BPFI 仅高约 **7.06%**（低于量级余量阈值 15%） |
| `dominant_characteristic` | `label=BPFI`、`ambiguous=true`、`margin_ratio=1.0706`、`tied_candidates=[BPFI@162.186Hz, BSF@70.5838Hz]` |
| 滚动体包络谱最大峰 | **162.0Hz**（幅值 9.638e-3） |
| BSF 理论位置附近 | 理论 70.58Hz，实测最近峰在 **69.0Hz**（幅值 7.542e-3） |
| 2×BSF 对齐（未动的局限） | 理论 141.17Hz，最近峰 **148Hz**（偏差 **4.84%**），超出 2% 容差 |
| `fault_frequency_matches` | 3 条：BPFI 1/2/3 阶（修正前仅 1 条） |
| 时域 | RMS 0.1374（基线 0.0732，约 1.878 倍）、峭度 2.965 |

**三个成因**：

1. **BPFI 族能量在噪声量级上仅高 7%，而选优规则没有量级余量判据。** `dominant_characteristic` 只按「各族取最大值」选优，于是硬选出 BPFI，`reason` 里也写成「包络谱中 BPFI（162.2Hz）及其 2、3 阶谐波频带能量为各特征频率中最高」，并作为证据进入检索查询。**7% 的差不足以支撑排他性判断。**
2. **谱峰对齐证据在这个样本上退化了。** `spectrum_peaks` 只保留「全谱幅值前 5 + 1kHz 以下前 5」共最多 10 个峰，滚动体样本与内圈样本一样只因为存在 162.0Hz 这个峰而匹配上 BPFI 1 阶；外圈样本同理，10 个峰全落在 538~3552Hz，1kHz 以下一个峰都没有，`fault_frequency_matches` 实测为空列表。也就是说「谱峰与特征频率对齐」这类证据在单点测点、1 秒数据下会整体退化。
3. **BSF 族的对外口径与知识库判据不一致，查询没传递关键区分点。** `characteristic_frequencies["bsf"]` 对外只给 **1×BSF = 70.5838Hz**，而 `ball_fault` 知识条目的判据是「**2×BSF = 141.1676Hz 附近出现峰值**」。查询文本里 BSF 族只以「含 2、3 阶谐波的包络能量」这种笼统表述出现，没有「以 2×BSF 为主」这一关键区分点，因此知识条目里最有辨识度的判据没被命中。

**必须纠正的一个常见误解**：说「`bsf` 只计算球自转频率、没计算 2×BSF」在**能量口径上并不成立**。`vibration._ENVELOPE_HARMONICS = (1, 2, 3)`，`characteristic_energy["bsf"]` **已经**把 1×BSF(70.58Hz)、2×BSF(141.17Hz)、3×BSF(211.75Hz) 一起求和，`_match_fault_frequencies` 也按 (1,2,3) 阶逐阶对齐谱峰。**真正的问题是第 3 点的口径没对齐，而不是没算 2×BSF。**

**上一轮已实施的修正（四项，均只改算法与查询构造，未改知识库）**：

- 给 `dominant_characteristic` 加**量级余量判据**（`_DOMINANT_MARGIN_RATIO = 1.15`）：最高与次高能量差低于 15% 时不输出排他性主导项，改给出 `tied_candidates` 并列候选、置 `ambiguous=True`。
- 把 **BSF 族口径对齐知识库**：对外同时给出 1×BSF 与 2×BSF（新增 `characteristic_frequencies["bsf_2x"]`）。
- **`build_query` 按特征频率族聚合**：同一族的 1/2/3 阶合并成一条，并列时把两个族一并写入，不再用「最突出」这类单点措辞。
- **扩大谱峰保留范围**：从「全谱前 5 + 1kHz 以下前 5」改为「全谱前 5 + 特征带（最高特征频率的 3 阶谐波）内前 20」，上限 10 → 25。

**本轮新增的修正（知识库判据表达 + 检索排序 + 并列族采信）**：

- **补知识库判据表达**：滚动体条目新增「包络能量分散：各特征频率族能量相当、无单一主导族，主次难以区分」，并在 front-matter 声明 `frequency_family: BSF`（内圈 / 外圈 / 保持架分别 `BPFI` / `BPFO` / `FTF`）。
- **检索候选透出 `frequency_family`**（`app/rag/store.py`），供编排层判断。
- **编排层新增 `_tied_family_resolution()`**：并列时**只有当检索首位候选声明的 `frequency_family` 落在并列族内**才采信该候选、继续出结论；族不匹配或条目未声明族时仍按证据不足处理（防回归守卫）。
- **报告层在并列采信时强制把置信度下调为 low**，并写明并列依据与一条现场复核建议（`app/agent/report.py`）。

**修正后的实际效果（实测）**：滚动体场景从「误判为内圈」→「无法确认」→ 本轮**检索首位变为 `ball_fault`（0.6283）、结论 `ball_fault`、置信 low**；外圈样本的 `fault_frequency_matches` 从空列表变为 6 条；内圈样本 top1 与结论不变（0.4453），前两名分差仍低于阈值、会触发候选分差复核告警。需要如实说明：**并列没有被消除**（BPFI 与 BSF 只差 7.06%），结论来自「首位候选声明的族落在并列族内」这一采信规则、**不代表主导族已经确定**；采信依赖条目的 `frequency_family` 元数据，ball 场景只有 1 条过阈值候选；弱冲击信噪比与共振带固定比例选带这两个根因仍未动，要进一步把并列消除需要谱峭度 / kurtogram 自适应选带。

**为什么这个案例值得专门讲**：一个能定位到「7% 能量差 + 缺量级余量判据 + BSF 口径不一致 + 知识库判据表达与族元数据缺失」这几层根因、并把结论从「过度解读」一路修正到「正确候选 + 置信 low + 如实保留并列局限」的失败案例，比一个漂亮的分数更能说明工程判断力。而且它是**可复现**的——同一条用例、同一份数据，测试断言的是「top1 = `ball_fault`（0.6283）+ 置信 low + `trace.tied_resolution`」这一确定行为（`test_normal_vs_ball_fault_tie_broken_by_top_candidate`）。

## 14. 「正常对正常」为什么会被判成 `insufficient_evidence`？这是怎么做到的？

靠 `orchestrator._no_significant_change(comparison)` 这道**语义拦截**。

判据是：**没有任何显著变化的特征**（`changed_features` 为空），并且**所有特征变化倍数与 1 的偏离都小于 0.3**（即变化不到 ±30%），并且**主频未迁移**（`frequency_change.shift_hz` 绝对值小于 1e-9）。三者同时成立才算「异常样本相对基线完全没有变化」。

为什么需要它：检索是用查询文本打分的，而「正常 vs 正常」这个对比虽然没有任何异常可解释，却仍然可能检索到一条高分候选。如果不拦，就会出现「输入两个正常样本、系统却报了一个故障类型」这种荒谬结果。拦截后 Agent 会把 `candidates` 清空、置 `status=insufficient_evidence`、跳过模型、报告结论写「无法确认」，并往 `trace.warnings` 写「异常样本相对基线未观察到任何显著变化……不存在可解释的异常，按证据不足处理，不做故障类型判断」。

实测：场景 1 正常 vs 正常 → `status=insufficient_evidence`、`tool_calls=4`、`elapsed_ms=98`、`retrieval=bm25`，**结论正确（没有基于候选下结论）**。

**此前存在的一个字段残留已修复**：早期实现里 `out["sources"]` 在检索完成后就赋值，而 `_no_significant_change` 在其后才清空 `candidates`，导致正常对正常时 `sources` 会残留 1 条来源（`inner_race_fault` 的来源）。现在拦截逻辑统一放在 `evidence_check_node`，拦截时把候选与来源一并清空，实测 `candidates==[]`、`sources==[]`、`report["conclusion_sources"]==[]`、报告正文含「无法确认」。回归测试为 `tests/test_api.py::test_normal_vs_normal_returns_no_sources`，已通过。

## 15. 有哪些东西你其实没有验证过？

必须如实说，这些是明确的负面结论，不代表已通过：

1. **真实混合检索质量尚未通过验收**。公开演示保持 `ENABLE_HYBRID_RETRIEVAL=false`；试运行混合检索时外圈样例曾把 `normal` 排在首位。假 embedding 测试只验证索引、打分和降级机制，不能证明语义召回质量。本文 7 个端到端场景及对应分数均来自 BM25。
2. **真实模型成功路径已跑通一次，但长期稳定性未评估**。`scripts/smoke_llm.py` 在 2026-09-25 的本地配置下实测 `llm_mode="llm"`、`llm_calls=1`、`status=ok`，模型选择的故障类型属于候选集合，单次模型调用约 9.4 秒。场景 7 的 `timeout` 仍使用注入的假客户端 `TimeoutLLMClient` 验证降级逻辑，**不等同于真实网络超时统计**；长期延迟、失败率、成本与措辞稳定性尚未评估。
3. **Docker 未构建**。仓库提供 `Dockerfile` 与 `.dockerignore`，但本机未安装 Docker（`docker --version` 报命令不存在），**镜像既未真实构建、也未启动容器验证 `/health`**，只做了静态检查。
4. **只覆盖单一工况**（12k 驱动端 / 0 hp / 1797rpm），变转速、变负载、多测点全部未验证。
5. **样本仅 1 秒**（12000 点），频率分辨率固定为 1Hz（BPFI 理论 162.186Hz 只能落到 162.0Hz），更长时长下的表现未验证。
6. **页面 DOM 的真实渲染效果**已在窄窗（内置浏览器 387px）与桌面宽度（无头 Chrome 1424px）完成端到端检查（含三栏布局：左栏卡片目录与会话窗口、中栏分析输入与结果视图、右栏与模型对话；SSE 执行时间线、卡片切换、受控追问与会话内自由问答；刷新页面后会话列表、时间线与对话记录的恢复）：默认样例可直接运行、结论摘要可见、两次诊断生成 2 个会话窗口且切换时按会话重放、对话回答带「模型生成（已通过会话事实校验）」标记、无横向溢出、控制台无 JS 异常。桌面宽度为无头浏览器 + DevTools 协议验收（内置浏览器窗口宽度固定 387px 无法拉宽），**不是人工交互式操作**；部署环境仍应按实际浏览器再做一次验收。
7. **会话持久化未做多副本与压力验证**：会话与对话消息落盘在单机 SQLite（`data/sessions.db`），刷新页面与服务重启都不丢；但多副本部署不共享该文件，也未做并发写入压测与超大量会话下的列表查询性能验证。多轮上下文只带最近 `CHAT_HISTORY_TURNS`（默认 3）轮、单条截断 400 字符。
8. **曲线与音频是降采样可视化，不参与判据**：时域波形是 600 桶 min/max 包络、幅度谱 600 点、包络谱裁到 0~1000Hz 共 400 点，抽点规则保证特征频率峰被保留，但**不是完整数据**；两路 WAV 是按全局峰值归一化的去均值信号重放，只用于辅助听冲击与调制形态，不参与任何特征计算与打分。音频旁挂文件落在单机本地盘 `AUDIO_DIR`（默认 `data/audio`），多副本不共享，按 `AUDIO_MAX_FILES`（默认 200）淘汰最旧的（被淘汰后旧会话播放会 404，页面降级为「无可播放音频」，诊断结果不受影响）；未做并发写盘压测，也未测「曲线/音频是否提升人工判读效率」。

## 16. 为什么用 LangGraph 画固定图，而不是让模型自由决定调哪个工具？

因为**诊断流程本身是确定的、需要被审计的**，把「调哪些工具、什么顺序、什么时候停」交给模型，会牺牲可复现性和可测试性。

- **流程固定**：校验 → 4 次 MCP 工具调用（校验/正常特征/异常特征/对比）→ RAG 检索 → 证据判断 →（条件分支）报告或模型。这个顺序和工业诊断的固定动作一一对应，用图表达比用提示词约束更可靠。
- **图只有 6 个节点、3 条条件路由**：`evidence_check_node` 之后按「证据是否充分」二选一（不足直接出模板报告，充分先经 `llm_node`）；`validate_node` 与 `mcp_analysis_node` 的失败是错误条件边，直接结束、`report=None`。**没有开放式循环**，模型不可能反复决定「要不要再调一次工具」。
- **工具调用被收口**：MCP 工具只能由 `mcp_analysis_node` 经 `app/mcp/adapter.py` 调用，`MAX_TOOL_CALLS`（默认 8）在每次调用前检查，一次正常诊断实测恰好 4 次。
- **模型职责不变**：模型只被要求在候选集合内选一个 `fault_type` 并组织中文表达（结构化 JSON），选到候选外就整条丢弃、回到模板。**模型永远不是故障类型的来源。**
- **契约与可观测性**：唯一外部入口仍是 `orchestrator.diagnose(request, *, max_tool_calls=None, llm_client=None)`，签名与原有 13 个响应键逐字未变（另新增 `session_id`）；本轮给 `features` 增加了 opt-in 的 `series` 子块（`extract_vibration_features` 的 `include_series`，为 `false` 时结构与旧版逐字一致）；`trace` 新增 `nodes` 键，实测正常路径为 `["validate_node", "mcp_analysis_node", "rag_retrieval_node", "evidence_check_node", "llm_node", "report_node"]`，每一步都可在响应里回溯。SSE 事件流（`/api/diagnose/stream`）、受控追问（`/api/sessions/{id}/ask`）与会话内自由问答（`/api/sessions/{id}/chat`）都是纯增量接口，同步接口零改动兼容；自由问答的回答先过「候选白名单 + 会话来源」两道校验，不过就整段作废、降级为确定性回答。

## 17. 混合检索为什么用 0.6 / 0.4？分数怎么归一化？怎么降级？

**融合公式**：`final_score = RAG_VECTOR_WEIGHT * vector_score + RAG_BM25_WEIGHT * bm25_score`，默认 `0.6 * 向量分 + 0.4 * BM25 分`。

- **为什么 0.6 / 0.4**：向量分负责语义相近，BM25 分负责术语精确匹配——本项目查询文本里大量出现 `BPFI`、`2×BSF`、`包络谱` 这类专有词，这是 BM25 的强项。因此向量权重略高，但两者都不为 0，避免任一分支失效时结果整体崩掉。两个权重都可用环境变量覆盖。**这个权重是工程取值，没有做离线召回评测来调优**，如实说明。
- **归一化**：两个分量在融合前各自归一到 `[0, 1]`——向量分由 Chroma 的余弦距离换算成 `1 - distance`；BM25 分先按固定系数 `_BM25_SCALE = 2.5` 放大再裁剪到 `[0, 1]`；融合结果同样裁剪到 `[0, 1]`。同一条知识条目命中多个文本块时只保留最高分块，之后再用 `RAG_SCORE_THRESHOLD`（默认 `0.35`）过滤。
- **降级路径**：`retrieval_mode` 只可能取 `hybrid` 或 `bm25`。① 未开启混合检索或未配置 embedding → 初始化即落到纯 BM25，**不发任何网络请求**；② 配置了 embedding 但索引不可用（目录/集合不存在）、或向量/embedding 调用抛异常 → 同样走 BM25，但 `store.degraded=True` 且给出 `degraded_reason`，该原因被追加进 `trace["warnings"]`，**不中断诊断**。这两条降级路径都有测试覆盖（`tests/test_rag_vector.py`、`tests/test_api.py::test_diagnose_still_works_when_chroma_unavailable`）。

## 18. Chroma 持久化和「配置 embedding 后必须重建索引」是怎么回事？

- **持久化位置**：索引默认写在 `data/chroma/`（`CHROMA_DIR` 可覆盖），集合名默认 `bearing_knowledge`（`CHROMA_COLLECTION` 可覆盖），该目录**已加入 `.gitignore`**，不进仓库。写入走 `PersistentClient` + `get_or_create_collection`（余弦空间、关闭匿名遥测）。
- **切分**：知识条目正文按「单块 ≤ `RAG_CHUNK_SIZE`（500）字符、相邻块重叠约 `RAG_CHUNK_OVERLAP`（80）字符」切分，块 ID 稳定可复现（形如 `inner_race_fault#chunk-000`），每个块带 `fault_type` / `title` / `source` / `source_url` / `locator` / `document` / `chunk_id` 元数据。实测 7 个条目切成 **14 个块**，块正文最大 **499** 字符，相邻块重叠实测 **79** 字符。
- **重建命令**：`& ".\.venv\Scripts\python.exe" -m app.rag.ingest --rebuild`（幂等：稳定块 ID + `upsert`，并清理已不存在的旧块），查看现状用 `--status`；两者都支持 `--knowledge-dir` / `--chroma-dir` / `--collection`。
- **为什么切换 embedding 后需要重建**：索引向量必须与查询使用相同的模型与维度，知识条目更新也需要同步索引。重建后应做质量评测，通过后才开启 `ENABLE_HYBRID_RETRIEVAL`。公开演示当前不依赖向量索引，不能把目录存在说成召回质量已验证。

## 19. 把这个原型 Docker 化，要注意什么？

仓库已经提供 `Dockerfile` 与 `.dockerignore`，思路是**只装运行依赖、只带运行需要的文件、密钥不落盘**：

- **基础镜像与依赖**：基于 `python:3.12-slim`，只装 `requirements.txt`；不复制 `.venv/`、`docs/`、`tests/`（`.dockerignore` 已排除），但**不排除** `data/samples/` 与 `requirements.txt`，否则内置样例和依赖都会丢。
- **向量索引**：声明 `VOLUME ["/app/data/chroma"]`。因为索引目录已加入 `.gitignore`、镜像里没有索引，不挂载时 Docker 会创建一个空卷，代码侧对索引缺失有降级处理（自动走 BM25）；要复用宿主机已建好的索引，就挂载 `-v "${PWD}/data/chroma:/app/data/chroma"`。
- **端口与启动**：`EXPOSE 8000`，`CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]`。构建与运行：`docker build -t bearing-diagnosis .`、`docker run --env-file .env -p 8000:8000 bearing-diagnosis`。
- **密钥**：`MODEL_API_KEY` 等只通过 `--env-file` / `-e` 注入，绝不写进镜像。
- **不传任何环境变量也能启动**：检索降级 BM25、报告走模板，可离线演示。
- **必须如实说明**：本机**没有安装 Docker**（`docker --version` 报命令不存在），因此**镜像既未真实构建、也未启动容器验证 `/health`**，`Dockerfile` 只做了静态检查，上面的命令未在本机跑过。

## 20. 会话为什么要落盘？多轮对话是怎么实现的？

**原来是内存存储**：会话保存在 `SessionStore` 的进程内 LRU（上限 50 个），服务重启即丢；刷新页面会话列表清空；`/chat` 不保存问答记录，多轮对话没有上下文。改造后三项一起解决：

- **落盘**：换成单文件 SQLite（`SESSIONS_DB_PATH`，默认 `data/sessions.db`），两张表——`sessions`（`session_id`、`created_at`、请求摘要、结果、事件轨迹，后三者按 JSON 文本存）与 `messages`（自增 id、`session_id`、`created_at`、`role`、`content`、`meta`，建 `(session_id, id)` 索引）。诊断结束 `INSERT OR REPLACE` 落盘，追问 / 自由问答各 `append_message`；仍以 50 个会话为上限，超出按创建时间淘汰最旧会话及其消息。
- **恢复**：`GET /api/sessions` 返回倒序摘要（含 `message_count`），`GET /api/sessions/{session_id}` 返回完整记录（结果 + 事件轨迹 + 全部消息）；前端用事件轨迹重放时间线、按 `meta.kind` 把历史问答还原到右栏对话，因此刷新页面甚至重启服务后都能继续在同一会话里追问。
- **多轮**：自由问答把最近 `CHAT_HISTORY_TURNS`（默认 3）轮对话（单条超 400 字符截断）拼进提示词，用来理解「它」「这个故障」这类指代；提示词同时声明**历史对话不是事实来源**，依据仍只有本次会话摘要。上下文因此是有界窗口，不随轮数无限增长。
- **并发**：自由问答跑在 `asyncio.to_thread` 里，所以连接用 `check_same_thread=False` 打开，全部读写由 `threading.Lock` 串行化，单进程内安全。
- **局限（如实说明）**：SQLite 是单机文件，多副本部署不共享；未做并发写入压测与超大量会话下的列表性能验证。改造新增 9 条用例（`tests/test_session_persistence.py`），覆盖重开仍在、列表 / 详情接口、404、消息落盘、轮数窗口与容量淘汰。

## 21. 为什么曲线和音频不新增一个 MCP 工具，而降采样和抽点又为什么这么设计？

**不能新增工具。** 一次正常诊断固定 4 次 MCP 调用（校验 → 正常特征 → 异常特征 → 对比），`trace.tool_calls` 有断言守着（损坏样本必须等于 1），再加第 5 个工具会直接推翻这条「固定 4 次」的口径，也会让工具调用预算的意义变模糊。所以曲线和音频挂到已有的第 2、3 次 `extract_vibration_features` 上：加一个**可选**入参 `include_series`（默认 `false`，不传时返回结构与旧版逐字一致），编排层固定传 `true`。

**抽点必须保峰。** 直接等间隔抽点会把 BPFO/BPFI/BSF 这些很窄的谱峰抽没——它们是这张图唯一的诊断信息。所以幅度谱和包络谱都按「每桶取幅值最大的 bin」抽点；波形则用分桶 min/max（600 桶），因为等间隔抽点会丢冲击尖峰。包络谱另外先裁到 0~1000Hz 再抽 400 点，理由是 BPFI 3 阶也只有约 486Hz，画到 6000Hz 等于把有效信息压掉九成。实测抽点后 `max(spectrum.amplitude)` 与全谱最大幅值一致，argmax 频点也在抽点频率里；加序列前后标量特征逐字段完全相等。

**PCM 不进响应、不进库。** WAV 用 `wave` 模块写成单声道 16bit PCM、按全局峰值归一化到 0.98 满量程，以 base64 随工具结果回到编排层，编排层**先 `pop` 出来落盘**成 `<session_id>_<slot>.wav`，再按 `AUDIO_MAX_FILES` 淘汰最旧的。所以响应体和 SQLite 里只剩 `audio` 的元数据与 `available` 标记，有用例断言整份响应 JSON 搜不到 `wav_base64`（一次完整诊断响应约 89KB，其中含序列的 `features` 约 64KB，单路 WAV 24KB 走的是懒加载端点）。

**如实说明**：曲线和音频都是给前端画图/试听用的降采样可视化，不是完整数据，也不参与任何特征计算与打分；音频按峰值归一化、是去均值信号的重放，只能听冲击与调制形态。

## 22. 知识库为什么要写成具名判据条款？条款核对为什么不让模型做？

**因为「像不像」和「对不对」是两件事。** BM25 / 向量检索只能回答「哪些条目和查询文本相似」，回答不了「这条知识里写的条件，和本次数据到底对得上吗」。外圈样本就是例子：`ball_fault` 条目检索分 0.3974 并不低，但它写的「峭度温和上升」「包络能量分散」这类判据和外圈数据是矛盾的——只按相似度排序，这种候选会一直挂在前面。

- **条款化**：每个知识条目在 front-matter 里带 `criteria` 列表，一条写成 `id` / `claim` / `metric` / `op` / `value`（可选 `family` / `order`）。`claim` 给人读，`metric` / `op` / `value` 给机器核对。当前 7 个条目共 **24 条**（内圈 6 / 外圈 6 / 滚动体 5 / 保持架 3 / 正常 2 / 不对中 1 / 不平衡 1）。
- **不让模型核对**：条款只允许用**受控词表**（15 个指标 × 6 个比较符），不做字符串求值、不做 `eval`——没有表达式注入风险，同一条款两次核对的结论必然一致。把这一步交给模型，等于丢掉「可复现」这个最关键的属性。
- **三态而不是两态**：`hit`（成立）/ `miss`（与数据矛盾）/ `unknown`（指标缺失）。`unknown` **不记为矛盾**——没有采到这项数据不等于这条判据不成立；这一点如果做成两态，就会系统性地冤枉候选项。
- **判据因子与重排**：`factor = 0.4 + 0.6 × 命中 / 已核对`，候选按 `alignment_score = 检索分 × factor` 重排；条目没有条款或没有可核对指标时不罚分（factor = 1.0）。**阈值过滤仍按检索分**，所以候选集合和改造前逐字一致，变的只有顺序。实测：内圈 top1 6/6 命中（0.4453，排序不被改写）；外圈 `ball_fault` 1 命中 / 3 未命中 → factor 0.55，0.3974 压到 **0.2186 沉到末位**。
- **冲突交人工**：首位候选如果 `miss ≥ hit`，编排层不采信它，写 `trace.criteria_conflict`、发一条 `warning`、置信度强制 low、追加人工复核建议——**不硬停机、也不让模型改判**，这就是「规则能判的规则判、规则冲突交受限模型裁量、仍不决交人工确认」这条主线。
- **局限（如实说明）**：24 条条款、词表固定，覆盖不了所有工况；核对只证明「条目写的可核对条件与本次数据一致」，不是物理层面的故障确认。

## 附：被问到数据时可以直接引用的一组实测值

- 测试：**138 passed**（`& ".\.venv\Scripts\python.exe" -m pytest -q`，最近一次实测 138 项全部通过、20.86 秒；耗时随负载波动），用例分布为 `test_unit_vibration` 15 / `test_unit_adapter_rag_report` 9 / `test_e2e_diagnose` 7 / `test_api` 14 / `test_mcp_cmd` 11 / `test_rag_vector` 7 / `test_graph_flow` 22（含并列族采信守卫）/ `test_sse_ask` 22（SSE 事件流、受控追问与会话内自由问答）/ `test_session_persistence` 9（会话落盘、列表与详情、多轮上下文、容量淘汰）/ `test_signal_series` 9（曲线序列与波形音频）/ `test_rag_criteria` 13（判据条款词表守卫、三态与因子、alignment_score 重排、条款证据进 trace/报告）；规模演进 38 → 49 → 77 → 79 → 80 → 92 → 102 → 111 → 116 → 125 → 138。
- 曲线与音频：`include_series` 默认 `false` 时 `features` 无 `series` 键（旧契约逐字不变），编排层传 `true` 后实测 `waveform` 600 桶、`spectrum` 600 点（最大幅值与全谱最大幅值一致，窄峰未被抽掉）、`envelope` 400 点且最大频率 1000.0Hz；加序列前后 `rms`/`peak`/`kurtosis`/`crest_factor`/`dominant_frequency`/`band_energy`/`spectrum_peaks` 逐字段完全相等。WAV 单路 24044 字节（`RIFF`/`WAVE`、1 声道、16bit、12000Hz、12000 帧、峰值归一化不削波）；含序列的 `features` JSON 约 64KB，一次完整诊断响应约 89KB，**响应体与数据库都不含 `wav_base64`**；音频端点正常 200 `audio/wav`，非法 slot / 未知会话 / `../` 均 404 `audio_not_found`；损坏样本无序列无音频、`tool_calls` 仍为 1。浏览器实测「信号分析」卡片新增 3 张图共 8 条 `polyline`、2 个 `<audio>`，控制台 0 报错、无 4xx/5xx。
- 特征频率（1797rpm，CWRU 6205）：转频 29.95Hz、FTF 11.9293Hz、BSF 70.5838Hz、BPFO 107.364Hz、BPFI 162.186Hz。
- 7 个端到端场景的 status 与候选：内圈 → `inner_race_fault` 0.4453（条款核对 6/6 命中；另有 `outer_race_fault` 0.3986 并触发候选分差复核告警）；外圈 → `outer_race_fault` 0.4412（6/6 命中；同场景 `ball_fault` 0.3974 判据 1 命中 / 3 未命中，alignment 0.2186 沉末位）；滚动体 → `ok`，`ball_fault` 0.6283（`frequency_family=BSF`，条款 5/5 命中），置信度 low（BPFI/BSF 包络能量并列，按并列族采信；修正前是误判 `inner_race_fault` 0.4614）；正常对正常 → `insufficient_evidence`（候选与来源均为空）；损坏样本 → `error`（`missing_value`，只有 1 次工具调用）。
- 后端性能：同一「正常 vs 内圈」，`local` 约 97~109ms，`mcp:stdio` 实测 8346ms（约 8.3 秒，历史另有 10773 / 11202 / 11223ms）。
