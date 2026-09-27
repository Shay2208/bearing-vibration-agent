# 轴承振动诊断报告

> 本文件由 `orchestrator.diagnose()` 实际运行生成，未手工修改。用例为内置样例 `data/samples/normal_1797.csv`（基线）与 `data/samples/inner_race_1797.csv`（异常），请求参数 `sampling_rate=12000`、`rotation_speed=1797`、`sensor_position="drive_end"`、`device_type="bearing"`；示例使用 `MCP_MODE=local`、BM25 检索与模板报告，报告含第 5 节的判据条款核对表。生成时间 2026-09-26 18:59。

## 1. 分析对象与输入信息

bearing · drive_end · 转速 1797.0 rpm · 采样率 12000.0 Hz · 异常样本 inner_race_1797.csv（基线 normal_1797.csv）

| 项目 | 取值 |
|---|---|
| 设备类型 | bearing |
| 采样率 (Hz) | 12000.0 |
| 转速 (rpm) | 1797.0 |
| 测点 | drive_end |
| 健康基线文件 | normal_1797.csv |
| 异常样本文件 | inner_race_1797.csv |
| 工具后端 | local |
| 工具调用次数 | 4/8 |
| 知识检索方式 | bm25 |

## 2. 数据质量检查

校验通过、两样本可比（正常样本 12000 点，异常样本 12000 点）。

| 检查项 | 结果 | 说明 |
|---|---|---|
| format | 通过 | 两个文件表头均为 timestamp,amplitude 或 amplitude（测点：drive_end） |
| columns | 通过 | amplitude 列存在 |
| values | 通过 | 无缺失值/非法值 |
| length | 通过 | normal=12000 点，abnormal=12000 点，最少要求 1024 点 |
| sampling_rate | 通过 | 采样率 12000Hz 有效且与时间戳推断一致 |
| comparability | 通过 | 两样本长度一致 |

## 3. 正常与异常特征对比表

| 特征 | 正常 | 异常 | 变化量 | 相对变化 | 说明 |
|---|---|---|---|---|---|
| RMS | 0.0732 | 0.2889 | 0.2157 | +295% | 上升 295%（约 3.95 倍） |
| 峰值 | 0.2845 | 1.5694 | 1.2849 | +452% | 上升 452%（约 5.52 倍） |
| 峭度 | 2.8700 | 5.6337 | 2.7637 | +96% | 上升 96%（约 1.96 倍） |
| 波峰因子 | 3.8884 | 5.4323 | 1.5439 | +40% | 上升 40%（约 1.40 倍） |
| 主频(Hz) | 1036.0000 | 3587.0000 | 2551.0000 | - | 主频上移 2551.0Hz |
| 主频幅值 | 0.0307 | 0.0547 | 0.0240 | +78% | 上升 78%（约 1.78 倍） |

频带能量（单位：幅值平方和）：

| 频带 | 正常 | 异常 | 倍数(异常/正常) |
|---|---|---|---|
| 0-1200Hz | 0.003800 | 0.005800 | 1.52 倍 |
| 1200-2400Hz | 0.000200 | 0.008300 | 34.0 倍 |
| 2400-3600Hz | 0.000000 | 0.039500 | 6584 倍 |
| 3600-4800Hz | 0.000000 | 0.008900 | 4434 倍 |
| 4800-6000Hz | 0.000000 | 0.000200 | 94.0 倍 |

## 4. 频谱或频带变化说明

主频由 1036.0Hz 迁移至 3587.0Hz（偏移 +2551.0Hz）；频带能量变化最明显的是 2400-3600Hz 由 0.000000 变为 0.039500（约 6584 倍），3600-4800Hz 由 0.000000 变为 0.008900（约 4434 倍）；按转速 1797.0rpm 换算的理论特征频率：转频 29.95Hz、BPFO 107.36Hz、BPFI 162.19Hz、BSF 70.58Hz、FTF 11.93Hz；谱峰与特征频率对齐情况：BPFO×3 107.36Hz 附近存在谱峰 323.00Hz（偏差 0.28%，幅值 0.0019）；BPFI 162.19Hz 附近存在谱峰 162.00Hz（偏差 0.11%，幅值 0.0078）；BPFI×2 162.19Hz 附近存在谱峰 323.00Hz（偏差 0.42%，幅值 0.0019）；BPFI×3 162.19Hz 附近存在谱峰 479.00Hz（偏差 1.55%，幅值 0.0052）；包络谱中 BPFI（162.2Hz）及其 2、3 阶谐波频带能量最高（1.51e-02，基线未检出）；各特征频率包络能量：BPFI 1.51e-02、BPFO 1.92e-03、BSF 1.89e-04；发生显著变化的特征：RMS、峰值、峭度、波峰因子、主频幅值。

## 5. RAG 检索到的故障候选

| 故障类型 | 标题 | 相似度 | alignment_score | 来源 | 定位 | 命中特征 |
|---|---|---|---|---|---|---|
| inner_race_fault | 滚动轴承内圈故障 | 0.4453 | 0.4453 | CWRU Bearing Data Center | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 | [inner-bpfi-peaks] 谱峰在 BPFI 族（含 2、3 阶谐波）至少 2 阶处对齐；[inner-bpfi-rank] 包络谱 BPFI 族能量为四个特征频率族中最高 |
| outer_race_fault | 滚动轴承外圈故障 | 0.3986 | 0.3189 | CWRU Bearing Data Center | 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表 | [outer-bpfo-peaks] 谱峰在 BPFO 族（含 2、3 阶谐波）至少 2 阶处对齐；[outer-bpfo-rank] 包络谱 BPFO 族能量为四个特征频率族中最高 |

判据条款核对（hit=成立、miss=与数据矛盾、unknown=指标缺失未核对）：

| 故障类型 | 条款 ID | 判据 | 核对结果 | 实测值 | 期望 | 定位 |
|---|---|---|---|---|---|---|
| inner_race_fault | inner-rms-up | RMS 相对基线上升 30% 以上，冲击能量明显增强 | hit | 3.9487 | rms_ratio >= 1.3 | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 |
| inner_race_fault | inner-kurt-up | 峭度相对基线上升 30% 以上，冲击成分增强 | hit | 1.9629 | kurtosis_ratio >= 1.3 | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 |
| inner_race_fault | inner-crest-up | 波峰因子相对基线上升 20% 以上，瞬时冲击相对水平升高 | hit | 1.3971 | crest_factor_ratio >= 1.2 | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 |
| inner_race_fault | inner-bpfi-peaks | 谱峰在 BPFI 族（含 2、3 阶谐波）至少 2 阶处对齐 | hit | 3.0000 | peak_match_orders(BPFI) >= 2 | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 |
| inner_race_fault | inner-bpfi-rank | 包络谱 BPFI 族能量为四个特征频率族中最高 | hit | 1.0000 | envelope_rank(BPFI) <= 1 | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 |
| inner_race_fault | inner-bpfi-margin | BPFI 主导族量级余量不低于 1.15，主导族可判定 | hit | 7.8903 | envelope_margin >= 1.15 | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 |
| outer_race_fault | outer-rms-up | RMS 相对基线上升 30% 以上，振动总量抬升 | hit | 3.9487 | rms_ratio >= 1.3 | 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表 |
| outer_race_fault | outer-kurt-up | 峭度相对基线上升 30% 以上，冲击序列稳定 | hit | 1.9629 | kurtosis_ratio >= 1.3 | 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表 |
| outer_race_fault | outer-crest-up | 波峰因子相对基线上升 20% 以上 | hit | 1.3971 | crest_factor_ratio >= 1.2 | 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表 |
| outer_race_fault | outer-bpfo-peaks | 谱峰在 BPFO 族（含 2、3 阶谐波）至少 2 阶处对齐 | miss | 1.0000 | peak_match_orders(BPFO) >= 2 | 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表 |
| outer_race_fault | outer-bpfo-rank | 包络谱 BPFO 族能量为四个特征频率族中最高 | miss | 2.0000 | envelope_rank(BPFO) <= 1 | 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表 |
| outer_race_fault | outer-bpfo-margin | BPFO 主导族量级余量不低于 1.15，主导族可判定 | hit | 7.8903 | envelope_margin >= 1.15 | 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表 |

## 6. 最终结论或无法确认说明

最可能故障类型：滚动轴承内圈故障（inner_race_fault）。判断依据：知识库候选给出的特征为「[inner-bpfi-peaks] 谱峰在 BPFI 族（含 2、3 阶谐波）至少 2 阶处对齐；[inner-bpfi-rank] 包络谱 BPFI 族能量为四个特征频率族中最高」，与数据事实（RMS 由 0.0732 变为 0.2889，上升 295%；峭度由 2.87 变为 5.63，冲击成分增强）方向一致。该结论来自知识库候选检索，仍需按复核项在现场确认。

## 7. 结论对应的振动证据

- RMS 由 0.0732 变为 0.2889，上升 295%
- 峭度由 2.87 变为 5.63，冲击成分增强
- 异常样本包络谱中 BPFI（162.2Hz）及其 2、3 阶谐波频带能量为各特征频率中最高（0.0151），且基线中未检出该成分
- 包络谱中 BPFI（162.2Hz）第 1 阶谐波处出现明显峰（162.0Hz，偏差 0.11%）

## 8. 结论对应的知识来源

| 来源 | URL | 定位 |
|---|---|---|
| CWRU Bearing Data Center | https://engineering.case.edu/bearingdatacenter/welcome | 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表 |

## 9. 置信说明

置信等级：low

top 候选 inner_race_fault 的 score=0.4453 低于 0.5，检索匹配偏弱，仅作参考。 判据条款核对（条款全部命中）：共 6 条，命中 6 条、未命中 0 条、未核对 0 条，判据因子 1.0。

## 10. 建议复核的数据或现场检查项目

- 数据层面：当前异常样本仅 12000 点（约 1 秒），建议延长采集时长并在同工况下多次采集，确认特征是否可重复。
- 数据层面：核对采样率设置与时间戳是否一致，并确认测点位置、传感器安装方向在两次采集中保持一致。
- 现场层面：在驱动端径向（水平/垂直）与轴向多点复测，排除传感器安装松动、结构共振造成的伪峰。
- 现场层面：按实际转速核算 BPFI 及其 2×BPFI、3×BPFI，并检查谱峰两侧是否存在转频 fr 间隔边带，确认谱峰偏差是否在工程容许范围内。
- 现场层面：结合包络解调或谱峭度选择最优解调频带后重建信号，确认冲击成分是否具有稳定周期性。
- 现场层面：检查润滑状态、异物污染、轴承温度与运行时长，作为严重程度判断的辅助依据。

## 11. 项目原型的适用边界

- 仅适配单列深沟球轴承（CWRU 6205-2RS JEM SKF 驱动端）与单点加速度信号，其他轴承型号、测点或传感器类型未验证。
- 只覆盖 0 hp / 1797 rpm 一种转速负载工况，变转速、变负载下特征频率换算与判定阈值均未验证。
- 每个样本仅 1 秒（12000 点）片段，样本量不足以支撑统计意义上可靠的诊断或模型训练。
- 知识库条目有限，RAG 候选排序能力受限于知识覆盖范围；检索为空时本原型只输出「无法确认」。
- 包络解调采用按奈奎斯特频率固定比例（0.33~0.83 倍）选定的共振带，未使用谱峭度/kurtogram 自适应选带；不做故障严重程度分级与剩余寿命预测。
- 输出为工程辅助建议，不能替代现场复测、拆检与专业人员判断。
- 样本真实故障标签仅用于离线评测，不进入在线诊断链路。
