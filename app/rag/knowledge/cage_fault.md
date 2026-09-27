---
fault_type: cage_fault
title: 滚动轴承保持架故障
applicable: 单列深沟球轴承（CWRU 6205-2RS JEM SKF 驱动端），保持架磨损、变形或断裂工况
frequency_family: FTF
typical_features:
  - 特征频率为亚同步的 FTF，显著低于转频 fr，出现在频谱低频段
  - 常伴随 FTF 的谐波与转频 fr 边带，谱峰族较为密集
  - 松动与摩擦导致底噪抬高，时域出现低频调制与不规则冲击
  - 保持架故障发展快，幅值可能在较短时间内迅速增大
frequency_signature: FTF ≈ 0.3983 × fr（6205 驱动端），特征峰位于亚同步低频区并伴随谐波
criteria:
  - id: cage-ftf-peaks | claim: 谱峰在 FTF 族（含 2、3 阶谐波）至少 1 阶处对齐 | metric: peak_match_orders | family: FTF | op: ">=" | value: 1
  - id: cage-ftf-rank | claim: 包络谱 FTF 族能量进入四族前二 | metric: envelope_rank | family: FTF | op: "<=" | value: 2
  - id: cage-mild | claim: 峭度相对基线变化不显著（小于 30%），表现为低频调制而非强冲击 | metric: kurtosis_ratio | op: "<" | value: 1.3
sources:
  - name: CWRU Bearing Data Center | url: https://engineering.case.edu/bearingdatacenter/welcome | locator: 驱动端故障频率系数表（6205 驱动端 FTF 系数 0.3983）与保持架/滚动体相关说明
  - name: Rolling element bearing diagnostics—A tutorial | url: https://doi.org/10.1016/j.ymssp.2010.07.017 | locator: 第 2 节 轴承运动学与特征频率定义（FTF 与 BPFO、滚动体数目的关系）
  - name: Bearing Fault Frequency Calculator and Guide | url: https://wertek.ai/engineering/vibration/bearing-frequencies/ | locator: SKF 深沟球轴承 62xx 系列表 6205 行（FTF 0.3983）及 FTF 计算说明
---

## 故障机理
保持架的作用是把滚动体均匀分隔并引导其公转，其公转频率 FTF 由轴承几何决定，典型值为轴频的 0.35～0.45 倍，对 6205 约为 0.3983×fr，故属于亚同步分量。当保持架发生磨损、变形、铆钉松动或断裂时，滚动体间距失控、彼此挤压摩擦，产生低频冲击与随机松动激励，频谱在 FTF 及其谐波附近出现峰值并伴随转频边带。保持架一旦开裂，故障会快速恶化并可能导致轴承卡死，因此识别后通常需要优先处理。

## 典型振动表现
- 时域指标：RMS 与峭度升高，波形出现低频调制与不规则冲击
- 频谱：FTF 及 2×FTF 等亚同步窄带峰，低频段谱峰族密集
- 调制特征：FTF 峰两侧出现转频 fr 边带，反映滚动体与保持架的相互作用
- 噪声底：松动摩擦使宽带底噪抬高，高频区能量整体上行
- 与基线对比：幅值增长速率通常快于滚道类故障，需重点关注趋势突变

## 建议复核
- 以实际转速核算 FTF（按轴承几何系数反推），核对低频峰位置
- 检查 FTF 及其谐波是否成族出现，并与转频边带间隔相互印证
- 因 FTF 频率低，需保证足够的频率分辨率与较长采样时长，避免低频峰被淹没
- 结合温度、噪声与润滑检查，保持架损伤常伴异物、缺油或装配不良
- 一旦确认存在保持架特征，应缩短复核周期并优先安排停机检查