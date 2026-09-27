---
fault_type: inner_race_fault
title: 滚动轴承内圈故障
applicable: 单列深沟球轴承（CWRU 6205-2RS JEM SKF 驱动端），中低转速电机拖动工况
frequency_family: BPFI
typical_features:
  - 时域 RMS 与峭度同步上升，冲击成分明显增强
  - 波峰因子升高，频谱在 BPFI 及其谐波处出现成组峰值
  - 内圈随轴转动，故障频率两侧出现转频 fr 间隔的边带（幅值调制）
  - 包络谱上 BPFI、2×BPFI 附近幅值成组抬升，常伴随 3×BPFI
frequency_signature: BPFI ≈ 5.4152 × fr（6205 驱动端），谐波 2×BPFI、3×BPFI 附近幅值抬升，并伴随转频 fr 边带
criteria:
  - id: inner-rms-up | claim: RMS 相对基线上升 30% 以上，冲击能量明显增强 | metric: rms_ratio | op: ">=" | value: 1.3
  - id: inner-kurt-up | claim: 峭度相对基线上升 30% 以上，冲击成分增强 | metric: kurtosis_ratio | op: ">=" | value: 1.3
  - id: inner-crest-up | claim: 波峰因子相对基线上升 20% 以上，瞬时冲击相对水平升高 | metric: crest_factor_ratio | op: ">=" | value: 1.2
  - id: inner-bpfi-peaks | claim: 谱峰在 BPFI 族（含 2、3 阶谐波）至少 2 阶处对齐 | metric: peak_match_orders | family: BPFI | op: ">=" | value: 2
  - id: inner-bpfi-rank | claim: 包络谱 BPFI 族能量为四个特征频率族中最高 | metric: envelope_rank | family: BPFI | op: "<=" | value: 1
  - id: inner-bpfi-margin | claim: BPFI 主导族量级余量不低于 1.15，主导族可判定 | metric: envelope_margin | op: ">=" | value: 1.15
sources:
  - name: CWRU Bearing Data Center | url: https://engineering.case.edu/bearingdatacenter/welcome | locator: 12k Drive End Bearing Fault Data 说明页中 Inner Race 内圈故障条目（IR007/IR014/IR021）及驱动端故障频率系数表
  - name: Rolling element bearing diagnostics using the CWRU data: A benchmark study | url: https://doi.org/10.1016/j.ymssp.2015.04.021 | locator: 第 3 节 特征频率与诊断方法、第 4 节 内圈故障诊断案例
  - name: Short-Time/-Angle Spectral Analysis for Vibration Monitoring of Bearing Failures under Variable Speed | url: https://doi.org/10.3390/app11083369 | locator: 第 2 节 Methods 表 3 中 BPFI 计算公式 fBPFI = N/2 · fr · (1 + (Bd/Pd)cosΦ)
---

## 故障机理
内圈与轴过盈配合、随轴一同旋转，当滚道表面出现剥落、点蚀或压痕时，滚动体每经过一次缺陷便产生一次冲击激励。由于内圈缺陷在承载区与非承载区间周而复始地进出，冲击强度被周期性调制，故其特征频率不仅出现在 BPFI 上，还会以转频 fr 为间隔形成边带。BPFI 由轴承几何与转速共同决定，对 6205 而言约为轴频的 5.4152 倍，因此转速变化会成比例地改变峰位，判断时须按实际转速换算而非固定频率。

## 典型振动表现
- 时域指标：RMS 上升，峭度与波峰因子同步增大，说明信号中冲击成分增多
- 频谱：BPFI 及其 2 倍、3 倍谐波处出现离散峰，峰高随故障尺寸增大而抬升
- 调制特征：BPFI 峰两侧伴随 fr 间隔边带，或包络谱中 BPFI 族成组出现
- 高频共振带：加速度包络在轴承共振频带上能量增强，宽带底噪随故障发展逐步抬高
- 与基线对比：以上指标相对健康基线呈单调或阶梯式增长，而非随机波动

## 建议复核
- 用实际转速重新计算 BPFI、2×BPFI，确认峰值偏差是否在工程容许范围内（一般延续 <2%）
- 检查边带间隔是否与转频 fr 一致，以确认是否属于内圈调制特征
- 在驱动端径向（水平/垂直）与轴向多点复测，确认峰值不是传感器安装或结构共振造成
- 结合包络谱与谱峭度最优频带，排除谱线泄漏与转速波动引起的误判
- 记录温度、润滑与运行时长，作为严重程度分级的辅助依据