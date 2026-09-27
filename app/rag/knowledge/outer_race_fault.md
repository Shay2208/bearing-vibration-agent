---
fault_type: outer_race_fault
title: 滚动轴承外圈故障
applicable: 单列深沟球轴承（CWRU 6205-2RS JEM SKF 驱动端），固定外圈、中低转速电机拖动工况
frequency_family: BPFO
typical_features:
  - 时域 RMS 与峭度同步上升，冲击序列较内圈更稳定
  - 频谱在 BPFO 及其 2 倍、3 倍谐波处出现清晰离散峰
  - 外圈静止、缺陷位置固定，边带相对内圈不显著或很弱
  - 包络谱中 BPFO 族成组出现，峰高随故障尺寸单调增大
frequency_signature: BPFO ≈ 3.5851 × fr（6205 驱动端），谐波 2×BPFO、3×BPFO 处幅值抬升，边带较弱
criteria:
  - id: outer-rms-up | claim: RMS 相对基线上升 30% 以上，振动总量抬升 | metric: rms_ratio | op: ">=" | value: 1.3
  - id: outer-kurt-up | claim: 峭度相对基线上升 30% 以上，冲击序列稳定 | metric: kurtosis_ratio | op: ">=" | value: 1.3
  - id: outer-crest-up | claim: 波峰因子相对基线上升 20% 以上 | metric: crest_factor_ratio | op: ">=" | value: 1.2
  - id: outer-bpfo-peaks | claim: 谱峰在 BPFO 族（含 2、3 阶谐波）至少 2 阶处对齐 | metric: peak_match_orders | family: BPFO | op: ">=" | value: 2
  - id: outer-bpfo-rank | claim: 包络谱 BPFO 族能量为四个特征频率族中最高 | metric: envelope_rank | family: BPFO | op: "<=" | value: 1
  - id: outer-bpfo-margin | claim: BPFO 主导族量级余量不低于 1.15，主导族可判定 | metric: envelope_margin | op: ">=" | value: 1.15
sources:
  - name: CWRU Bearing Data Center | url: https://engineering.case.edu/bearingdatacenter/welcome | locator: 12k Drive End Bearing Fault Data 说明页中 Outer Race 故障条目（OR007/OR014/OR021）及驱动端故障频率系数表
  - name: Rolling element bearing diagnostics using the CWRU data: A benchmark study | url: https://doi.org/10.1016/j.ymssp.2015.04.021 | locator: 第 3 节 特征频率与诊断方法、第 4 节 外圈故障诊断案例
  - name: Bearing Fault Frequency Calculator and Guide | url: https://wertek.ai/engineering/vibration/bearing-frequencies/ | locator: SKF 深沟球轴承 62xx 系列表 6205 行（BPFO 3.5850）与计算说明
---

## 故障机理
外圈通常固定于轴承座，缺陷位置不随轴旋转，因此滚动体在承载区内规律地碾过缺陷，产生一组幅值稳定、间隔均匀的冲击序列。由于外圈不动，冲击强度不经历随轴旋转的周期性调制，故频谱以 BPFO 及其谐波为主，边带不明显，这一点是与内圈故障区分的关键。BPFO 值由滚动体数量、节圆直径与接触角决定，对 6205 约为轴频的 3.585 倍；当外圈缺陷落入非承载区时幅值可能偏弱，需在多点测量确认。

## 典型振动表现
- 时域指标：RMS 与峭度上升，冲击脉冲间隔均匀、幅值较一致
- 频谱：BPFO、2×BPFO、3×BPFO 处出现成组窄带峰，底部噪声抬升
- 调制特征：边带较弱；若出现明显 fr 边带，应怀疑存在复合故障
- 高频共振带：加速度包络在轴承共振区能量集中，谱峭度可定位最优频带
- 与基线对比：相对健康基线，BPFO 族峰高呈现稳定增长趋势

## 建议复核
- 依实际转速核算 BPFO 及谐波位置，核对峰值与理论值的偏差百分比
- 对比其他部位特征：若该族峰明显而边带很弱，则更支持外圈故障
- 在轴承座不同角度（承载区/非承载区）复测，排查外圈缺陷周向位置影响
- 结合包络解调与谱峭度最优频带，排除结构共振与其他激振源混淆
- 关注温度与润滑状态，确认是否由润滑不良诱发的表面损伤