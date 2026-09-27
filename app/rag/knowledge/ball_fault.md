---
fault_type: ball_fault
title: 滚动轴承滚动体（钢球）故障
applicable: 单列深沟球轴承（CWRU 6205-2RS JEM SKF 驱动端），滚动体表面剥落或点蚀工况
frequency_family: BSF
typical_features:
  - 滚动体缺陷自转一周内两次撞击滚道，频谱在 2×BSF 附近出现峰值
  - 冲击幅值随滚动体自转姿态不断变化，时域表现为不规则、幅值起伏的脉冲
  - 峭度、波峰因子升高，但谱峰不如内外圈故障规整
  - 包络谱在 2×BSF 及谐波处抬升，峰幅波动较大
  - 包络能量分散：各特征频率族能量相当、无单一主导族，主次难以区分
frequency_signature: BSF ≈ 2.3568 × fr（6205 驱动端），特征峰通常出现在 2×BSF 附近，峰幅波动明显
criteria:
  - id: ball-rms-up | claim: RMS 相对基线上升 30% 以上，存在可解释的异常 | metric: rms_ratio | op: ">=" | value: 1.3
  - id: ball-kurt-mild | claim: 峭度相对基线变化不显著（小于 30%），不呈强冲击特征 | metric: kurtosis_ratio | op: "<" | value: 1.3
  - id: ball-crest-mild | claim: 波峰因子相对基线变化不显著（小于 20%），冲击规律性差 | metric: crest_factor_ratio | op: "<" | value: 1.2
  - id: ball-dispersed | claim: 包络谱各特征频率族能量相当，无单一主导族 | metric: dominant_ambiguous | op: "==" | value: true
  - id: ball-tie-bsf | claim: 并列族中包含本条目声明的 BSF 族，与并列现象自洽 | metric: tied_family | family: BSF | op: "==" | value: true
sources:
  - name: CWRU Bearing Data Center | url: https://engineering.case.edu/bearingdatacenter/welcome | locator: 12k Drive End Bearing Fault Data 说明页中 Ball（滚动体）故障条目（B007/B014/B021）及驱动端故障频率系数表
  - name: Rolling element bearing diagnostics using the CWRU data: A benchmark study | url: https://doi.org/10.1016/j.ymssp.2015.04.021 | locator: 第 3 节 特征频率与诊断方法、第 4 节 滚动体故障诊断案例
  - name: Bearing Defect Frequency Calculator: BPFO, BPFI, BSF and FTF Formulas with Worked Examples | url: https://iotbearings.com/how-to-calculate-bearing-defect-frequencies-from-datasheet/ | locator: "The Four Bearing Defect Frequencies" 一节中 BSF 公式与"缺陷每转撞击内外滚道各一次、故峰位于 2×BSF"的说明
---

## 故障机理
滚动体既绕轴公转，又绕自身轴线自转。当滚道表面存在剥落时，缺陷在滚动体自转一周内会分别撞击轴承两侧滚道各一次，因此频谱中更常见的是两倍于球自转频率的 2×BSF，而非 BSF 本身。随着滚动体在保持架内姿态不断变化，缺陷与滚道的接触角和承载状态持续改变，冲击幅值被不规则调制，导致时域脉冲高度起伏、频域谱峰不如滚道类故障稳定，这也是滚动体故障较难可靠识别、易被低估的原因。

## 典型振动表现
- 时域指标：峭度与波峰因子明显升高，脉冲幅值起伏大、规律性差
- 频谱：2×BSF 附近出现峰，峰高低于同等尺寸的滚道类故障
- 调制特征：峰幅随滚动体姿态周期性波动，边带结构零散
- 高频共振带：加速度包络能量分散，谱峭度最优频带定位结果可能不唯一
- 与基线对比：需较长时间趋势观察，相对基线的抬升往往缓于滚道类故障

## 建议复核
- 以实际转速核算 BSF 与 2×BSF，重点核对 2×BSF 附近是否存在峰值
- 结合其他特征频率共同判断，排查是否伴随复合故障
- 复测多点位与多次采集，因滚动体故障信号重复性差，单次测量不足以免判
- 用包络解调与谱峭度选择最优频带后重建信号，检验冲击周期性是否稳定
- 检查润滑与异物污染，滚动体表面损伤常与润滑劣化或颗粒污染并存