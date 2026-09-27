---
fault_type: normal
title: 未见明显故障特征（健康/基线状态）
applicable: 常规运行工况下，特征指标与历史基线接近、无显著谐波抬升与冲击成分
typical_features:
  - 时域 RMS、峭度等指标与历史基线接近，波动在正常范围内
  - 频谱以转频及其低次谐波为主，无轴承故障特征频率峰及其谐波族
  - 无高频冲击与宽带底噪抬升，包络信号无明显周期性脉冲
  - 变化率平稳，未出现指标阶梯式或单调快速上升
frequency_signature: 仅见 1×转频及少量低次谐波，无亚同步保持架频率峰，也无数倍频特征峰族
criteria:
  - id: normal-no-change | claim: 无任何显著变化特征（RMS、峭度、峰值相对基线变化均在 30% 以内） | metric: changed_feature_count | op: "<=" | value: 0
  - id: normal-no-shift | claim: 主频未迁移（迁移量小于 1 Hz） | metric: dominant_shift_hz | op: "<" | value: 1.0
sources:
  - name: CWRU Bearing Data Center | url: https://engineering.case.edu/bearingdatacenter/welcome | locator: Normal Baseline Data 说明页（正常基线数据用于与各类故障数据对照）
  - name: Vibration Insights for Maintenance Decisions | url: https://www.aquip.com.au/from-data-to-action-vibration-insights-for-maintenance-decisions/ | locator: "Establishing Baseline Measurements and Alarm Thresholds" 一节：以基线为参照，ISO 10816/20816 分区阈值与变化率判据
  - name: Rolling element bearing diagnostics—A tutorial | url: https://doi.org/10.1016/j.ymssp.2010.07.017 | locator: 第 2 节 特征频率定义：健康轴承不出现由几何决定的离散特征频率及其谐波
---

## 故障机理
在正常状态下，轴承滚道与滚动体之间为连续的弹性接触，无局部缺陷引发的周期性冲击，因此振动主要来自转子旋转、结构传递与润滑流体作用，能量集中在转频及其低次谐波上。由于不存在缺陷表面的反复撞击，由轴承几何决定的各阶离散特征频率不会出现显著峰值；时域指标与历史基线保持一致，说明载荷、转速与润滑条件稳定。判断"无明显异常"的依据是"与基线接近且无特征频率抬升"，而非绝对幅值本身。

## 典型振动表现
- 频谱：以 1×转频及少量低次谐波为主，高次谐波与底噪水平低
- 特征频率：各阶轴承特征频率附近无离散峰，无成族谐波
- 时域指标：RMS、峭度、波峰因子围绕基线小幅波动，无突增
- 包络/高频区：无周期性冲击，无宽带能量抬升
- 趋势：多次测量结果一致，变化率平缓

## 建议复核
- 与同一测点历史基线逐项对比，确认各指标落在正常波动带内
- 按实际转速换算各阶特征频率位置，逐一核对无峰值
- 关注变化率而非单次绝对值，排除工况（转速、负载）变化引起的假异常
- 若指标虽在阈值内但呈现持续上升趋势，应缩短复核周期并留存记录
- 确认测点与采集参数未变（传感器位置、采样率），保证基线的可比性