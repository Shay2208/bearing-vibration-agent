---
fault_type: unbalance
title: 转子不平衡
applicable: 电机-轴承-转子系统中的转子质量偏心，非轴承局部损伤类故障
typical_features:
  - 频谱以 1×转频（1X）为主导单峰，谐波能量很少
  - 振动主要集中在径向（水平/垂直），轴向分量通常较低
  - 幅值随转速平方增长，转速越高越明显
  - 无明显轴承特征频率与高频冲击成分
frequency_signature: 主导峰位于 1×fr（转频），几乎无 2×、3× 谐波，径向幅值远大于轴向
criteria:
  - id: unbalance-1x-dominant | claim: 主频落在 1×转频附近（偏差 10% 以内），与不平衡的 1X 主导一致 | metric: dominant_near_shaft | order: 1 | op: "==" | value: true
sources:
  - name: Machinery fault diagnosis using vibration analysis（Practical Machinery Vibration Analysis and Predictive Maintenance 第 5 章） | url: https://www.drahmednagib.com/onewebmedia/Eng._Akram_Machinery_Fault_Diagnosis.pdf | locator: 5.2.1 节 Unbalance：各类不平衡的 FFT 谱均以 1× rpm 主导，幅值随转速平方变化
  - name: Vibration Analysis Basics: What Your Equipment is Telling You | url: https://dovient.com/learning/vibration-analysis-basics | locator: "Frequency: The Key to Diagnosis" 一节 Imbalance 谱型说明（强 1X、几乎无其他显著频率）
---

## 故障机理
转子质量分布相对旋转中心线不均匀，旋转时产生与转速平方成正比的离心力，经轴承传递为机壳振动。该离心力每转一次达到峰值，因此振动以 1×转频为主要分量，且因力沿径向指向，径向振动显著强于轴向。不平衡本身不产生轴承滚道冲击，故频谱中不会出现 BPFO/BPFI 等非整数倍特征频率；但长期运行会加速轴承与密封的疲劳，所以识别不平衡有助于防止其诱发二次损伤。

## 典型振动表现
- 频谱：1×转频处单峰突出，2×、3×谐波很小，高次谐波与底噪均低
- 方向：径向（水平、垂直）幅值高，轴向分量低，与不平衡类型无关
- 转速相关：幅值随转速平方增长，升速过程中 1X 峰快速变大
- 相位：同一转子上两支承径向相位稳定、近似同相
- 与轴承故障区别：无 1×以外的非整数倍特征频率，无高频冲击

## 建议复核
- 记录 1X 幅值随转速的变化，验证是否呈平方关系
- 比较多点径向与轴向振动，确认以径向、低轴向为特征
- 检查是否存在 2×转频等谐波抬升，以排除同时存在不对中
- 核对是否有 BPFO/BPFI 等轴承特征峰，避免把转子问题误判为轴承故障
- 安排动平衡校正前后对比测量，用幅值下降验证诊断结论