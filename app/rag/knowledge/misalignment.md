---
fault_type: misalignment
title: 轴系不对中（联轴器对中不良）
applicable: 电机与负载通过联轴器连接的轴系，平行/角度/复合不对中工况
typical_features:
  - 频谱以 2×转频（2X）为主要分量，常同时存在 1X、3X
  - 轴向振动明显偏高，是区别于不平衡的关键特征
  - 联轴器两侧测点相位接近 180°（反相）
  - 往往伴随较高谐波与底噪，但无轴承局部特征频率
frequency_signature: 主导峰位于 2×fr（转频二倍频），1×、3× 同时存在，轴向振动显著高于径向同类故障
criteria:
  - id: misalign-2x-dominant | claim: 主频落在 2×转频附近（偏差 10% 以内），与不对中的 2X 主导一致 | metric: dominant_near_shaft | order: 2 | op: "==" | value: true
sources:
  - name: Dynamic Response of a Rotating Assembly under the Coupled Effects of Misalignment and Imbalance | url: https://doi.org/10.1155/2020/8819676 | locator: 第 1 节 Introduction 与第 3 节 讨论：不平衡主要对应 1X 峰，不对中主要对应 2X 峰且轴向振动升高
  - name: Vibration Caused by Misalignment | url: https://xyobalancer.com/vibration-caused-by-misalignment/ | locator: 正文对平行/角度不对中的谱特征说明：2X 显著、轴向振动强、联轴器两侧 180° 相位差
  - name: Shaft Alignment: Why Even Perfect Balancing Cannot Replace It | url: https://vibromera.uk/knowledge-base/cluster/shaft-alignment | locator: "Signs of misalignment in the vibration spectrum" 一节及不对中/不平衡对比表（2X 主导、轴向高）
---

## 故障机理
两根轴的中心线在联轴器处不共线（平行偏移、角度偏斜或两者复合）时，联轴器在每转中被迫完成两次拉伸与压缩的循环，从而在 2×转频处产生主要激励；同时角度不对中把部分振动力沿轴向传递，使轴向振动显著升高。由于激励源自联轴器与轴承的约束受力，其频率与转频成整数倍关系，不出现轴承滚道特征频率。不对中会长期附加载荷于轴承与密封，是轴承早期失效和轴封泄漏的常见诱因。

## 典型振动表现
- 频谱：2×转频为主要峰，1X、3X 亦常出现，谐波族较丰富
- 方向：轴向振动明显（角度不对中尤甚），与不平衡的低轴向形成对照
- 相位：联轴器两侧测点相位接近 180° 反相
- 噪声底：较不平衡略高，但一般无色散性高频冲击
- 与轴承故障区别：峰位落在转频整数倍，无 BPFO/BPFI 等非整数倍特征频率

## 建议复核
- 分别测量径向与轴向振动，确认 2X 主导并伴随高轴向分量
- 联轴器两侧同步采集，核对相位是否接近 180°
- 检查 1X 与 2X 的相对关系，判断是否同时存在不平衡（需先对中后平衡）
- 排除机械松动（多倍谐波、噪声底更高）等相似谱型干扰
- 使用激光对中仪复测并记录校正前后振动变化，验证诊断