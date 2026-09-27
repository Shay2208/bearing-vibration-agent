# 公开数据集说明

## 1. 数据来源

- 名称：**CWRU Bearing Data Center**（凯斯西储大学轴承数据中心）
- 主页：<https://engineering.case.edu/bearingdatacenter/welcome>
- 下载页：<https://engineering.case.edu/bearingdatacenter/download-data-file>
- 对应文献：Smith, W. A., & Randall, R. B. (2015). Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. *Mechanical Systems and Signal Processing*.

数据文件为公开的 `.mat`，脚本直接从官方站点下载，不附带任何私有数据。

## 2. 本首版固定的取值（不随请求变化）

| 维度 | 固定值 |
|---|---|
| 采样率 | 12000 Hz（12k Drive End 数据） |
| 负载 / 转速 | 0 hp / 1797 rpm |
| 测点 | `drive_end`，只取 `X{n}_DE_time` 变量 |
| 切窗 | 统一 **12000 点不重叠**片段（= 1 秒），第 k 个窗口起点 = `k * 12000` |
| 时间戳 | `t = i / 12000`，从 0 开始 |

统一采样率、统一测点、统一窗长与不重叠切窗，是为了让各状态的样本在同一尺度上可比。

## 3. 选用的 fileId 与状态

| fileId | 变量名 | condition | condition_label |
|---|---|---|---|
| 97 | `X097_DE_time` | `normal` | 正常 |
| 105 | `X105_DE_time` | `inner_race_fault` | 内圈故障 |
| 130 | `X130_DE_time` | `outer_race_fault` | 外圈故障 |
| 118 | `X118_DE_time` | `ball_fault` | 滚动体故障 |

四个故障文件均为 **0.007 inch** 故障直径；外圈故障位于 6 点钟方向。
另有 `corrupted_1797`（`condition = corrupted`）为手工构造的损坏样本，用于验证数据校验与错误返回，不对应任何真实工况。

## 4. 标签的使用边界（重要）

`condition` / `condition_label` **仅用于离线评测与样本标注，不进入在线诊断输入**。
在线链路只能看到波形数据（`timestamp, amplitude`），不允许把真实故障标签透传给模型或 Agent，否则评测结果不成立。

## 5. 已知局限

- **工况单一**：本首版只覆盖 0 hp / 1797 rpm 一种转速负载，模型在变转速、变负载下不一定成立。
- **数据泄漏风险**：若切窗时允许重叠、或把同一段连续信号的不同窗口分别划入训练集与测试集，会造成严重泄漏并高估指标。因此本仓库固定使用不重叠窗口，并建议按文件（甚至按工况）划分而非按窗口随机划分。
- **样本量小**：每个状态只有 1 秒片段，仅够跑通端到端演示与冒烟测试，不足以训练可靠模型。
- 数据本身来自实验台单点加速度计，与真实工业现场的噪声、传递路径差异较大。

## 6. 内置样例刷新

```bash
python data/prepare_data.py --target samples --force
```

离线环境可用本地 `.mat`：

```bash
python data/prepare_data.py --target samples --local-mat-dir <存放 .mat 的目录>
```