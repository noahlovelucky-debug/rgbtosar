# HiFC 无配对 RGB→SAR 下一阶段优化方案

本文档记录当前 `hifc_unpaired_conditioned_v1` 基线之后的可归因优化路线。
目标不是继续堆叠 loss，而是同时解决两个问题：

1. 生成图看起来像 SAR，但真实 SAR 上的车型信息不能迁移；
2. native SAR classifier、条件判别器或 RGB identity 可能提供捷径，让训练指标很高，
   却没有学到真实目标域的稳定散射结构。

所有正式实验都必须使用固定 parent、固定 data split、固定 seed 和同一 TSTR 协议。
真实 test 集永远不能进入训练或 meta-query。

## 1. 当前基线和问题定位

最终 HiFC 运行是 8 卡 DDP、120 epoch、全条件训练（X/KU、HH/HV/VH/VV、15/30/45/60°）。
生成器训练结束后的独立 TSTR 是：

| 指标 | 当前值 |
|---|---:|
| 真实 X/HH Top-1（3 seed mean） | 48.32% |
| 真实 X/HH Top-5（3 seed mean） | 74.42% |
| 方位 Top-1 | 60.92% |
| 方位 circular MAE | 42.49° |
| 15/30/45/60° Top-1 | 44.53/54.32/55.27/38.86% |
| native class accuracy | 约 100% |

旧 V1 的相同类型 TSTR 约为 14.75%/39.06%，所以 HiFC 路径已经明显改善；但生成训练集
接近 100%，真实测试只有 48%，仍存在明显 synthetic→real domain gap。60° 是最弱的
俯视角域。native 100% 不能证明真实泛化，因为 native classifier 的 embedding 和
geometry heads 直接参与 G 的梯度。

当前 generator 的有效监督大致分为：

```text
RGB identity/cross-view       保留 RGB 车型身份
LTC                            匹配无配对 SAR 局部纹理统计
SFM                            native embedding + D feature moments
geometry                       native band/pol/depression/azimuth heads
adversarial                    条件 PatchGAN 真实性
```

其中 LTC 原始数值约 `1e-4`，而 SFM 和 geometry 的梯度更大。第一优先级不是改纹理项，
而是确认 native 路径是否把 G/E 推向 native teacher 的高频决策边界。

## 2. 已实现的第一步：A0 native 梯度断路

训练入口 `code/train_hifc_unpaired_sar_gan.py` 新增：

```bash
--native-gradient-mode {full,embedding_off,all_off}
```

含义如下：

| 模式 | SFM 的 native embedding 梯度 | geometry 梯度 | teacher 数值诊断 |
|---|---|---|---|
| `full` | 开 | 开 | 开 |
| `embedding_off` | 关 | 开 | 开 |
| `all_off` | 关 | 关 | 开 |

`all_off` 不是删除 loss。它仍计算并记录 SFM/geometry 的数值，唯独将 fake→native
teacher 的反向路径 detach；D feature moments、LTC、GAN 和 RGB identity 仍正常更新。
这样可以把“teacher 指标仍然很高”和“teacher 是否在塑造 G”区分开。

SFM 的梯度路径为：

```text
fake -> frozen native embedding -> cosine/mean SFM -> G/E       (full only)
fake -> frozen D feature       -> feature moments -> G          (all modes)
real -> detach native embedding/D feature                      (all modes)
```

geometry 在 `all_off` 下只用于日志：

```text
fake -> frozen native auxiliary heads -> band/pol/dep/az CE
                                      X (no G/E gradient in all_off)
```

### A0 的实验方式

A0 是反事实诊断，不直接宣布为最终模型。推荐从同一个 parent、同一 seed 启动两臂：

```bash
# control
python code/train_hifc_unpaired_sar_gan.py ... \
  --native-gradient-mode full \
  --output runs/hifc_a0_full_seed2718

# candidate
python code/train_hifc_unpaired_sar_gan.py ... \
  --native-gradient-mode all_off \
  --output runs/hifc_a0_all_off_seed2718
```

先做 10 epoch/固定少量 batch 的 paired screen，再对通过者做至少 3 seed 的短确认。
从 epoch 120 resume 只能回答“已有 shortcut 能否被维持”，不能回答 shortcut 是否在训练
早期形成；要证明形成机制，必须从同一初始化或同一早期 milestone 重跑双臂。

选择判据：

- native accuracy 不参与排序；
- 真实 X/HH TSTR Top-1、Top-5 是主指标；
- 方位 MAE、四个 depression marginal、频谱统计和多噪声 diversity 是辅助指标；
- `all_off` 若 TSTR 几乎不降而 native 下降，说明 native 是 shortcut，应继续断路；
- `all_off` 若 TSTR 显著下降，说明 teacher 也包含有效身份信息，不能恢复原 hard CE，
  应转入独立多编码器替代；
- 任何候选必须保持视觉非劣化，不能以单个 native accuracy 上升作为通过理由。

## 3. 下一项创新：Meta-TSTR Scattering Generator

A0 只诊断已有路径。真正直接针对 domain gap 的训练信号是：

> 让“用生成 SAR 训练出来的分类器”在未参与该分类器训练的真实 SAR 上立即有效。

这不是把 test 标签反复喂回训练，而是在训练集内部做严格三分：

```text
R_gan   : 当前 GAN 的 real SAR，供 D/LTC/SFM 使用
R_meta  : 只作为 meta-query，不能进入 D/SFM/statistics
R_audit : 完全封存，训练和调参都不能读取
```

`R_meta` 与 `R_gan` 至少按车型×俯视角×波段×极化分组隔离；`R_audit` 使用独立
validation/test split。最终官方 TSTR 仍只在封存的真实 X/HH 上进行。

### 3.1 一步虚拟分类头

对一个 synthetic support batch 和一个 real query batch：

```text
x_s = G_theta(RGB_s, condition_s, noise_s)
x_q = real SAR from R_meta

phi' = phi - alpha * grad_phi CE(C_phi(x_s), y_s)
L_meta = CE(C_phi'(x_q), y_q)
```

`C_phi` 是轻量、每个 episode 重新初始化的 class head 或小 probe；probe 参数、query
特征和 native/D 参数都冻结。对 `theta` 的超梯度为：

```text
grad_theta L_meta =
  -alpha * (d2 CE_support / dtheta dphi)^T * grad_phi' CE_query
```

它要求 synthetic feature 不只是能被 native teacher 分类，而是能让一个只看 synthetic
的分类头在真实 query 上泛化。这个目标与最终 TSTR 的因果方向一致。

### 3.2 防止 meta 信号再次变成 teacher shortcut

- 每 200 step 重置 probe；不保存 probe 的持久状态；
- 轮换三个小 probe 结构和轻微增强，另加 band/polarization mask；
- `lambda_meta` 从 epoch 5 到 15 线性 ramp，最大值建议 0.2；每 4 个 G step 计算一次；
- meta 梯度范数限制为当前 realism 梯度的 0.25；与 realism 冲突时只对 meta 梯度做
  PCGrad 投影；
- meta-query 只用 train split，官方 TSTR test 永远不进入 manifest；
- 记录 support CE、query CE、hypergradient norm、gradient cosine 和 query hash。

首个实现不应同时改变 D、LTC、SFM、数据 sampler 或生成器结构。先在当前 HiFC
generator 上验证 meta 梯度有限差分正确，再考虑结构创新。

### 3.3 Meta-TSTR 的硬门

机制门：

- 有限差分与 autograd 二阶梯度误差 ≤ `1e-4`；
- 每个 episode support CE 下降；real query CE 在大多数 episode 下降；
- probe/E/D/native 的非目标梯度为 0；主 E 的梯度不被意外修改；
- `lambda_meta=0` 与旧基线 bitwise no-op；
- meta-query 记录固定、未读取 audit/test。

效果门：

- 三个 GAN seed 都先过原有 7 项视觉门；
- 相对 A0/当前 parent，真实 X/HH Top-1 和 Top-5 均值至少提升 0.5 pp，至少 2/3
  seed 提升，单 seed 不超过既定退化底线；
- 60° marginal Top-1 不下降，最好提升至少 2 pp；
- 独立 classifier seed 复核后提升仍存在；
- native accuracy 上升但 TSTR 不升时直接关闭。

## 4. 最终论文级结构：稀疏复散射场 + Meta-TSTR

在 Meta-TSTR 当前结构验证通过后，再引入结构变量，避免一次大改无法归因。建议的
`Meta-TSTR Scattering Generator` 将 SAR 生成分成三个可解释层：

```text
RGB multi-view canonical token + target condition
        |
        +--> sparse scattering amplitude A(x,y)
        +--> phase/anisotropy field phi(x,y)
        |
        +--> condition-dependent anisotropic PSF / coherent sum
                |
                +--> clean reflectivity
                        |
                        +--> independent learned speckle renderer
                                |
                                +--> observed SAR
```

关键点：

1. RGB 多视角先聚合成 canonical token，避免单个侧视图的外观细节被当成 SAR 像素；
2. 输出稀疏散射中心、方向性和相位，而不是直接把 RGB feature 拷贝成强度纹理；
3. X/HH、波段、极化、俯视角控制 PSF 和相干叠加参数；
4. speckle 作为独立随机 renderer，避免 classifier 通过固定噪声纹理识别车型；
5. unconditional multi-scale/FFT critic 只负责真实 SAR 频谱和纹理，conditional critic
   负责条件与车型；native classifier 只做审计或 meta-query 的 real-only probe，不再给
   G hard class CE。

对应 loss 仍应保持少而正交：

```text
L_meta       : synthetic support -> real query 泛化（核心创新）
L_adv_real   : unconditional/multi-scale SAR realism
L_cond       : target condition + class consistency
L_scatter    : 散射稀疏度、各向异性和频谱统计
L_diversity  : 同条件多 noise 的真实域 diversity 匹配
L_rgb        : 仅约束 RGB identity，不让 SAR teacher 反向改写 E
```

`L_meta` 不是简单地再加一个 native CE；它替换“native teacher 直接告诉 fake 应该长
什么样”的 shortcut 监督。所有结构项都不能恢复 RGB/SAR 的硬像素对齐，因为本任务的
RGB 是车侧视图而 SAR ROI 是不同视角/实例。

## 5. loss 和模块的保留策略

| 组件 | 当前处理 | 下一阶段策略 |
|---|---|---|
| RGB identity + cross-view | 保留 | 先只限制其更新 E；不要同时删两项 |
| LTC | 保留 | 保持无配对统计，后续可拆亮度/残差/频谱做单变量 |
| SFM native embedding | 当前保留 | 先 A0 断路；若 TSTR不降则永久不回传 |
| geometry auxiliary | 当前保留 | 先 A0 断路；只作为条件审计或 real-only probe |
| conditional PatchGAN | 保留 | A0 后再测 unconditional/conditional 双 critic |
| pixel64/SSIM | HiFC 当前没有 | 回到 paired V1 时只保留弱 translation-tolerant 项；不做硬 RGB/SAR 对齐 |
| physics/scattering | V1 有效但可能含 pixel-like map | 先保留，再替换为散射统计/频谱项做单变量 |
| feature match/statistics | 不凭数值相似删除 | 通过梯度范数和 TSTR 冗余实验决定合并 |

## 6. 推荐执行顺序

```text
A0 native-gradient all_off
  -> 若通过，固定为新 parent
  -> Meta-TSTR 当前结构（只新增一个 meta 项）
  -> Meta-TSTR 的 class-complete support / cross-depression query（一次只改一个）
  -> unconditional + conditional 双 critic
  -> 稀疏复散射场/条件 PSF 结构变量
  -> 三 seed GAN × 三 seed TSTR 的最终确认
```

每一步都必须保留 control、候选、mechanism audit、视觉曲线和真实 X/HH TSTR JSON。
任何“native 指标更高但 TSTR 不升”的分支都应被判定为 shortcut 失败，而不是继续调权重。

当前代码已经实现 A0 开关和梯度测试；Meta-TSTR 与散射场结构应在 A0 结果确认后
单独提交，避免把多个创新点混成不可解释的大改动。
