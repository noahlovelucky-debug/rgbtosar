# 最终 HiFC 风格无像素配对 RGB→SAR 工作流

本文档描述当前最终采用的 `hifc_unpaired_conditioned_v1`。它是一个针对
SOC_40classes_cut 的、按车型弱匹配的 RGB→SAR ROI 生成基线：RGB 车辆侧视图和
SAR ROI 没有同坐标关系，因此训练中不做 RGB/SAR 像素重建，也不调用平移对齐。

文档中的代码入口是：

- `code/hifc_unpaired_sar_gan.py`：编码器、生成器、判别器和 loss 实现；
- `code/train_hifc_unpaired_sar_gan.py`：数据切分、训练循环、DDP、EMA、checkpoint；
- `code/render_hifc_unpaired_sar.py`：从 checkpoint 按目标条件渲染 SAR；
- `code/train_generated_sar_classifier_64.py`：独立 TSTR 分类器评估。

## 1. 最终结论

最终 GAN 使用 8 张 A100 完成 120 个 epoch。最后的模型文件没有上传到 GitHub，避免
仓库被 356 MB checkpoint 占满，checkpoint 保存在训练服务器：

```text
/data/newdata/A25_T37_down_大图/code/runs/
  hifc_unpaired_all_conditions_ddp/epoch_120.pt
  hifc_unpaired_all_conditions_ddp/latest.pt
```

最终 checkpoint 的 EMA encoder/generator 用于渲染和 TSTR。TSTR（Train on Synthetic,
Test on Real）只用生成 SAR 训练一个全新的 40 类 SAR 分类器，再用真实 X/HH 测试集
测试，测试集共 5260 张；真实 SAR 像素没有进入该分类器的训练过程。

| TSTR 分类器 seed | Top-1 | Top-5 |
|---:|---:|---:|
| 415 | 48.95% | 74.26% |
| 1618 | 47.59% | 74.30% |
| 31415 | 48.42% | 74.71% |
| 平均 ± seed std | **48.32% ± 0.56%** | **74.42% ± 0.21%** |

旧 V1 在相同类型的真实 X/HH TSTR 上约为 Top-1 `14.75%`、Top-5 `39.06%`；因此
当前结果有明确改善，但仍不是完美的真实域复现。生成训练集上的分类准确率接近 100%，
真实测试约 48%，说明仍有域差异和一部分生成器 shortcut。最终 native teacher 的
车型准确率约 100% 只能作为训练诊断，不能替代 TSTR。

## 2. 数据和任务定义

### 2.1 数据目录

```text
/data/newdata/A25_T37_down_大图/A02/RGB
/data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train
/data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/test
```

数据包含 40 个车型。SAR 每条记录由一个 TIFF 和同名 XML 组成，XML 提供车型、
波段、极化、俯视角和方位角；RGB 是按车型分目录保存的多视角 PNG。当前全条件训练
使用：

```text
band:         X, KU
polarization: HH, HV, VH, VV
depression:   15, 30, 45, 60 degrees
azimuth:      XML 中的实际方位角
```

全条件扫描共有 68091 条有效 TIFF/XML 记录。最终训练脚本按 `(class, band, pol,
depression)` 分组，用固定 seed 的 SHA256 顺序取 15% 验证集：

```text
train records:      57881
validation records: 10210
epoch_size:         24000 random samples
```

验证集的分组切分写入 `split_manifest__all_all_all.json`，后续 resume 不重新随机切分。
测试集只在独立 TSTR 中使用 X/HH 原图，不参与 GAN 的 loss。

### 2.2 RGB/SAR 的弱匹配关系

一个训练 item 的来源如下：

1. 读取一条真实 SAR TIFF/XML，得到真实 ROI、车型标签和目标 SAR 条件；
2. 在同车型 RGB 文件夹中随机选一个源视角 `rgb`；
3. 再随机选一个不同的同车型视角 `rgb_alt`；
4. RGB 与 SAR 只保证车型相同，不保证同一实例、同一视角、同一时间或同一像素位置；
5. SAR ROI 是 64×64，RGB 输入缩放到 128×128；
6. RGB 的源视角角度和 SAR bbox 宽高不放进目标条件，避免让网络依赖不可靠信息。

因此任务不是 pix2pix paired translation，而是：

```text
same-class RGB identity + target SAR acquisition condition
                      -> SAR ROI distribution
```

### 2.3 输入增强

RGB 的每个视角在读取时使用轻微 gain、bias 和高斯噪声增强；真实 SAR 和生成 SAR
送入判别器或 LTC 前使用相同形式的 gain、bias 和小幅乘性扰动。训练不使用会改变
角度语义的水平翻转或几何变换。

## 3. 12 维目标条件

`condition_from_batch()` 输出固定顺序的 12 维向量：

```text
[ azimuth_sin, azimuth_cos,
  dep_15, dep_30, dep_45, dep_60,
  band_X, band_KU,
  pol_HH, pol_HV, pol_VH, pol_VV ]
```

### 方位角

方位角 `theta` 使用 `sin(theta)` 和 `cos(theta)`，所以 0° 与 360° 在输入空间中连续，
不会产生普通数值编码的断点。native teacher 的辅助监督为了分类统计，另外把方位角
量化为 12 个 30° bin：

```python
azimuth_bin = ((azimuth + 15) % 360) // 30
```

### 俯视角、波段和极化

俯视角使用四维 one-hot；波段使用 `[X, KU]` two-hot；极化使用
`[HH, HV, VH, VV]` four-hot。训练时使用 `all`，否则常量条件无法学习：例如只训练
X/HH 时，模型不能从数据中真正学习 X/KU 或 HH/HV/VH/VV 的变化。

## 4. 完整架构

![整体架构与梯度路径](visualizations/hifc_unpaired_final/workflow_overview.png)

### 4.1 RGB 身份编码器 `LargeRGBIdentityEncoder`

文件：`code/dual_component_sar_gan.py`。

输入为 `[B, 3, 128, 128]`，经过四个 stride-2 stage：

```text
stage 1: 3   -> 64   , spatial 128 -> 64
stage 2: 64  -> 128  , spatial 64  -> 32
stage 3: 128 -> 256  , spatial 32  -> 16
stage 4: 256 -> 512  , spatial 16  -> 8
```

每个 stage 是两层卷积、GroupNorm 和 SiLU。最后对 8×8 特征做全局平均池化，再经过
Linear、LayerNorm 和 SiLU 得到：

```text
identity z:       [B, 512]
RGB classifier:   [B, 40]
RGB pyramid:      [B, 64, 64, 64]
                  [B, 128, 32, 32]
                  [B, 256, 16, 16]
                  [B, 512, 8, 8]
```

`rgb` 和 `rgb_alt` 使用同一套 encoder 权重。encoder 的 40 类 head 只用于
`L_rgb_identity`；四层 pyramid 都作为生成器的空间调制条件，防止生成器只拿一个全局
向量而丢失车身的空间轮廓。逐层张量和调制细节见
[`HIFC_EPOCH120_LAYERWISE_ARCHITECTURE_ZH.md`](HIFC_EPOCH120_LAYERWISE_ARCHITECTURE_ZH.md)。

### 4.2 一阶段生成器 `HIFCUnpairedGenerator`

文件：`code/hifc_unpaired_sar_gan.py`，继承
`one_stage_wavelet_sar_gan.py` 的 `OneStageWaveletSARGenerator`。

它是一个共享 decoder，一次 forward 同时产生 clean reflectivity 和随机观测散斑：

```text
z [512] + condition [12]
        |
        +-- condition MLP: 12 -> 256 -> 256
        |
        +-- concat -> FC -> feature [B,512,4,4]
                              |
                              +-- block 1: 4  -> 8
                              +-- block 2: 8  -> 16
                              +-- block 3: 16 -> 32
                              +-- block 4: 32 -> 64
                                      |
                                      +-- clean_head -> clean [B,1,64,64]
                                      +-- noise head -> log_noise [B,1,64,64]
                                      +-- compose -> observed [B,1,64,64]
```

每个 decoder block 包含：

1. 双线性上采样；
2. 固定 `[1,2,1]` separable blur，降低 resize 的棋盘格伪影；
3. depthwise condition convolution 产生 SPADE-like scale/bias；
4. 两层卷积、GroupNorm、SiLU 和 residual skip；
5. 对应尺度的 RGB pyramid 通过 depthwise 3×3 和 1×1 卷积产生 SPADE-like
   `scale/bias`，调制 decoder 的 GroupNorm 后特征；它不直接加到 decoder feature。

clean 输出为 `tanh` 的一通道幅度图。空间噪声是一个 `[B,1,64,64]` 的随机场：

```text
correlated = AvgPool3(noise)
random_field = 0.70 * noise + 0.30 * correlated
scale      = 0.04 + 0.38 * sigmoid(noise_scale(feature, clean, correlated))
bias       = 0.12 * tanh(noise_bias(feature, clean, correlated))
log_noise  = scale * random_field + bias
log_noise  = mean-center over H/W, then clamp to [-0.8, 0.8]
```

SAR 观测合成在幅度域进行：

```text
clean_amplitude = (clean + 1) / 2
observed        = clamp(clean_amplitude * exp(log_noise), 0, 1)
```

最终送入判别器和大部分 loss 的是 `observed`；`clean` 主要作为可解释的去散斑反射率
输出保存。模型 forward 返回：

```text
(clean, log_noise, observed, final_feature)
```

### 4.3 共享条件判别器 `HIFCConditionedDiscriminator`

文件：`code/hifc_unpaired_sar_gan.py` 和 `code/v4_spade_gan.py`。

最终模型只有一个共享的 conditional projection PatchGAN：

```text
SAR [B,1,64,64]
       -> spectral Conv 64 -> 128 -> 256 -> 512
       -> patch score Conv + mean
       -> projection(class embedding + condition embedding)
       -> scalar score
```

class embedding 是 40 类车型 embedding；geometry MLP 把 12D 目标条件映射到 512 维，
与最后的 feature map 做 projection。判别器同时返回：

```text
score:    [B]
feature:  [B,512,4,4]
```

真实 SAR 还会配上 batch 中轮换后的错误车型条件和错误采集条件，训练 D 识别
`real image + wrong label/condition`。当前最终模型不是 K+1 真假分类器，也没有单独的
SAR 分类器 head；原代码中的 `SARClassDiscriminator64` 是历史实验模块，不在这个
最终 HiFC 训练中实例化。

### 4.4 冻结 native SAR teacher

文件：`code/sar_classifier_64.py`；checkpoint：

```text
code/server_results/sar_native64_multitask_v1/best.pt
```

teacher 只接收一通道 SAR 强度图，不接收条件 metadata：

```text
stem: 1 -> 48
residual stages: 48 -> 96 -> 192 -> 384
embedding: 384
class head:        40 classes
band head:          2
polarization head:  4
depression head:    4
azimuth head:      12 bins
```

训练 GAN 时 teacher 的参数全部 `requires_grad=False` 并保持 `eval()`；但是 fake 图像
仍允许对 teacher 输入求梯度，这样 `L_sfm` 和 `L_geometry` 可以把梯度传回 generator。
真实 SAR 的 teacher feature 在 `no_grad/detach` 下计算，不能反向更新 teacher 或把
真实样本泄漏进 generator 的参数计算。

## 5. Loss 总览

顶层有 **5 个 generator loss + 1 个 discriminator aggregate loss**。CSV 中的
`disc_wrong_class`、`disc_wrong_condition` 和 `r1` 是 D aggregate 的诊断子项，
不是额外的顶层优化器。

最终 generator 权重为：

```text
L_G = 1.0 * L_adv
    + 1.0 * L_rgb_identity
    + 2.0 * L_ltc
    + 2.0 * L_sfm
    + 0.3 * L_geometry
```

第一个 epoch 的 `L_adv` 暂时关闭，作为 adversarial warmup；从第二个 epoch 开始恢复
权重 1.0。

### 5.1 `L_rgb_identity`：RGB 身份和跨视角一致性

代码：`rgb_identity_loss()`。

```text
L_rgb_identity = 0.5 * CE(logits(rgb),     class, label_smoothing=0.03)
               + 0.5 * CE(logits(rgb_alt), class, label_smoothing=0.03)
               + 0.5 * (1 - cosine(z(rgb), z(rgb_alt)))
```

它保证两个 RGB 视角都能识别同一个车型，并让两个视角的 512 维身份向量接近。这个
loss 只直接更新 RGB encoder；它不比较 RGB/SAR 像素，也不直接判断 SAR 是否真实。

### 5.2 `L_adv`：条件对抗真实性

代码：训练循环中的：

```python
L_adv = -D(fake_observed, class_id, target_condition).mean()
```

G 要让生成图在指定车型、方位、俯视角、波段和极化条件下获得更高 D 分数。D 看到
detached fake，因此 D 步不会把梯度传进 G；G 步冻结 D 参数，但保留从 D 输入到 fake
的梯度。

### 5.3 `L_ltc`：无像素配对的局部纹理统计

代码：`local_texture_signature()` 和 `local_texture_contrast_loss()`。

先把图像转成幅度：

```text
amplitude = clamp((image + 1) / 2, 0, 1)
```

再计算：

```text
local3  = AvgPool3(amplitude)
local7  = AvgPool7(amplitude)
res3    = amplitude - local3
res7    = amplitude - local7
cont3   = res3 / (abs(local3) + 0.04)
cont7   = res7 / (abs(local7) + 0.04)
haar    = horizontal/vertical/diagonal Haar detail energy
```

每个量只取每张图的 `mean`、`std`、`abs-mean`，然后再取 batch mean/std，最终以
SmoothL1 比较 fake 与 real 的统计签名。它比较的是局部纹理分布，不比较同一坐标：

```text
禁止：fake[:, :, y, x] 与 real[:, :, y, x] 的 L1
禁止：_align_translation
禁止：RGB/SAR 同坐标重建
```

### 5.4 `L_sfm`：深层语义特征映射

代码：`semantic_feature_mapping_loss()`。

teacher 对 fake 和 real 分别输出 384 维 pre-classifier embedding，D 也输出最后一层
feature map。真实特征全部 detach：

```text
fake_norm = normalize(fake_teacher_feature)
real_norm = normalize(real_teacher_feature.detach())

L_cosine = 1 - mean(sum(fake_norm * real_norm, dim=1))
L_mean   = SmoothL1(mean(fake_norm), mean(real_norm))
L_Dfeat  = SmoothL1(
             concat(mean(fake_D_feature), std(fake_D_feature)),
             concat(mean(real_D_feature.detach()), std(real_D_feature.detach())))

L_sfm = L_cosine + 0.5 * L_mean + 0.5 * L_Dfeat
```

这里使用 native teacher 的深层 embedding，而不是 native class hard CE；加入 D 的
feature moments 是为了让生成图靠近判别器看到的 SAR feature 分布。它仍然依赖一个
冻结 teacher，因此最终是否学到真实车型信息必须用 TSTR 验证。

### 5.5 `L_geometry`：四个采集条件辅助任务

代码：`geometry_auxiliary_loss()`。

从同一个 fake teacher feature 接四个冻结辅助 head：

```text
L_geometry = 0.25 * (
    CE(band_head(fake),          band_target)
  + CE(polarization_head(fake),  polarization_target)
  + CE(depression_head(fake),    depression_target)
  + CE(azimuth_head(fake),       azimuth_bin_target)
)
```

目标全部来自 XML/metadata；其中 band、polarization、depression、azimuth 是目标 SAR
条件，不能把它们误认为车型分类准确率。最终训练中不加入 native teacher 的 40 类
hard class CE，以减少直接优化旧分类器决策边界的捷径。

### 5.6 `L_D`：判别器 aggregate

主 hinge：

```text
L_hinge = mean(ReLU(1 - D(real, class, condition)))
        + mean(ReLU(1 + D(fake.detach(), class, condition)))
```

错误条件负样本：

```text
L_wrong_class     = mean(ReLU(1 + D(real, wrong_class, condition)))
L_wrong_condition = mean(ReLU(1 + D(real, class, wrong_condition)))
```

总 D loss：

```text
L_D = L_hinge + 0.25 * L_wrong_class
              + 0.25 * L_wrong_condition
              + lazy_R1
```

R1 每 16 个 batch 触发一次，代码使用 `0.5 * r1_weight * r1_every * r1` 做稀疏项
校正；正式权重 `r1_weight=0.25`。D 的总 loss 在 hinge GAN 中不需要单调下降，约 2 附近
表示 real/fake 和错误条件约束处于可训练平衡，不能单独用它判断图像质量。

## 6. 一个 batch 的执行顺序和梯度路径

![训练流程图](visualizations/hifc_unpaired_final/workflow_overview.png)

### 6.1 取样

```text
JointROIDataset
    -> rgb, rgb_alt, real_roi
    -> class_id
    -> metadata, depression, azimuth
    -> condition_from_batch(metadata, depression)
```

### 6.2 D step

1. `encoder(rgb)` 和 `generator(...)` 在 `no_grad` 下得到 fake；
2. fake `detach()`；
3. D 计算真实配对、fake 配对、错误车型配对和错误条件配对；
4. 计算 hinge、wrong-class、wrong-condition 和按间隔触发的 R1；
5. 只更新 D 参数。

### 6.3 G/E step

1. 重新 forward RGB encoder 和 generator，确保 E/G 图仍有清晰梯度路径；
2. D 参数临时关闭梯度，但允许 `D(fake)` 对 fake 求导；
3. 计算 `L_adv` 和 `L_rgb_identity`；
4. 计算 fake/real 的 LTC 统计；
5. FP32 运行 frozen native teacher，计算 `L_sfm` 和 `L_geometry`；
6. 按权重合成 `L_G`；
7. 只更新 RGB encoder 和 generator，梯度范数裁剪到 5；
8. 用 decay `.999` 更新 EMA encoder/generator。

### 6.4 参数和数值设置

```text
E learning rate:       1.0e-4
G learning rate:       1.5e-4
D learning rate:       1.0e-4
E/G optimizer:         AdamW(betas=(0, .99), weight_decay=1e-4)
D optimizer:           Adam(betas=(0, .99))
gradient clipping:     5.0
EMA decay:             .999
AMP:                   enabled on CUDA
```

### 6.5 DDP

正式运行从单卡 epoch 16 的 checkpoint 续训，epoch 17–120 使用 8 卡 DDP：

```text
GPU count:              8 A100
per-rank batch:         8
effective global batch: 64
epoch_size:             24000
steps per epoch:        375
```

`DistributedSampler` 为每个 epoch 设置固定 seed；所有 rank 的 loss/计数先 all-reduce，
只有 rank 0 写 checkpoint、history 和 validation preview。保存的 DDP checkpoint 经过
`unwrap()`，不带 `module.` 前缀，可以用普通单卡脚本加载。

## 7. 与原 V1 的区别

这里的 HiFC 模型是一个新的独立入口，原 V1 文件没有被覆盖。概念变化如下：

| 部分 | 原 V1/历史实验 | 最终 HiFC unpaired |
|---|---|---|
| RGB/SAR 配对 | 同车型或方向弱配对，部分实验仍调用 translation alignment | 同车型随机视角，明确无像素配对 |
| RGB 身份 | `rgb_identity`、`cross_view` 等多条身份约束 | 合并为 `L_rgb_identity`（双视角 CE + cosine） |
| 车型 teacher | `sar_class`、cluster/prototype 等直接分类或簇约束 | 不使用 native class hard CE；只用 embedding SFM |
| 结构/统计 | pixel64/32/16、SSIM、physics、statistics、feature-match 等历史组合 | LTC 局部 residual/contrast/Haar moments + SFM feature moments |
| 感知项 | 部分 V1 实验使用 perceptual pyramid | 当前最终基线未接入 perceptual 链路 |
| 判别器 | 历史 PatchGAN 或实验性融合分类器 | 一个共享 conditional projection PatchGAN |
| 生成器 | 多个 V1 generator 变体 | 一阶段 alias-free SPADE-like decoder + learned speckle |
| 条件 | 可能含 source RGB angle、bbox 等输入 | 只含 target azimuth/dep/band/polarization 12D |
| 像素 loss | 某些 V1 路径有 pixel/translation 对齐 | 完全没有 RGB↔SAR pixel L1 |
| 主要评估 | native classifier、结构 loss、视觉 preview | 独立 TSTR 是真实泛化主指标 |

原 V1 中看起来相似的 loss 不能仅凭数值相近就删除；它们可能作用在不同模块、不同
特征层和不同梯度路径。当前 HiFC 是一次明确的架构重组，不是对旧 V1 做逐项因果消融，
因此结果比较必须注明 checkpoint、seed、数据 split 和 TSTR protocol。

## 8. 指标如何阅读

### 8.1 训练 history 中的 loss

| 字段 | 解释 |
|---|---|
| `generator` | 加权后的 `L_G` 总和 |
| `adversarial` | `-D(fake)`，对抗平衡指标，不要求单调下降 |
| `rgb_identity` | 两个 RGB 视角的身份 CE 和 cosine |
| `ltc` | 局部纹理统计 SmoothL1，数值较小是统计接近 |
| `sfm` | teacher embedding 和 D feature moments |
| `geometry` | 四个采集条件辅助 CE 的平均值 |
| `discriminator` | D aggregate，包括 hinge、错误条件和 R1 |
| `disc_wrong_class` | 真实图配错误车型时 D 的负样本项 |
| `disc_wrong_condition` | 真实图配错误采集条件时 D 的负样本项 |
| `r1` | 稀疏 R1 梯度正则 |

### 8.2 native teacher diagnostics

| 字段 | 解释 |
|---|---|
| `rgb_accuracy` | RGB encoder 在原视角和另一视角上的车型准确率 |
| `native_class_accuracy` | frozen native teacher 对 fake 的车型准确率 |
| `native_band_accuracy` | fake 的 X/KU 识别率 |
| `native_polarization_accuracy` | fake 的 HH/HV/VH/VV 识别率 |
| `native_depression_accuracy` | fake 的 15/30/45/60 识别率 |
| `native_azimuth_accuracy` | fake 的 12 个方位 bin 识别率 |
| `validation_*` | 在 held-out GAN validation records 上的相同诊断 |

native class 100% 可能只是生成图匹配了 teacher 的高频决策边界；只有 TSTR 中一个
没有看到真实 SAR 训练像素的全新分类器也能识别真实 SAR，才说明信息真正迁移。

## 9. 最终训练结果和分析

### 9.1 GAN history

最终 epoch 120：

```text
G total:                 1.89769
D total:                 1.92676
rgb_identity:            0.23937
LTC:                     0.0001268
SFM:                     0.30024
geometry:                0.26855
validation LTC:          0.0001275
validation SFM:          0.29820
validation geometry:     0.25645
validation native class: 1.00000
```

从单卡 epoch 16 到最终 epoch 120，validation SFM 从约 `0.3193` 降到 `0.2982`，
validation geometry 从约 `0.2645` 降到 `0.2565`。约 epoch 50 以后曲线基本平台，
geometry 有轻微回升，所以更长训练并不自动等于更好的真实泛化。

### 9.2 独立 TSTR

TSTR 流程是：

```text
final EMA encoder/generator
    -> 按 train condition records 生成 synthetic SAR
    -> 只用 synthetic SAR 训练新的 SARClassifier64 30 epochs
    -> 只在 held-out real X/HH TIFF 上测试
```

TSTR 分类器输入只有一通道 SAR 图像；band、polarization、depression、azimuth 只作为
训练标签的辅助 head，不作为输入条件。三个 seed 的最终文件分别是：

```text
code/runs/hifc_tstr_epoch120_classifier415/selected_metrics.json
code/runs/hifc_tstr_epoch120_classifier1618/selected_metrics.json
code/runs/hifc_tstr_epoch120_classifier31415/selected_metrics.json
```

真实测试 Top-1 按俯视角平均为：

| 俯视角 | Top-1 |
|---:|---:|
| 15° | 44.53% |
| 30° | 54.32% |
| 45° | 55.27% |
| 60° | 38.86% |

方位辅助结果为 Top-1 `60.92%`，circular MAE `42.49°`。60° 是当前最弱域，说明
俯视角泛化仍需要单独改进。

### 9.3 结论边界

当前结果可以支持：

- HiFC 无像素配对路径比旧 V1 的 TSTR 迁移明显更好；
- 生成图中确实包含一部分可迁移到真实 X/HH 的车型和几何信息；
- 生成器没有完全依赖 native teacher 的 class CE 才能工作；
- 三个 classifier seed 结果接近，提升不是单个随机初始化偶然造成的。

当前结果不能支持：

- native class accuracy 100% 就等于真实 SAR 车型信息完整；
- 48% Top-1 已经达到真实 SAR 分类器上限；
- 所有角度、极化和波段都已同样学好；
- 生成图已经是物理严格的 SAR 重建。

## 10. 可视化和文件说明

完整图表在 [`visualizations/hifc_unpaired_final`](visualizations/hifc_unpaired_final/README.md)。

| 文件 | 内容 |
|---|---|
| `workflow_overview.png` | 输入、E/G/D、teacher、loss 和梯度更新关系 |
| `training_curves.png` | 120 epoch 的 G/D、四个 G 组件、验证 loss、teacher diagnostics |
| `tstr_final_results.png` | 三个 TSTR seed 及四个俯视角的真实测试结果 |
| `validation_120.png` | RGB、真实 SAR、clean fake、observed fake 的最终 contact sheet |
| `validation_060.png`、`validation_090.png` | 中期视觉演化对照 |
| `history.csv` | 合并后的 epoch 1–16 单卡和 17–120 DDP 原始记录 |
| `config.json` | 最终 DDP 参数、条件布局、参数量和 loss 描述 |
| `tstr_summary.json` | 三 seed 汇总和旧 V1/epoch5 对照数字 |

## 11. 复现和渲染

### 11.1 单机 8 卡训练

```bash
cd /data/newdata/A25_T37_down_大图/code

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=8 --master_port=29680 \
  train_hifc_unpaired_sar_gan.py \
  --rgb-root /data/newdata/A25_T37_down_大图/A02/RGB \
  --sar-train-root /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train \
  --native-classifier-checkpoint server_results/sar_native64_multitask_v1/best.pt \
  --output runs/hifc_unpaired_all_conditions_ddp \
  --band all --polarization all --depression all \
  --epochs 120 --epoch-size 24000 --batch-size 8 --workers 2 \
  --generator-lr 0.00015 --identity-lr 0.0001 --discriminator-lr 0.0001 \
  --resume runs/hifc_unpaired_all_conditions/latest.pt \
  --device cuda:0
```

已经完成训练时不要重复运行这条命令；直接加载 `epoch_120.pt` 做渲染或 TSTR。

### 11.2 生成一个车型的方位角 sweep

```bash
python render_hifc_unpaired_sar.py \
  --gan-checkpoint runs/hifc_unpaired_all_conditions_ddp/epoch_120.pt \
  --rgb-root /data/newdata/A25_T37_down_大图/A02/RGB \
  --class-name Buick_GL8 --source-angle 0 \
  --depression 30 --band X --polarization HH \
  --output runs/Buick_GL8_azimuth_sweep.png \
  --device cuda:0
```

模型的方位编码是连续的，因此可以在代码层面传入非 30° 的浮点角度做插值；当前
渲染 CLI 默认输出 `0,30,...,330` 十二个角度。俯视角、波段和极化仍是训练数据中的
离散条件，未观测角度的细节不能被视为物理保证。

### 11.3 重跑 TSTR

```bash
for seed in 415 1618 31415; do
  python train_generated_sar_classifier_64.py \
    --gan-checkpoint runs/hifc_unpaired_all_conditions_ddp/epoch_120.pt \
    --gan-weights ema \
    --rgb-root /data/newdata/A25_T37_down_大图/A02/RGB \
    --condition-root /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train \
    --real-test-root /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/test \
    --output runs/hifc_tstr_epoch120_classifier${seed} \
    --epochs 30 --batch-size 128 --workers 4 \
    --seed ${seed} --checkpoint-selection final --device cuda:0
done
```

## 12. 当前下一步

当前最有价值的改进不是继续盲目增加 loss，而是保持这个 TSTR 评估协议，针对已暴露的
问题做单变量实验：

1. 单独改善 60° 俯视角的采样和条件覆盖；
2. 检查 TSTR classifier 的 confusion matrix，区分车型相似度和生成伪纹理；
3. 对 final generator 做多次 noise 渲染，确认分类信息不是固定噪声模式；
4. 增加独立真实 SAR feature/频谱审计，而不是继续依赖同一个 native teacher；
5. 任意 loss 合并或删减都要保持梯度量级可比，并以三 seed TSTR 作为主判据。
