# HiFC Epoch 120 逐层架构、张量与梯度说明

本文对应 `hifc_unpaired_all_conditions_ddp/epoch_120.pt`。它描述的是原始 120 epoch
基线：native SAR teacher 参数冻结，但 teacher embedding 和 geometry 辅助头对生成图的
输入梯度均开启。它不是后续 `native-gradient-mode=all_off` 消融版本。

## 1. 每个 batch 的输入

每个 GPU 的 batch 为 `B=8`，8 卡 DDP 的有效全局 batch 为 64。图像全部归一化到
`[-1,1]`。

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `rgb` | `[B,3,128,128]` | 同车型的随机 RGB 侧视图 |
| `rgb_alt` | `[B,3,128,128]` | 同车型的另一张 RGB 视图 |
| `real` | `[B,1,64,64]` | 真实单通道 SAR ROI |
| `y` | `[B]` | 40 类车型编号 |
| `epsilon` | `[B,1,64,64]` | 无参数的空间随机噪声 |
| `c` | `[B,12]` | 目标 SAR 条件 |

`rgb` 和 `rgb_alt` 都 resize 到 128x128、转成 RGB 三通道，并施加轻微 gain、bias、噪声
增强。`real` resize 到 64x64、转成灰度单通道。RGB 和 SAR 仅按车型弱匹配，不存在 RGB/SAR
同坐标像素配对。

```text
c = [sin(azimuth), cos(azimuth),
     onehot(depression: 15/30/45/60),
     onehot(band: X/KU),
     onehot(polarization: HH/HV/VH/VV)] = [B,12]
```

源 RGB 拍摄角度和 SAR bbox 尺寸不输入生成器或判别器。

## 2. RGB 身份编码器 E

模块：`LargeRGBIdentityEncoder`。每个 stage 均为：

```text
Conv4x4 stride2 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU
```

```text
rgb [B,3,128,128]
  -> p0 [B,64,64,64]
  -> p1 [B,128,32,32]
  -> p2 [B,256,16,16]
  -> p3 [B,512,8,8]
  -> AdaptiveAvgPool -> Flatten -> Linear(512,512) -> LayerNorm -> SiLU
  -> z [B,512]
  -> Linear(512,40) -> l_rgb [B,40]
```

`z` 是车辆身份的全局摘要；`p0..p3` 是保留空间位置的多尺度 RGB feature map。`rgb_alt`
经过相同且共享权重的 E，得到 `z_alt [B,512]`、`l_rgb_alt [B,40]`。只有第一张 `rgb`
的 `z,p0,p1,p2,p3` 进入生成器；`rgb_alt` 只参与 RGB 身份 loss。

## 3. 生成器 G 的全局起点

模块：`HIFCUnpairedGenerator`，实际继承 `OneStageWaveletSARGenerator`。

```text
c [B,12]
  -> Linear(12,256) -> SiLU -> Linear(256,256) -> SiLU
  -> g(c) [B,256]

concat(z,g(c)) [B,768]
  -> Linear(768,512*4*4)
  -> reshape
  -> h0 [B,512,4,4]
```

`h0` 是 decoder 的初始隐藏 feature，不是 SAR 图像。它包含全局车型信息和目标 SAR
条件，但没有高分辨率空间细节。

## 4. 四层 RGB pyramid 全部如何使用

| decoder block | decoder 输入 | decoder 输出 | 使用的 RGB pyramid |
| --- | --- | --- | --- |
| `block0` | `h0 [B,512,4,4]` | `h1 [B,512,8,8]` | `p3 [B,512,8,8]` |
| `block1` | `h1 [B,512,8,8]` | `h2 [B,256,16,16]` | `p2 [B,256,16,16]` |
| `block2` | `h2 [B,256,16,16]` | `h3 [B,128,32,32]` | `p1 [B,128,32,32]` |
| `block3` | `h3 [B,128,32,32]` | `h4 [B,64,64,64]` | `p0 [B,64,64,64]` |

因此 `p1` 控制 32x32 SAR 解码层，`p0` 控制最终 64x64 SAR 解码层；二者没有被跳过。
最终 `h=h4 [B,64,64,64]` 是 forward 返回的 `final_feature`，同时供 clean head 和
speckle branch 使用。

### 4.1 一个 `AliasFreeSPADEBlock` 内部

当前 HiFC **不执行** `decoder_feature + Conv1x1(p_i)`。这属于历史
`DenoisedSARGenerator`，不是本 checkpoint 的实现。

当前 block 的 decoder residual 结构为：

```text
x -> bilinear upsample -> fixed [1,2,1] blur -> x_up
  -> 1x1 Conv ---------------------------------> shortcut s

x_up -> RGB spatial modulation -> Conv3x3 -> GN -> SiLU -> Conv3x3 -> main m
out = SiLU(m + s)
```

RGB spatial modulation 的精确路径是：

```text
p_i -> depthwise Conv3x3 -> SiLU -> Conv1x1(输出 2*C 通道)
    -> split(gamma [B,C,H,W], beta [B,C,H,W])

modulated(x_up) = GroupNorm(x_up) * [1 + 0.25*tanh(gamma)] + beta
```

`gamma` 和 `beta` 是逐位置、逐通道的空间图，不是单个数：

- `gamma` 决定某个位置、某个 feature channel 应增强还是减弱。
- `beta` 决定归一化后该位置、该 channel 应整体抬高还是降低。
- RGB pyramid 不直接变成 SAR feature，而是告诉 decoder 在对应位置应更倾向生成什么 SAR
  结构。
- `shortcut s` 只来自 decoder 自己的 `x_up`，不是 RGB shortcut。

## 5. clean SAR head 与 h 的含义

```text
h [B,64,64,64]
  -> Conv3x3(64->64) -> SiLU -> Conv3x3(64->1) -> tanh
  -> clean = fake_clean [B,1,64,64]
```

`clean` 是没有随机散斑后的反射率/幅度底图。它没有单独的像素级 loss；后续会经过可微
speckle renderer 形成 `fake`，所有 SAR loss 通过 `fake` 回传给 clean head 和 `h`。

## 6. Learned speckle branch

这不是第二个独立 generator。它与 clean head 共用 `h`，一次 forward 同时输出
`clean`、`log_noise` 和 `fake observed SAR`。

### 6.1 随机场

```text
epsilon [B,1,64,64]
correlated = AvgPool3x3(epsilon)
random_field = 0.70*epsilon + 0.30*correlated
```

`epsilon` 没有可训练参数，只让同一车、同一条件可以生成不同散斑细节。`correlated` 使其
不完全是逐像素白噪声。

### 6.2 学习 scale 与 bias

```text
concat(h, clean, correlated) = [B,66,64,64]
  -> Conv3x3(66->64) -> SiLU -> Conv3x3(64->64) -> SiLU
  -> noise_feature [B,64,64,64]

scale = 0.04 + 0.38*sigmoid(Conv3x3(noise_feature))
bias  = 0.12*tanh(Conv3x3(noise_feature))
```

`scale` 的范围是 `(0.04,0.42)`，`bias` 范围约为 `[-0.12,0.12]`；两者都是
`[B,1,64,64]` 的位置相关图。

```text
raw_log_noise = scale * random_field + bias
log_noise = raw_log_noise - spatial_mean(raw_log_noise)
log_noise = clamp(log_noise,-0.8,0.8)
```

去除每张图的空间均值，防止噪声分支只通过整体变亮或变暗欺骗判别器。

### 6.3 可微观测 SAR

```text
clean_amplitude = clamp((clean+1)/2,1e-4,1)
observed_amplitude = clamp(clean_amplitude * exp(log_noise),0,1)
fake = 2*observed_amplitude - 1
```

`fake [B,1,64,64]` 是 D、LTC、native teacher、geometry loss 的监督对象。反传时：

```text
fake -> clean -> clean head -> h -> four decoder blocks -> z,p0..p3 -> E
fake -> log_noise -> scale/bias heads -> noise_feature -> h -> G/E
```

`epsilon` 会有数学梯度，但它不是模型参数，不会被优化器更新。

## 7. D、teacher、loss 与反传

条件判别器 D 将图像编码为 `f_D [B,512,4,4]`，并把车型 embedding 与 12D 条件 MLP
embedding 投影到 feature map 上，输出 `D(fake,y,c) [B]`。D step 使用 `fake.detach()`，
只更新 D；G/E step 冻结 D 参数但保留从 `D(fake)` 到 fake 的输入梯度。

native teacher T 将 `[B,1,64,64]` SAR 映射为 `f_T [B,384]`，另有 40 类、band 2 类、
polarization 4 类、depression 4 类、azimuth 12 类输出头。T 参数被冻结；原始 epoch120
基线允许 fake 经 T 的 embedding 与辅助头回传梯度。T 的 40 类 logits 只记录 accuracy，
不作为 hard class CE loss。

```text
L_EG = L_adv + 1.0*L_rgb + 2.0*L_ltc + 2.0*L_sfm + 0.3*L_geometry
```

| loss | 更新路径 | 被 detach/frozen 的部分 |
| --- | --- | --- |
| `L_rgb` | 两张 RGB -> E | G 不参与 |
| `L_adv` | D 输入梯度 -> fake -> G -> E | D 参数冻结 |
| `L_ltc` | fake -> G -> E | real 统计 detach |
| `L_sfm` | T/D feature -> fake -> G -> E | T/D 参数、real feature detach |
| `L_geometry` | T aux heads -> fake -> G -> E | T 参数冻结 |

`L_rgb` 使用双视角 label-smoothed CE 和 `1-cos(z,z_alt)`；`L_ltc` 匹配 residual、
contrast、Haar 的 batch moments，无像素 L1；`L_sfm` 匹配 teacher embedding cosine、
batch mean 和 D feature moments；`L_geometry` 是 band、polarization、depression、azimuth
四个 CE 的平均。首 epoch 的 `L_adv` 权重为 0，之后为 1。

总 loss 反传后，E 与 G 参数一起做 `clip_grad_norm(...,5.0)`，再由 AdamW 更新；随后
EMA 用 `theta_ema = 0.999*theta_ema + 0.001*theta` 更新。

## 8. 画图时的三个关键结论

1. `p0..p3` 全部使用，但它们生成 `gamma/beta` 调制 decoder，不直接相加到 decoder。
2. `h [B,64,64,64]` 是最后 decoder feature，同时进入 clean head 和 speckle branch。
3. `clean` 是中间反射率图，`fake observed SAR` 才是 D、LTC、teacher、geometry 的监督图。

实现入口：`code/dual_component_sar_gan.py`、`code/one_stage_wavelet_sar_gan.py`、
`code/hifc_unpaired_sar_gan.py`、`code/train_hifc_unpaired_sar_gan.py`。
