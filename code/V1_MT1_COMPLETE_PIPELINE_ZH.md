# V1 / MT1 RGB 到 SAR 完整流程与画图规范

本文以当前代码中唯一被接受的 MT1 配置为准，完整描述数据、网络、loss、训练顺序、梯度归属、与原始 V1 的差异，以及已经关闭的实验分支。

这份文档服务于画模型流程图。图中必须区分：

1. 当前训练真正启用的模块和 loss；
2. 代码中存在、但当前权重为 0 或没有打开的可选分支；
3. 历史 Fused V2 和 MT2--MT6 实验，它们不是当前流程。

## 1. 版本定义

### 1.1 V1 控制模型

V1 的模型检查点架构标签是 continuous_spatial_v1，由三部分组成：

~~~text
RGBIdentityEncoder -> SpatialROIGenerator -> ContinuousROIDiscriminator
~~~

其中判别器是带条件投影的 PatchGAN；SAR 分类器是一个独立的冻结 teacher，不是 V1 判别器的一部分。

V1 的 ablation 入口是 train_continuous_spatial_v1_ablation.py。它保留了 V1 的网络和 loss，并把每个实验参数暴露出来。

原始 ablation 入口的 parser 默认 sar_class_weight=12。先前单变量实验已经证明把它降到 1 更适合真实域迁移，因此推荐 V1 和 MT1 都显式使用：

~~~text
sar_class_weight = 1
cluster_weight    = 5
~~~

这不是 MT1 中动态改变的 loss，而是 MT1 所继承的推荐 V1 基线配置。

### 1.2 当前 MT1

当前 MT1 = 推荐 V1 + 一个只向 generator 注入的 meta-transfer 超梯度：

~~~text
V1 的 E/G/D 和普通 loss
        +
synthetic support -> 虚拟分类头一步更新 -> real X/HH query CE
        -> 只回传 generator 的 hypergradient
~~~

当前 MT1 不使用：

- K+1 融合分类判别器；
- 非对齐 distributional structure；
- 删除 64x64 像素项；
- perceptual loss；
- MT2-AZ、MT3-XDEP、MT4-ZH、MT5-EP2、MT6-U32。

## 2. 数据和配对关系

### 2.1 当前数据范围

当前训练固定为：

~~~text
类别：SOC40 的 40 个车型
SAR：X 波段、HH 极化
俯视角 depression：15、30、45、60 度
目标方位角 azimuth：由 XML 记录提供，可连续取值
RGB：每个车型的多视角 PNG（通常 0、30、...、330 度）
~~~

默认路径：

~~~text
RGB: /data/newdata/A25_T37_down_大图/A02/RGB
SAR: /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train
~~~

数据扫描和返回字段在 joint_data.py 的 JointROIDataset 中实现。

### 2.2 一个训练样本

每个 SAR XML/TIFF 记录包含：

~~~text
(sar_tif, nearest_rgb_path, class_name, bbox, meta, nearest_rgb_angle)
~~~

__getitem__ 返回：

| 字段 | 形状/类型 | 用途 |
|---|---|---|
| rgb | 3 x 128 x 128，[-1, 1] | 当前 source RGB 视图 |
| rgb_alt | 3 x 128 x 128，[-1, 1] | 同一车型的另一 RGB 视图 |
| roi | 1 x 64 x 64，[-1, 1] | 当前真实 SAR ROI |
| meta | 10 维 float | SAR 条件和 bbox 元数据 |
| class_id | 0--39 | 车型标签 |
| azimuth | 整数角度 | 目标 SAR 方位角 |
| depression | 15/30/45/60 | 目标 SAR 俯视角 |
| rgb_angle | 整数角度 | 当前 source RGB 方位角 |

RGB 预处理：双线性缩放到 128；随机 gain、bias 和小幅噪声；归一化到 [-1, 1]。source_view_mode=mixed 时，约一半使用最接近 SAR 方位的 RGB 视图，另一半随机使用该车型的其他视图。

真实 SAR ROI 已来自 SOC_40classes_cut，因此默认 pre_cropped=True，不再用 bbox 对 TIFF 二次裁剪；灰度缩放到 64x64，再归一化到 [-1, 1]。

### 2.3 元数据向量和生成条件

metadata_vector(meta, bbox) 先构造 10 维向量：

~~~text
[ sin(az_target), cos(az_target), depression/60,
  band_is_X,
  pol_HH, pol_HV, pol_VH, pol_VV,
  bbox_width/128, bbox_height/128 ]
~~~

target_condition(meta, source_rgb_angle) 将 bbox 宽高置零，再拼上 source RGB 方位的正余弦：

~~~text
c = [ sin(az_target), cos(az_target), depression/60,
      band_is_X, pol_HH, pol_HV, pol_VH, pol_VV,
      0, 0,
      sin(az_source), cos(az_source) ]       # 12 维
~~~

因此 generator 同时看到目标 SAR 的方位/俯视角、波段/极化，以及 source RGB 的观察方位；不会看到真实 SAR 的像素或真实 SAR bbox 尺寸。

### 2.4 弱配对，不是像素配对

真实 SAR 和 RGB 只按车型、条件和近似方位建立弱配对。它们不是同一时刻、同一投影、同一像素坐标，因此不能把真实 SAR 当作 RGB 的严格像素重建目标。

但当前 V1 的 structure 和 physics 内部仍会对真实 SAR 做一个小范围的离散平移选择，再比较像素/散射项。这是允许小位移的弱对齐，不是固定坐标的严格配准；它在当前 MT1 中仍然存在。

### 2.5 本机读取核验

本次在训练机上对上述目录做了只读检查：RGB 目录存在 40 个车型子目录；SAR 目录存在 12,157 个 `X_HH_*.tif`，且每个都有对应 XML。随机样本可正常打开（RGB 为 RGBA、原始尺寸 4000x3000；SAR 为灰度 L、示例尺寸 54x54）。当前 MT1 的固定 split 使用其中全部 12,157 条 X/HH 记录：

~~~text
train split:       10,190 records
validation split:   1,967 records
visual proxy:         640 records
~~~

active trainer 的 `pre_cropped=True` 会把这些已裁剪 SAR 直接缩放到 64x64；不会再按 XML bbox 对 TIFF 做第二次裁剪。因此数据入口、XML 标签、PNG/TIFF 解码和 64x64 tensor 化均已验证可用。

## 3. 当前网络架构

### 3.1 RGBIdentityEncoder

实现：joint_models.py 中的 RGBIdentityEncoder。

输入是 B x 3 x 128 x 128。四个下采样 stage 均为：

~~~text
Conv2d(kernel=4, stride=2, padding=1)
GroupNorm
SiLU
~~~

通道和空间尺寸：

| stage | 输出尺寸 | 输出通道 |
|---|---:|---:|
| 1 | 64 x 64 | 32 |
| 2 | 32 x 32 | 64 |
| 3 | 16 x 16 | 128 |
| 4 | 8 x 8 | 256 |

得到四层 RGB pyramid：

~~~text
pyramid = (P64, P32, P16, P8)
~~~

最后对 P8 做全局平均池化，再经过：

~~~text
Linear(256 -> 256) -> LayerNorm(256) -> SiLU
~~~

输出：

~~~text
identity z: B x 256
RGB class logits: B x 40
RGB pyramid: P64/P32/P16/P8
~~~

RGB 分类头是 CosFace 风格的余弦头：

~~~text
q = normalize(z)
w_k = normalize(W_k)
logit_k = 20 * (q · w_k)
~~~

这个模块支持带标签时对正确类别减去 margin m=0.15；但是当前 V1/MT1 的 encoder forward 没有把 labels 传进 class_logits，因此当前实际训练的 rgb_identity 使用的是不加 margin 的 scaled cosine logits。bias 参数为了兼容旧 checkpoint 保留，但 CosineClassifier forward 不使用该 bias。

rgb_alt 经过同一个 encoder，权重完全共享。

### 3.2 SpatialROIGenerator

实现：joint_models.py 中的 SpatialROIGenerator。

输入：

~~~text
identity z: B x 256
condition c: B x 12
pyramid: P64/P32/P16/P8
~~~

条件支路：

~~~text
Linear(12 -> 128) -> SiLU -> Linear(128 -> 128) -> SiLU
~~~

将 z 与条件 embedding 拼接成 384 维，再经过：

~~~text
Linear(384 -> 256*4*4) -> reshape 为 B x 256 x 4 x 4
~~~

解码器有四个上采样块，每个块为：

~~~text
双线性上采样 x2
Conv3x3 -> GroupNorm -> SiLU
Conv3x3 -> GroupNorm -> SiLU
~~~

四个块的主分支通道：

~~~text
256 -> 128 -> 64 -> 32 -> 16
~~~

每个解码尺度都加入对应的 RGB pyramid 投影：

~~~text
P8  -> 第 1 个 8x8 解码层
P16 -> 第 2 个 16x16 解码层
P32 -> 第 3 个 32x32 解码层
P64 -> 第 4 个 64x64 解码层
~~~

投影使用 1x1 convolution，最后：

~~~text
Conv3x3(16 -> 1) -> Tanh
~~~

得到 clean SAR：

~~~text
fake_clean: B x 1 x 64 x 64，[-1, 1]
~~~

### 3.3 可微 SAR speckle 观测模型

训练后期将 clean 输出转换到 [0, 1]，加入：

1. 白噪声与 3x3 平均池化相关噪声的混合；
2. 对数乘性 speckle，默认最大强度 0.32；
3. 低频 gain、gamma 和照明场变化；
4. 正接收机噪声 floor 和 Rayleigh 项；
5. 少量零均值 receiver noise；
6. 截断到 [0, 1]，再转换回 [-1, 1]。

训练策略是前 8 个 epoch speckle warmup，随后 5 个 epoch 线性 ramp 到最大强度。当前从 epoch 100 继续训练时已经处于满 speckle 强度。

因此生成器有两个重要输出：

~~~text
clean  -> structure / physics / angle
fake   -> discriminator / statistics / native teacher / feature match
~~~

### 3.4 V1 Conditional Projection PatchGAN

实现：joint_models.py 中的 ContinuousROIDiscriminator。

输入：

~~~text
SAR image: B x 1 x 64 x 64
condition: 默认完整 12 维 c
~~~

四个 spectral-normalized convolution block：

~~~text
Conv4x4, stride=2 + LeakyReLU(0.2)
~~~

通道/空间尺寸：

| block | 空间 | 通道 |
|---|---:|---:|
| 1 | 32x32 | 32 |
| 2 | 16x16 | 64 |
| 3 | 8x8 | 128 |
| 4 | 4x4 | 256 |

条件投影支路：

~~~text
Linear(12 -> 256) -> SiLU -> Linear(256 -> 256)
~~~

将条件向量映射成 256 个通道的 projection，与 feature map 逐通道相乘后求和，加入一个 spectral-normalized Conv3x3(256 -> 1) 分数图。

输出：

~~~text
score map: B x 16      # 4x4 PatchGAN 展平
D features: B x 256 x 4 x 4
~~~

代码还包含一个 40 类 auxiliary classifier head，但当前 MT1 中：

~~~text
discriminator_class_mode = disabled
discriminator_class_weight = 0
generator_discriminator_class_weight = 0
~~~

所以该 head 不参与训练，当前 D 仍然是原 V1 PatchGAN，不是分类器判别器。

### 3.5 冻结的 native SAR classifier teacher

实现：sar_classifier_64.py 中的 SARClassifier64，checkpoint：

~~~text
server_results/sar_native64_multitask_v1/best.pt
~~~

它只输入一通道 64x64 SAR 强度图，不把波段、极化、俯视角、方位角作为输入；这些只是训练标签。

结构：

~~~text
Stem: Conv3x3 1->48 + GroupNorm + SiLU
Stage 1: 48 channels, 2 residual blocks, 1x downsample
Stage 2: 96 channels, 2 residual blocks, 2x downsample
Stage 3: 192 channels, 3 residual blocks, 2x downsample
Stage 4: 384 channels, 2 residual blocks, 2x downsample
AdaptiveAvgPool -> LayerNorm -> Dropout(0.15)
~~~

输出 384 维 SAR feature，并有多个 head：

~~~text
class head:        40 类
band head:          2 类
polarization head:  4 类
depression head:    4 类
azimuth head:      12 个 30 度 bin
~~~

在 MT1 中它被冻结，只用于：

- sar_class 的 native class CE；
- cluster 的 class/depression 条件 feature prototype；
- 可选但当前关闭的 perceptual、margin、geometry auxiliary 等项；
- 日志和诊断，不作为 TSTR 的最终质量标准。

### 3.6 Real feature prototype

使用冻结 native classifier 在训练集真实 SAR 上计算：

~~~text
prototype[class_id, depression_id] in R^384
~~~

形状为 40 x 4 x 384，每个 prototype 是对应车型/俯视角的归一化平均 feature，并缓存到 native_conditional_prototypes.pt。

## 4. 当前训练总流程图

下面的 Mermaid 图可以直接作为总图草稿。实线表示 forward；虚线表示梯度方向；灰色模块是当前存在但关闭的可选分支。

~~~mermaid
flowchart TD
    R[真实 SAR ROI<br/>1x64x64, X/HH] --> D0[Conditional Projection PatchGAN]
    R --> T[冻结 native SAR classifier<br/>40-class + 384-d feature]
    R --> Q[MT1 real query pool<br/>仅在 meta 事件使用]

    RGB[RGB source view<br/>3x128x128] --> E[RGBIdentityEncoder<br/>4-stage CNN]
    RGB2[同车型 RGB alternate view] --> E2[共享 RGB encoder]
    E --> Z[z 256-d]
    E --> P[P64/P32/P16/P8]
    E2 --> Z2[z_alt 256-d]
    C[12-d target condition<br/>SAR target az/dep + X/HH + source RGB az] --> G[SpatialROIGenerator<br/>FC + 4 upsample blocks + RGB FPN]
    Z --> G
    P --> G
    G --> CLEAN[fake_clean<br/>64x64, no speckle]
    CLEAN --> S[可微 speckle model]
    S --> FAKE[fake observed SAR<br/>64x64]

    FAKE --> D0
    FAKE --> T
    T --> TC[sar_class CE]
    T --> CP[40x4 real feature prototypes]
    CP --> CL[cluster cosine pull]
    CLEAN --> ST[structure<br/>64/32/16 pixel + edge + global SSIM]
    CLEAN --> PH[physics<br/>log amplitude + scattering map + correlation]
    CLEAN --> AN[angle smoothness<br/>0 vs +5 degree]
    FAKE --> RS[statistics<br/>intensity/edge moments]
    D0 --> ADV[G adversarial score]
    D0 --> FM[feature-match<br/>D feature mean/std]

    Z --> RI[rgb_identity CE]
    Z2 --> RI
    Z --> CV[cross_view cosine]
    Z2 --> CV

    RI -. E only .-> E
    CV -. E only .-> E
    TC -. coupled V1 route .-> G
    TC -. coupled V1 route .-> E
    CL -. coupled V1 route .-> G
    CL -. coupled V1 route .-> E
    ST -.-> G
    ST -.-> E
    PH -.-> G
    PH -.-> E
    AN -.-> G
    AN -.-> E
    RS -.-> G
    ADV -. coupled .-> G
    ADV -. coupled .-> E
    FM -. coupled .-> G
    FM -. coupled .-> E

    FAKE --> MS[MT1 synthetic support]
    MS --> VP[冻结 synthetic-only probe backbone]
    VP --> VH[复制 probe class head<br/>一步 support CE SGD]
    Q --> QP[同一个 probe backbone<br/>real query features detach]
    VH --> MQ[updated real-query CE]
    QP --> MQ
    MQ -. hypergradient 只注入 G .-> G

    OD[可选 D class head / wrong-az negative] -. 当前关闭 .-> D0
    OP[可选 perceptual / margin / geometry aux / feature spread] -. 当前关闭 .-> G
~~~

## 5. Loss 完整定义

定义：

~~~text
y       = 车型标签，40 类
z,z_alt = RGB encoder 的两个 256 维 identity
clean   = 无 speckle 生成图
fake    = 加 speckle 生成图
real    = 真实 SAR ROI
~~~

普通 V1 总体形式是：

~~~text
L_encoder = 10 * L_rgb_identity + 2 * L_cross_view

L_generator =
    1 * L_sar_class
  + 5 * L_cluster
  +20 * L_structure
  + 5 * L_statistics
  + 3 * L_physics
  + .2 * L_angle
  + 2 * L_adversarial
  + 5 * L_feature_match
  + optional terms with weight 0 by default

L_V1_normal = L_encoder + L_generator
~~~

当前 MT1 在此基础上增加一个独立的 meta 梯度，但该项不进入普通 LOSS_NAMES 或 total_loss 数值和：

~~~text
gradient_G <- gradient_G(L_V1_normal)
             + 2.064245482476472 * gradient_G(L_MT1)
~~~

### 5.1 Loss 表

| 名称 | 当前权重 | 精确定义/输入 | 当前状态和梯度 |
|---|---:|---|---|
| rgb_identity | 10 | 0.5 * [CE_ls(E(rgb), y) + CE_ls(E(rgb_alt), y)]；当前使用 scale=20 的归一化 cosine logits（实际路径不加 CosFace margin），V1 label smoothing=.02 | 启用；只更新 RGB encoder |
| cross_view | 2 | 1 - mean(cosine(z, z_alt)) | 启用；只更新 RGB encoder |
| sar_class | 1 | 冻结 native teacher 对 fake 的 40 类 CE；当前 native_head，label smoothing=.02 | 启用；默认 coupled 时更新 E+G，teacher 不更新 |
| cluster | 5 | 1 - cosine(normalize(f_fake), prototype[class,depression]) | 启用；默认 coupled 时更新 E+G，prototype detach |
| structure | 20 | 对 clean 和经过最佳小平移选择的 real 计算 64/32/16 L1、Sobel edge L1、global SSIM | 启用；默认 coupled 时更新 E+G |
| statistics | 5 | 对 fake/real 的强度 mean/std 和 Sobel magnitude mean/std 做 L1 | 启用；不做像素格点对应 |
| physics | 3 | 对 clean/平移后 real 的 log amplitude moments、正散射中心 map、多尺度、局部相关做约束 | 启用；仍含弱平移和位置敏感 scattering map |
| perceptual | 0 | native teacher feature pyramid：归一化 content L1 + channel moments，stage 权重 1,.75,.5,.25 | 代码存在；当前关闭，不进入 MT1 |
| angle | .2 | 默认 L1(avgpool4(G(c)), avgpool4(G(c+5°)))；可选 curvature 模式 | 启用 V1 first-order；更新 E+G |
| adversarial | 2 | -mean(D(fake,c)) | 启用；G 步冻结 D 参数但保留 fake 输入梯度，默认 coupled 时更新 E+G |
| feature_match | 5 | fake/real D feature map 的 spatial mean 和 std 的 L1 | 启用；real feature detach，默认 coupled 时更新 E+G |
| feature_spread | 0 | 相对 class/depression prototype 的 tangent residual mean/std | 代码存在；当前关闭 |
| sar_margin_ceiling | 0 | ReLU(m_fake - q90_real)，m=logit_y-max_wrong(logit)，真实参考 q90=6.73144 | 代码存在；当前关闭 |
| native_geometry_aux | 0 | native teacher 的 depression CE 与 azimuth-bin CE 平均 | 代码存在；当前关闭 |

### 5.1.1 代码中可替换的 loss 模式

下面是为了单变量 ablation 保留的模式，不是当前 MT1 同时使用的组合：

| 参数 | 当前模式 | 其他模式 |
|---|---|---|
| `rgb_loss_mode` | `separate`：分别记录并加权 `10*rgb_identity + 2*cross_view` | `joint_equivalent`：仅把同一个加权和写成一个 bookkeeping 项，不改变数学目标 |
| `sar_class_loss_mode` | `native_head`：native teacher 的 40 类 CE | `real_prototype`：真实 prototype 余弦 logits；`real_logit_direction`：生成/真实 centered native logits 方向余弦 |
| `cluster_loss_mode` | `v1`：`1-cosine` | `real_radius`：真实 validation 半径 hinge；`soft_real_radius`：半强度 V1 + 半强度 hinge |
| `local_texture_loss` | `v1_statistics`：全局强度/edge moments | `haar`：Haar detail moments，替换 statistics，不叠加 |
| `angle_loss_mode` | `first_order`：0 和 +5 度输出接近 | `curvature`：-5、0、+5 度二阶平滑 |
| `sar_class_teacher_mode` | `primary`：native teacher | `replacement`：仅 class logits 换成 FGSM 鲁棒 teacher |
| `discriminator_condition` | `full`：完整 12 维条件 | `target`：只给 D 目标 azimuth/depression 三维 |

如果把这些模式画进“实验树”，应画成从 V1/MT1 主干分出的灰色单变量候选，而不是画成同时生效的 loss。

代码还保留两种 teacher-side 单变量实验，它们当前都关闭：

- `sar_class_teacher_mode=replacement`：只把 `sar_class` 的 class logits 换成独立的 FGSM 鲁棒 native teacher（`epsilon=1/255`）；cluster 仍使用 primary teacher。
- `secondary_native_classifier_checkpoint`：加载第二个冻结的 SARClassifier64，`sar_class` 改为 primary/secondary 两个 class CE 的 0.5/0.5 平均；cluster 仍来自 primary teacher，且可用 `secondary_teacher_active_epochs` 限定前若干 epoch。

这两项都不是新增网络的当前路径，也不能和 MT1 一起画成同时生效的目标。

### 5.2 rgb_identity 和 cross_view 为什么仍然分开

两项都约束 RGB identity，但优化对象不同：

~~~text
rgb_identity：要求 z 能被 40 类分类头识别，直接更新 classifier head 和 encoder 表征。
cross_view：不经过分类头，只要求同一车型两视图的 z 方向相近，更新共享 trunk/embedding。
~~~

因此即便数值相关，也不是同一梯度。等权梯度审计的平均 cosine 只有约 0.1525；按当前权重 10/2 组合后，原始和候选向量 cosine 约 0.5986，相对梯度差约 0.8108。当前 joint_equivalent 只是把加法写在一起，数学上仍保留两项，不能解释为删除了其中一项。

### 5.3 structure 的精确定义

weighted_aligned_structure_loss(clean, real) 首先调用 _align_translation：

1. 对真实 ROI 枚举 dy,dx ∈ {-4,-2,0,2,4}，共 25 个候选平移；
2. 用 avg_pool2d(...,4) 后的平均绝对误差选择每张图的最佳候选；
3. 选择过程在 no_grad 中，后续 loss 仍对 clean 可微。

选中的 real_aligned 上计算：

~~~text
pixel_64 = L1(clean, real_aligned)
pixel_32 = L1(avgpool2(clean,2), avgpool2(real_aligned,2))
pixel_16 = L1(avgpool2(clean,4), avgpool2(real_aligned,4))
pixel_part = (1*pixel_64 + .5*pixel_32 + .25*pixel_16) / 1.75

edge = L1(SobelMagnitude(clean), SobelMagnitude(real_aligned))
ssim = global_SSIM_loss(clean, real_aligned)

L_structure = pixel_part + .5*edge + 1*ssim
~~~

当前 MT1 的 structure_pixel_64_weight=1，所以 64x64 项没有删除。

### 5.4 statistics 的精确定义

对 x=(image+1)/2：

~~~text
intensity = L1(mean(x), std(x))
edge      = L1(mean(|Sobel(x)|), std(|Sobel(x)|))
L_statistics = intensity + edge
~~~

这里比较的是每张图的统计量，不比较相同 (h,w) 位置。

### 5.5 physics 的精确定义

同样先选择最佳小平移，然后令 a=(image+1)/2、log_a=log(clamp(a,1e-4))：

~~~text
amplitude = L1(mean(log_a_fake), std(log_a_fake))

scatter_a = ReLU(a - AvgPool9(a))
scatter_a 在 1x/2x/4x 三个尺度归一化后做 L1，权重 1/.5/.25

correlation = log_a 在 (0,1)、(1,0)、(1,1) 位移上的局部相关系数 L1
L_physics = amplitude + scatter + correlation
~~~

它不是纯统计项，因为 scatter 对位置有约束；也不是普通 reconstruction，因为目标先允许小范围平移。

### 5.6 判别器 loss

当前 D 每两批更新一次，使用 hinge：

~~~text
L_D_real = mean(ReLU(1 - D(real, c)))
L_D_fake = mean(ReLU(1 + D(stopgrad(fake), c)))
L_D      = L_D_real + L_D_fake
~~~

G 的 adversarial 项为：

~~~text
L_adversarial = -mean(D(fake, c))
~~~

当前 wrong_azimuth_discriminator_weight=0，所以没有错误方位负样本；当前 D 也没有启用 class CE。

代码中的可选 D 分类分支不是当前 MT1 流程的一部分。打开 discriminator_class_mode=real_only 时，D 会从同一个 4x4 feature map 的 spatial mean（或 mean+max）预测 40 类，并增加：

~~~text
L_D_class = CE(D_class(real,c), y)                 # 只用真实 SAR 训练 D 分类头
L_D_total = L_D + discriminator_class_weight*L_D_class
L_G_class = CE(D_class(fake,c), y)                 # 只有显式 G 权重时才启用
~~~

另有可选的错误方位项：

~~~text
L_D_wrong = mean(ReLU(1 + D(real, rotate_target_condition(c))))
~~~

当前三项的作用权重均为 0，因此主图只画 real/fake hinge，不画 D 分类梯度。

### 5.7 MT1 meta-transfer loss

MT1 使用一个独立的 SARClassifier64 probe。probe 的 backbone 和持久化 classifier head 都被冻结，但每个 meta episode 会复制一个临时 head：

~~~text
support_images = 当前 generator 生成的 synthetic SAR
support_features = probe_backbone(support_images)
W0,b0 = frozen probe classifier head 的 detached copy

L_support = CE(label_smoothing=.03, Linear(support_features, W0, b0), y_support)
gW,gB = ∂L_support/∂(W0,b0)，create_graph=True
W1 = W0 - 0.1*gW
b1 = b0 - 0.1*gB

query_features = probe_backbone(real_XHH_query)，no_grad + detach
L_MT1 = CE(Linear(query_features, W1, b1), y_query)
~~~

关键点：

- support 是生成图，默认使用当前主 batch 的 32 个样本；
- query 是固定真实 X/HH 记录，当前 640 条池，按车型和 depression 匹配；
- support/query 的类别相同，depression 相同；
- real query feature 不进入 generator 图；
- hypergradient 通过 W1,b1 反向穿过 support feature 和 generator；
- support 的 identity/pyramid 在 MT1 支路先 detach，所以 MT1 不更新 encoder；
- probe、native teacher、D、query 数据都不更新；
- 每 4 个普通 batch 执行一次；
- 当前固定权重是 2.064245482476472，由独立 8-event gradient-RMS calibration 得到。

在代码里，普通 V1 backward、AMP unscale、原始 joint clip 完成后，才把 MT1 的加权 generator gradient 加到 generator 参数上。它不改变普通 total_loss 的记录，也不会把 meta gradient 加到 encoder。

### 5.8 代码中的可选梯度路由（当前均未打开）

当前 MT1 使用 `gradient_routing=coupled` 和 `sar_class_gradient_route=coupled`。代码还提供以下诊断开关，但它们不属于当前 MT1 主流程：

| 开关 | 作用 |
|---|---|
| `gradient_routing=generator_only` | 对 generator 输入的 identity/pyramid detach，使所有 SAR-side loss 只更新 G；E 只收 RGB identity/cross_view 梯度 |
| `sar_class_gradient_route=generator_only` | 仅把 sar_class（以及 margin/geometry class route）限制到 G，其他项保持原路由 |
| `sar_class_gradient_route=encoder_only` | 仅把 class teacher 项送到 E，作为对称诊断，不是推荐训练方式 |
| `teacher_gradient_mode=pcgrad` | 将 weighted sar_class+cluster 与其余 loss 分开求梯度，发生冲突时投影；当前为 none |
| `sar_class_gradient_filter=binomial3_rms` | forward 保持 native teacher 输入不变，只对 class teacher 的反向图像梯度做 3x3 低通；当前为 identity |
| `teacher_eot_views>1` | 对 class/cluster 使用 gain 或小平移的多视图平均；当前为 1 |
| `teacher_speckle_views=2` | 对同一 clean 输出采两个独立 speckle 观测并平均 teacher block；当前为 1 |

这些开关即使只改变梯度归属，也必须作为独立实验，不能与 MT1、loss 合并或判别器变更同时打开。

## 6. 一次训练迭代的完整顺序

### 6.1 初始化阶段

1. 固定随机种子、数据 split 和 640 条 visual validation proxy。
2. 从 V1 parent checkpoint 加载 encoder、generator 和旧 V1 D。
3. 加载并冻结 native SAR classifier；根据真实训练 SAR 计算/读取 40x4 prototypes。
4. 若启用 MT1，加载 synthetic-only probe seed 1729；检查其 metadata，禁止它读真实 test 图。
5. 构造 disjoint 的 real X/HH meta outer query pool。
6. 建立三个 optimizer：

~~~text
E + G: Adam(betas=(.5,.999)), E lr=1e-4, G lr=2e-4
D:     Adam(betas=(.5,.999)), D lr=5e-5
gradient clip: 5
~~~

### 6.2 普通 forward

对每个 batch：

1. 读取 rgb、rgb_alt、真实 roi、XML condition 和 class label。
2. 共享 RGB encoder 得到 z、z_alt、P64/P32/P16/P8。
3. 用 z、pyramid、12 维 condition 生成 clean。
4. 对 clean 施加 speckle 得到 fake。

### 6.3 D step

1. D 看到 real 和 fake.detach()，条件是完整 12 维 condition。
2. 计算 real/fake hinge loss。
3. 当前每 2 个 batch 更新一次 D；fake 不把梯度传回 G。

### 6.4 G/E 普通 loss

1. 暂时冻结 D 的参数，但允许 D(fake) 对 fake 输入产生梯度。
2. native teacher 对 fake 做 class/feature forward；teacher 参数冻结。
3. 计算所有当前启用的普通 loss。
4. 当前默认 gradient_routing=coupled，因此 SAR-side loss 通过 fake 回到 G，并继续通过 identity/pyramid 回到 E。
5. 计算：

~~~text
L_encoder = 10*rgb_identity + 2*cross_view
L_generator = sar_class + cluster + structure + statistics + physics
              + angle + adversarial + feature_match
L_V1_normal = L_encoder + L_generator
~~~

6. 对 L_V1_normal 做一次普通 E/G backward。

### 6.5 MT1 event（每 4 个 batch）

1. 在私有 RNG 中重跑一次 generator，support 的 E 特征 detach，support generator graph 只保留 G 参数。
2. 用生成 support 做 probe 临时 head 的一步可微 SGD。
3. 用真实 X/HH query 计算更新后 head 的 CE。
4. 对 L_MT1 求 generator-only hypergradient。
5. 普通 V1 梯度 unscale 和原始 clip 后，将 2.064245482476472 * grad_MT1 只加到 G 的 .grad。
6. 检查 encoder grad 未改变、probe/native teacher/D 没有梯度、随机数状态恢复。

### 6.6 optimizer step 和 checkpoint

1. 更新 E/G。
2. 若本步无 overflow，才更新可选的 E/G EMA；当前 MT1 ema_decay=0，默认不开 EMA。
3. 记录每个 raw loss、加权 contribution、D 指标、teacher 指标、MT1 diagnostics。
4. 保存 epoch checkpoint 和 config.json。

## 7. 当前 MT1 与原始 V1 的区别

| 项目 | 原始/控制 V1 | 当前 MT1 |
|---|---|---|
| RGB encoder | RGBIdentityEncoder | 完全相同 |
| generator | SpatialROIGenerator | 完全相同 |
| discriminator | conditional projection PatchGAN | 完全相同；不是 K+1 分类 D |
| D condition | 默认完整 12 维 | 完整 12 维 |
| normal loss | V1 的 RGB、SAR teacher、cluster、structure、statistics、physics、angle、adv、FM | 完全保留 |
| sar_class | 推荐 V1 配置为 1；ablation parser 默认值仍是 12 | 仍为 1 |
| structure 64px pixel | 保留 | 保留 |
| perceptual | active config 为 0 | 仍为 0 |
| native teacher | frozen SARClassifier64 | 仍 frozen，只用于普通 V1 teacher loss |
| real query set | 没有 | 增加固定 disjoint 的真实 X/HH query pool |
| synthetic-only probe | 没有 | 增加 seed-1729 probe，仅由生成图训练 |
| 新增 objective | 没有 | L_MT1，不进入普通 LOSS_NAMES/total_loss |
| MT1 梯度 | 不存在 | 只加到 generator，不更新 encoder/probe/D/teacher |
| 普通梯度路由 | coupled | 仍然 coupled |
| 训练 parent | V1 milestone 70 是推荐 V1 的起点 | 推荐 V1 epoch 100 checkpoint |

因此，MT1 不是重写 V1 loss，而是以 V1 作为稳定基线，新增一个和最终 TSTR 目标一致的训练内代理。

## 8. 与历史 Fused V2 的区别

历史 continuous_spatial_fused_v2 曾经同时改变判别器、loss 权重、结构 loss、学习率和初始化：

~~~text
SARClassDiscriminator64：40 个真实类别 logits + 1 个 fake logits
non-registered structure：低频分布 + residual/Sobel moments + D feature moments
G class-adversarial loss
angle curvature loss
去除像素对齐项
~~~

这个版本一次改变太多因素，视觉和真实迁移结果变差，所以当前工作流已经回到 V1。当前代码中虽然仍保留 Fused V2 入口和相关函数，但它们不能画在 MT1 的主流程图中。

## 9. 其他 MT 变体做了什么，以及为什么关闭

这些分支都以 MT1 为控制，只改变一个 meta 变量：

| 变体 | 唯一改变 | 结果/状态 |
|---|---|---|
| MT2-AZ | 增加 frozen probe 的 azimuth-head real-query hypergradient | 机制通过，但 class Top-1 -0.196 pp、Top-5 -0.957 pp，角度改善不足 1 度，关闭 |
| MT3-XDEP | support depression 与 real query depression 改为不同，测试跨 depression 迁移 | support CE 8/8 下降，但 cross-depression query CE 仅 4/8 下降，机制门失败 |
| MT4-ZH | 虚拟 class head 每个 episode 从全零开始，而不是从 trained probe head 开始 | 机制通过，但 Top-1 -4.588 pp、Top-5 -5.279 pp，关闭 |
| MT5-EP2 | 两个独立 synthetic-only probe 的 hypergradient 取平均 | Top-1 平均 +0.716 pp，但 Top-5 仅 +0.361 pp 且 seed 一致性不够，关闭 |
| MT6-U32 | support 采样改为 32 个不同类别 | 机制通过，但 meta/base gradient 最大比例 0.4512 > 0.30，在进入 TSTR 前关闭 |

更早的 A2、L4、K2、RT1 是 native teacher 或 teacher-side 变体，也没有成为当前 accepted baseline。它们不能与 MT1 同时画入当前训练路径。

## 9.1 原 V1 loss 单变量筛选记录

下面是原 V1 ablation protocol 中的三 seed 短筛结果。数值是候选相对匹配 V1 control 的生成到真实域迁移增量；`pp` 是百分点，`MAE` 是方位角圆周误差变化（负值更好）。这些结果用于决定哪些项暂时保留，不代表可以把多个项同时删除或合并。

| 只改变这一项 | Identity Top-1 | Depression Top-1 | Azimuth MAE | 判定 |
|---|---:|---:|---:|---|
| `sar_class: 12 -> 1` | +0.89 pp | +2.14 pp | -2.21 deg | 通过；推荐 V1/MT1 使用 1 |
| `sar_class: 12 -> 0` | +0.31 pp | +2.19 pp | -1.89 deg | 拒绝；identity 两 seed 回退 |
| `cluster: 5 -> 4` | +0.21 pp | -0.94 pp | -0.29 deg | 拒绝 |
| `structure pixel64: 1 -> 0` | -0.36 pp | +0.00 pp | +0.68 deg | 拒绝；保留弱对齐 64px 项 |
| `structure SSIM: 1 -> 0` | +0.00 pp | +1.41 pp | +2.65 deg | 拒绝；方位角每 seed 变差 |
| `structure edge: .5 -> 0` | -0.16 pp | -0.47 pp | +0.39 deg | 拒绝 |
| `physics scatter: 1 -> 0` | -0.21 pp | -0.68 pp | -0.69 deg | 拒绝；identity/depression 变差 |
| `angle weight: .2 -> 0` | -0.47 pp | +1.04 pp | +0.85 deg | 暂不删除；没有稳健收益 |
| `angle: first_order -> curvature` | -0.31 pp | -0.62 pp | +0.59 deg | 拒绝 |
| `D condition: full -> target` | -0.52 pp | -0.73 pp | -1.24 deg | 拒绝 |
| `wrong-azimuth negative: 0 -> .25` | -0.47 pp | -1.88 pp | -0.75 deg | 拒绝；仅作诊断项 |
| `cross_view: 2 -> 0` | -0.52 pp | +0.42 pp | +0.00 deg | 拒绝；identity 每 seed 回退 |
| `feature_match: 5 -> 0` | -0.52 pp | -0.57 pp | +0.01 deg | 拒绝 |
| `statistics: 5 -> 0` | -0.52 pp | +0.16 pp | +0.81 deg | 拒绝；一项方位角门失败 |

这张表的实际含义是：`sar_class=1` 是目前唯一被提升为推荐基线的 loss 改动；64px、SSIM、physics scatter、cross-view、feature-match、statistics 等不能因“看起来相似”就直接删除。若要合并或改权，必须保持总梯度量级可比，并重新做固定 batch 梯度审计和三 seed TSTR。

## 10. 如何判断是否真的缓解分类器捷径

### 10.1 不能使用的唯一标准

下面的指标不能单独证明生成器学到了真实 SAR：

~~~text
native classifier(fake) accuracy = 100%
synthetic-only probe(fake) accuracy = 100%
~~~

因为这些 classifier 的决策边界可能正是 generator 正在利用的 shortcut。

### 10.2 当前主指标：TSTR

TSTR = Train on Synthetic, Test on Real：

1. 用 generator 生成原始 X/HH 图，不使用 native teacher 标签作为训练图像；
2. 用生成图训练一个独立的 40 类 CNN，classifier seed=415；
3. 只在从未用于生成器训练的真实 X/HH ROI 上测试，共 5260 条；
4. 记录真实域 class Top-1、Top-5。

MT1 短筛结果：

| 指标 | G0 | MT1 | 增量 |
|---|---:|---:|---:|
| 真实 X/HH Top-1 | 14.7465% | 19.6895% | +4.9430 pp |
| 真实 X/HH Top-5 | 39.0621% | 45.0697% | +6.0076 pp |

三个 GAN seed 都是正增量，因此 MT1 是当前唯一通过真实域迁移门槛的捷径缓解方案。

### 10.3 辅助诊断

同时记录：

- frozen geometry validator 的 depression Top-1；
- azimuth circular MAE；
- 0/360 闭环、5/30 度响应；
- generated/real frozen feature cosine；
- aligned low-pass L1；
- D real/fake hinge 和 feature moments；
- MT1 support CE before/after；
- MT1 real-query CE before/after；
- gradient finite、gradient ownership、RNG 恢复。

这些是机制和回归诊断，最终选择仍由独立 real TSTR 主导。

### 10.4 当前能下的结论

~~~text
分类器捷径：有证据被缓解，但没有证据已经消失。
生成器真实 SAR 信息：仍然没有完全学到，绝对 TSTR 仍低。
MT1：是有效的训练内 real-domain transfer signal。
原 V1 loss：不能说完全不能动；只能说目前不应多项同时改。
~~~

后续若要合并 rgb_identity/cross_view 或 structure/statistics/feature_match，必须：

1. 只改变一个 loss 关系；
2. 保持总梯度量级可比；
3. 先做固定 batch 的 gradient cosine/RMS 审计；
4. 再做三 seed 短训练；
5. 最后用同一 TSTR protocol 判断。

## 11. 画图时必须标出的梯度归属

建议在图中用不同颜色或线型：

~~~text
蓝色实线：RGB -> encoder -> identity/pyramid -> generator
橙色实线：generator -> clean/fake
红色虚线：adversarial/feature-match -> generator，并在 coupled 路由下继续到 encoder
紫色虚线：native sar_class/cluster -> generator，并在 coupled V1 中继续回到 encoder
绿色虚线：MT1 real-query hypergradient -> generator only
灰色虚线：optional/disabled module，不参与当前训练
~~~

必须画出的关键边：

1. rgb 和 rgb_alt 进入同一个共享 RGB encoder；
2. identity 向量和四层 pyramid 都进入 generator；
3. target SAR condition 与 source RGB angle condition 进入 generator；
4. generator 输出 clean 和 speckled fake 两条支路；
5. D 只负责真实性 hinge，当前不是分类器；
6. native teacher 是独立冻结模块；
7. structure/physics 作用于 clean，statistics/teacher/D/FM 作用于 fake；
8. MT1 的真实 query 只用于 outer CE，不能画成普通 real-image reconstruction；
9. MT1 的绿色梯度只到 generator，不能连接到 encoder、D 或 probe；
10. structure 仍包含弱平移后的 pixel64/pixel32/pixel16、edge 和 global SSIM。

不要在当前 MT1 主图中画：

~~~text
K+1 classifier-discriminator
fake 类别 logit
distributional non-aligned structure
删除 pixel64
MT2/MT3/MT4/MT5/MT6 支路
perceptual active path
~~~

## 12. 代码位置索引

| 内容 | 文件/位置 |
|---|---|
| 当前主训练和 loss 组合 | train_continuous_spatial_v1_ablation.py |
| 数据扫描、RGB/SAR 配对 | joint_data.py::JointROIDataset |
| RGB encoder、generator、PatchGAN、structure/physics | joint_models.py |
| frozen native SAR classifier | sar_classifier_64.py::SARClassifier64 |
| MT1 virtual-head loss | train_continuous_spatial_v1_ablation.py::meta_transfer_head_loss |
| MT1 support/query 训练循环 | train_continuous_spatial_v1_ablation.py 中 meta_transfer_active 分支 |
| 推荐 V1 命令 | run_v1_recommended.sh |
| MT1 命令 | run_v1_meta_transfer.sh |
| MT6-U32 负结果记录 | HIFC_INSPIRED_V1_PLAN.md 第 MT6-U32 节 |
| 所有 V1 loss ablation 结果 | V1_ABLATION_PROTOCOL.md |

当前 MT1 的完整配置实例保存在：

~~~text
runs/v1_ablation/MT1_seed2718_epoch0104/config.json
~~~

该 config.json 是已接受实验的权威参数记录。`run_v1_meta_transfer.sh` 允许通过环境变量覆盖参数，脚本默认值是 smoke/short-run 设置；复现实验结果时应按 config.json 使用 `epoch_size=4000`、4 个 continuation epoch、`meta_transfer_every=4` 和三个 GAN seeds 2718/451/9201。

主实现：

~~~text
/data/newdata/A25_T37_down_大图/code/train_continuous_spatial_v1_ablation.py
~~~
