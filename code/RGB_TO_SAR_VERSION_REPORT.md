# RGB 图像生成 SAR 图像：模型版本、技术流程与结果分析报告

> 更新时间：2026-08-09  
> 当前范围：40 类车辆，RGB 为每车最多 12 个方位视图，输出为 64×64 单通道 SAR ROI；正式连续模型限定 X 波段、HH 极化、15°/30°/45°/60°俯视角和连续方位角。  
> 本报告只把正式训练或具有完整结果的实验作为“主要版本”；`smoke`、`benchmark`、速度测试目录只用于验证代码能否运行，不作为模型效果结论。

## 1. 先说结论

当前视觉效果最好的模型仍然是 **Continuous Spatial V1**：

- 代码入口：[train_continuous_spatial_roi_gan.py](train_continuous_spatial_roi_gan.py)
- 公共模型定义：[joint_models.py](joint_models.py)
- 结果目录：[runs/continuous_spatial_x_hh](runs/continuous_spatial_x_hh)
- 最佳权重：[best.pt](runs/continuous_spatial_x_hh/best.pt)
- 连续角度可视化：[Buick_GL8_all_depressions.png](runs/continuous_spatial_x_hh/Buick_GL8_all_depressions.png)

它目前最好，不是因为网络最大，而是因为它的任务拆分比较合适：先用一个较稳定的生成器产生车体的低频结构和主要散射形态，再用不可学习的随机公式添加 SAR 风格的散斑。后续模型加入了更多噪声网络、判别器、小波约束或多视图混合，理论目标更丰富，但也引入了互相冲突的监督和更多伪影。

不过 V1 仍不能被称为“物理正确的 SAR 模拟器”。它在生成图上的分类准确率为 100%，而同一个分类器在真实 X/HH 测试集上为 92.34%，这是明显的风险信号：生成器可能学会了分类器偏好的车型模板或捷径。现阶段更准确的判断是：

> V1 是当前最好的视觉基线和下一版的初始化起点，但不是已经完成物理真实性验证的最终模型。

---

## 2. 这个项目中的“RGB 图生 SAR”到底是什么

### 2.1 输入和输出

一条训练样本包含：

1. 一张 128×128 RGB 车辆图；
2. 一张同车型、相近方位条件的 64×64 真实 SAR ROI；
3. 车型编号；
4. SAR 目标方位角；
5. SAR 俯视角；
6. 固定实验中的波段 X 和极化 HH；
7. RGB 输入图对应的相机方位角。

RGB 文件原始命名约定为：

- `1.png = 0°`
- `2.png = 30°`
- ……
- `12.png = 330°`

数据读取代码位于 [joint_data.py](joint_data.py)。它会根据真实 SAR 标注中的方位角，在该车型可用的 RGB 图中寻找最近视角。需要特别强调：

> RGB 与 SAR 不是同一传感器在同一时刻采集、逐像素对齐的严格配对图像。它们只是按车型和角度做弱配对。

所以模型不能简单学习“RGB 第 `(x,y)` 个像素应该变成 SAR 第 `(x,y)` 个像素”。真实可学习的关系主要是车型、朝向、长宽比例、轮廓和统计散射形态。

### 2.2 为什么 SAR 不是把 RGB 变成灰度图

RGB 表示可见光反射颜色；SAR 表示微波照射目标后返回的电磁散射强度。两者的亮点含义完全不同：

- RGB 的白色车漆很亮，不代表 SAR 中同一位置一定亮；
- 金属边缘、角反射结构、轮毂、车底等位置可能在 SAR 中形成强散射点；
- SAR 还包含散斑、成像方向、俯视角、阴影和传感器噪声；
- 仅凭普通 RGB 不能唯一推导材料、电磁参数和完整三维结构。

因此当前网络学习的是一个条件分布：

```text
P(SAR | RGB 外形、车型、目标方位角、俯视角、X/HH 条件)
```

而不是精确的物理电磁仿真。

### 2.3 通用的 RGB→SAR 网络流程

大部分 GAN 版本都可以抽象为下面的过程：

```text
RGB 图像
  │
  ├─ RGB 编码器 ─► 全局身份向量：这是什么车
  │                 例如 256 维数字
  │
  └─ 多尺度特征 ─► 64/32/16/8 尺度的空间特征
                    轮廓、长宽比例、局部部件和姿态

目标条件
  ├─ sin(方位角), cos(方位角)
  ├─ 俯视角
  ├─ RGB 源视角
  └─ 波段/极化
             │
             ▼
生成器：4×4 特征逐级放大到 8×8、16×16、32×32、64×64
             │
             ├─ 在各尺度注入 RGB 空间特征
             ├─ 注入方位角和俯视角条件
             ▼
基础 SAR 幅度/反射率图
             │
             ├─ 某些版本直接作为最终结果
             ├─ 某些版本再由噪声网络生成散斑
             └─ V1/V3 等版本用随机成像公式添加散斑
             ▼
最终 64×64 SAR ROI
```

训练时还有一个判别器。判别器同时观察真实 SAR 和生成 SAR，学习判断哪张更像真实图。生成器则反过来尝试欺骗判别器：

```text
真实 SAR ─┐
          ├─► 判别器 ─► 真/假分数
生成 SAR ─┘

判别器：努力分对
生成器：努力让生成图被判为真
```

除了对抗损失，项目中还尝试过：

- 车型识别损失：生成图应该被识别为正确车型；
- 特征簇中心损失：生成图特征接近同车型真实 SAR 的平均特征；
- 结构损失：平移对齐后比较低频轮廓；
- 幅度统计损失：比较均值、标准差、分位数；
- 频谱损失：比较空间频率能量；
- 角度平滑损失：相邻方位角的主体不能突然跳变；
- 判别器特征匹配：生成图和真实图在判别器中间层的统计接近；
- 噪声随机性和身份泄漏审计。

这些损失只能约束某些性质。某一项指标很好，不等于整张图真实。例如生成图分类率高于真实图，通常不是生成质量“超过真实”，而是生成器找到了分类器捷径。

---

## 3. 数据与评估协议

### 3.1 当前正式数据

- RGB 根目录：`amplitude 8-bit data_地距幅度8位数据.7z/RGB`
- SAR 训练根目录：`amplitude 8-bit data_地距幅度8位数据.7z/SOC_40classes_cut/train`
- SAR 正式测试根目录：`amplitude 8-bit data_地距幅度8位数据.7z/SOC_40classes_cut/test`
- SAR 输出分辨率：64×64
- 正式条件：X/HH，俯视角 15°、30°、45°、60°
- 类别数：40

从 latent diffusion 版本开始，较规范的固定划分为：

- 原训练集中的 10,335 张用于训练；
- 原训练集中的 1,822 张用于验证和选择权重；
- 原始 test 的 5,260 张始终隔离，只做正式审计。

V1 较早，尚未采用这套固定的训练/验证协议。因此 V1 的视觉优势可信，但它与后续模型的训练指标不是完全公平的同协议比较。下一轮应当在固定划分上重新训练或保守微调 V1。

### 3.2 当前最重要的评估误区

1. **分类准确率不是视觉真实性。** 分类器可能只利用少量高频纹理或固定亮点。
2. **GAN loss 的绝对值不能跨架构比较。** 不同判别器的标度不同。
3. **逐像素 L1 不适合未配准的 RGB-SAR 弱配对。** 它容易把多个合理结果平均成模糊图。
4. **Lee 滤波得到的“干净 SAR”和“噪声”不是物理真值。** 用它们强监督两个生成器会把分解误差也学进去。
5. **相同验证器既参与训练又参与最终评价，会高估结果。** Dual V2 的生成角度指标优于真实图，就是典型警告。

---

## 4. 版本和代码位置总表

| 版本/实验 | 模型代码 | 训练入口 | 评估/渲染 | 结果目录 |
|---|---|---|---|---|
| 早期 Identity-first ROI GAN | [joint_models.py](joint_models.py) | [train_joint_roi_gan.py](train_joint_roi_gan.py) | [evaluate_joint_roi_gan.py](evaluate_joint_roi_gan.py)、[visualize_joint_roi_gan.py](visualize_joint_roi_gan.py) | [runs/joint_identity_roi_gan](runs/joint_identity_roi_gan) |
| Continuous Spatial V1 | [joint_models.py](joint_models.py) 中 `SpatialROIGenerator` | [train_continuous_spatial_roi_gan.py](train_continuous_spatial_roi_gan.py) | [evaluate_continuous_spatial_roi_gan.py](evaluate_continuous_spatial_roi_gan.py)、[render_continuous_spatial_sar.py](render_continuous_spatial_sar.py) | [runs/continuous_spatial_x_hh](runs/continuous_spatial_x_hh) |
| Style latent 分支 | [joint_models.py](joint_models.py) 中 `SARStyleEncoder`/`StyleSpatialROIGenerator` | [train_style_spatial_roi_gan.py](train_style_spatial_roi_gan.py) | 训练预览和拟合先验权重 | [runs/continuous_spatial_style_v2](runs/continuous_spatial_style_v2) |
| Spatial codebook 分支 | [joint_models.py](joint_models.py) 中 `SARSpatialCodeEncoder`/`CodebookSpatialROIGenerator` | [train_codebook_spatial_roi_gan.py](train_codebook_spatial_roi_gan.py) | 训练预览和 codebook 权重 | [runs/continuous_spatial_codebook_v3](runs/continuous_spatial_codebook_v3) |
| Latent diffusion V3.0 | [v3_latent_sar.py](v3_latent_sar.py) | [train_v3_latent_sar.py](train_v3_latent_sar.py) | [audit_v3_latent_sar.py](audit_v3_latent_sar.py) | [runs/v3_latent_sar](runs/v3_latent_sar) |
| SPADE GAN V4 | [v4_spade_gan.py](v4_spade_gan.py) | [train_v4_spade_gan.py](train_v4_spade_gan.py) | 训练验证图 | [runs/v4_spade_xhh](runs/v4_spade_xhh) |
| Hybrid GAN V5 | [v5_hybrid_sar_gan.py](v5_hybrid_sar_gan.py) | [train_v5_hybrid_sar_gan.py](train_v5_hybrid_sar_gan.py) | 训练验证图 | [runs/v5_hybrid_xhh](runs/v5_hybrid_xhh) |
| Continuous V1.1/V1.2 真实域微调 | V1 生成器 + [v5_hybrid_sar_gan.py](v5_hybrid_sar_gan.py) 的多域判别器 | [train_continuous_spatial_v11.py](train_continuous_spatial_v11.py) | 训练验证图 | [runs/continuous_spatial_v11_xhh](runs/continuous_spatial_v11_xhh)、[runs/continuous_spatial_v12_xhh](runs/continuous_spatial_v12_xhh) |
| Dual Component V1 | [dual_component_sar_gan.py](dual_component_sar_gan.py) | [train_dual_component_sar_gan.py](train_dual_component_sar_gan.py) | [render_dual_component_sar.py](render_dual_component_sar.py) | [runs/dual_component_xhh](runs/dual_component_xhh) |
| One-stage Wavelet | [one_stage_wavelet_sar_gan.py](one_stage_wavelet_sar_gan.py) | [train_one_stage_wavelet_sar_gan.py](train_one_stage_wavelet_sar_gan.py) | [render_one_stage_wavelet_sar.py](render_one_stage_wavelet_sar.py) | [runs/one_stage_wavelet_xhh](runs/one_stage_wavelet_xhh) |
| Dual Component V2 | [dual_component_sar_gan_v2.py](dual_component_sar_gan_v2.py) | [train_dual_component_sar_gan_v2.py](train_dual_component_sar_gan_v2.py) | [render_dual_component_sar_v2.py](render_dual_component_sar_v2.py)、[audit_dual_component_sar_v2.py](audit_dual_component_sar_v2.py) | [runs/dual_component_v2_xhh](runs/dual_component_v2_xhh) |
| Continuous Spatial One-stage V3 | [continuous_spatial_one_stage_v3.py](continuous_spatial_one_stage_v3.py) | [train_continuous_spatial_one_stage_v3.py](train_continuous_spatial_one_stage_v3.py) | [render_continuous_spatial_one_stage_v3.py](render_continuous_spatial_one_stage_v3.py)、[audit_continuous_spatial_one_stage_v3.py](audit_continuous_spatial_one_stage_v3.py)、[benchmark_continuous_spatial_one_stage_v3.py](benchmark_continuous_spatial_one_stage_v3.py) | [runs/continuous_spatial_one_stage_v3_xhh](runs/continuous_spatial_one_stage_v3_xhh) |

补充说明：`joint_pilot_v1/v2/v3`、各种 `smoke` 目录是上述架构形成前的调试或小规模训练；`rgb3d_nerf_*`、`rgb3d_gsplat_*` 是曾尝试用 12 张 RGB 做三维重建和补视角的支线，没有进入当前 RGB→SAR 主模型；`realism/style/codebook` 是围绕 V1 的短支线实验。

---

## 5. 每个主要版本的完整流程和评价

## 5.1 早期 Identity-first ROI GAN

### 如何从 RGB 变成 SAR

```text
单张 RGB
  └─► RGBIdentityEncoder
        ├─► 车型分类 logits
        └─► 256 维身份向量

身份向量 + SAR 元数据
  └─► 全连接层得到 4×4×256 特征
        └─► 4 次双线性上采样
              └─► 64×64 基础 SAR
                    └─► 固定随机散斑公式
                          └─► 最终 SAR
```

这个版本最重要的设计是身份识别器不冻结：RGB 身份分类损失和 SAR 生成损失共同更新 RGB 编码器。生成器主要得到全局身份向量，尚未充分得到 RGB 的空间轮廓。

### 可视化

图中从左到右为 RGB、真实 SAR、生成 SAR。

![早期 Identity-first ROI GAN](runs/joint_identity_roi_gan/rgb_to_sar_visualization.png)

### 评价

- 优点：建立了“身份识别器与 GAN 联合训练”的主框架；生成结果已有明显车型级差异。
- 问题：生成器过度依赖全局身份向量，容易生成每个类别的固定模板；RGB 的具体外形和局部结构利用不足。
- 风险证据：旧分类器在生成图上接近 100%，但在真实图上约 77%；换成原生 64×64 分类器审计后，所选生成器只有 61.94%，真实图为 92.31%，说明早期评价受分类器和预处理影响很大。

## 5.2 Continuous Spatial V1——当前视觉最佳

### 相比早期版本改了什么

V1 不再只使用 256 维全局身份向量。RGB 编码器同时输出四层空间特征：

```text
128×128 RGB
  ├─► 64×64 RGB 特征
  ├─► 32×32 RGB 特征
  ├─► 16×16 RGB 特征
  └─►  8×8 RGB 特征 + 全局身份向量
```

生成器从 4×4 开始解码。每放大一级，就把对应尺度的 RGB 特征投影后加进生成特征：

```text
4×4 隐变量
  └─► 8×8  + RGB 8×8 特征
       └─► 16×16 + RGB 16×16 特征
            └─► 32×32 + RGB 32×32 特征
                 └─► 64×64 + RGB 64×64 特征
                      └─► 基础 SAR
```

目标方位角用 `sin/cos` 表示，解决 359° 和 0° 数值上很远但实际上相邻的问题。条件中还包含俯视角与 RGB 源视角。真实 SAR 标注框的宽高在进入生成器前被置零，避免直接泄漏目标 SAR 的尺寸信息。

随后使用解析散斑模型：

```text
基础幅度 A0
  × 相关对数正态乘性散斑
  × 弱低频照明场、增益和 gamma 变化
  + Rayleigh 接收机底噪与微弱零均值噪声
  = 最终 SAR 幅度
```

这里的噪声由随机公式产生，不由第二个网络自由绘制，所以比较难把完整车体轮廓藏进“噪声图”。

### 训练约束

V1 同时训练 RGB 编码器、生成器和一个条件 PatchGAN 判别器。主要损失包括：RGB 身份、跨 RGB 视图身份一致性、冻结 SAR 分类器的车型损失、车型/俯视角特征簇中心、平移对齐多尺度结构、SAR 幅度统计、物理先验、感知特征、相邻角度平滑、对抗和判别器特征匹配。

散斑不是从第一轮就全强度加入，而是在预热后逐步升到 0.32，先让网络学主体，再增加随机纹理。

### 连续角度可视化

每一行对应一个俯视角，每一列为 0° 到 345°、间隔 15°的目标方位角。

![Continuous Spatial V1](runs/continuous_spatial_x_hh/Buick_GL8_all_depressions.png)

### 评价

- 当前所有版本中，目标最紧凑，黑色背景、主要强散射点和中等强度散斑的组合最自然。
- 0°–345°变化总体连续，没有 Wavelet 版本明显的固定弧线，也没有 V3 那样严重的横竖条纹。
- 15°、30°和45°的视觉效果最好；60°仍明显更乱，存在竖向结构和纹理重复。
- 正式测试中真实 SAR 特征余弦为 `0.5595`，是现有主要 GAN 版本中最高的已记录结果；分俯视角为 `0.5485/0.5990/0.6030/0.4878`。
- 生成图分类率 100%，真实图为 92.34%，说明它可能过度靠近分类器原型。视觉最好与分类捷径风险可以同时存在。

## 5.3 Style latent 分支

### 实现思路

这一分支从 V1 初始化并冻结 RGB 编码器，新增一个真实 SAR 风格编码器：

```text
真实 SAR ─► Style Encoder ─► 均值 μ、方差 logσ² ─► 32维 style
RGB ──────► V1 RGB Encoder ─► 身份和空间特征

身份 + 空间特征 + style + 角度条件 ─► StyleSpatialROIGenerator ─► SAR
```

训练时 style 来自真实 SAR 后验；推理时需要从标准正态、拟合的全局先验或类别/俯视角先验中抽样。

### 可视化

![Style latent 分支](runs/continuous_spatial_style_v2/preview_0015.png)

### 评价

- 优点：开始显式描述真实 SAR 中无法由 RGB 唯一确定的采集风格和纹理多样性。
- 问题：训练时看到了目标真实 SAR 的 style，纯 RGB 推理却只能从拟合先验抽样，训练和推理存在差距。
- 图像出现更多高亮纹理，但并没有稳定地变得更真实；部分结果偏散、目标轮廓减弱。
- 该分支更适合作为风格采样研究，不宜作为当前主生成器。

## 5.4 Spatial codebook 分支

### 实现思路

```text
真实 SAR ─► SARSpatialCodeEncoder ─► 64×8×8 散射空间码
RGB ──────► V1 RGB Encoder ───────► 身份和空间特征

RGB 特征 + 散射空间码 + 条件 ─► CodebookSpatialROIGenerator ─► SAR
```

散射码在 8/16/32/64 各解码尺度中注入。训练结束后把真实 SAR 的散射码形成 codebook，推理时检索或采样。

### 可视化

![Spatial codebook 分支](runs/continuous_spatial_codebook_v3/preview_0020.png)

### 评价

- 优点：比单一全局 style 更能控制散射中心的空间布局，训练结构损失较低。
- 问题：训练阶段直接编码目标真实 SAR，严格来说不是纯 RGB→SAR；推理依赖 codebook 后，生成结果有复制真实散射模板的风险。
- 可视化中细节丰富，但强散射结构有时跟 RGB 轮廓关系不清楚，容易把检索到的 SAR 结构当成目标结构。
- 可作为“有 SAR 参考图时的条件生成”方案，不适合作为无 SAR 输入的主任务基线。

## 5.5 Latent Diffusion V3.0

### 两阶段流程

第一阶段训练 SAR 自编码器：

```text
真实 64×64 SAR
  └─► SAR Encoder ─► 16×8×8 latent
        └─► SAR Decoder ─► 重建 64×64 SAR
```

第二阶段训练 RGB 条件扩散：

```text
RGB + 车型 + 方位/俯视角 ─► 8×8 条件特征
随机高斯噪声 ─► 经过多个去噪时间步 ─► 16×8×8 SAR latent
SAR Decoder ─► 最终 SAR
```

扩散模型训练时学习预测给 latent 加入的噪声；推理时从随机噪声开始，当前实现默认执行约 20 次去噪前向，因此天然比单次前向 GAN 慢。

### 可视化

从左到右为 RGB、真实 SAR、自编码器重建、RGB 条件扩散生成。

![Latent Diffusion V3.0](runs/v3_latent_sar/diffusion_validation_080.png)

### 评价

- SAR 自编码器能重建真实 SAR，说明 decoder 学到了部分真实 SAR 图像流形。
- RGB 条件扩散失败：最终列接近与条件无关的散射噪声，车辆结构利用不足。
- 扩散生成数据训练的分类器在真实测试集 Top-1 只有 2.93%，接近 40 类随机水平 2.5%。
- 原因是噪声预测 MSE 可以在没有充分使用 RGB 条件时下降；模型学会了总体 SAR latent 分布，却没有学好 RGB→SAR 条件映射。
- 对本项目“快速推理”目标不合适，暂不作为主线。

## 5.6 SPADE GAN V4

### 实现思路

SPADE 的核心是不用简单相加 RGB 特征，而是用 RGB 条件为生成特征预测空间相关的缩放和偏置：

```text
归一化后的生成特征 × (1 + RGB预测的scale) + RGB预测的bias
```

V4 使用 RGB FPN、车型 embedding、目标几何条件和 32 维随机 style，从 4×4 逐级生成 64×64 SAR。判别器在原分辨率和半分辨率上判断真伪，并使用错误车型/错误角度真实图作为困难负样本。生成器还通过冻结 SAR 自编码器比较 latent，通过冻结分类器约束车型。

### 可视化

从左到右为 RGB、真实 SAR、生成 SAR。

![SPADE GAN V4](runs/v4_spade_xhh/validation_100.png)

### 评价

- 生成主体清晰，分类器验证准确率达到 97.75%。
- 但结果有明显的规则横纹、竖纹和网格状纹理，且不同车型共享相似背景纹理。
- 生成器倾向于输出容易被分类的高亮车型模板；SPADE 在所有尺度持续注入未严格配准的 RGB 空间特征，会把可见光边缘变成不真实 SAR 条带。
- 分类表现较好，但视觉真实性明显弱于 V1。

## 5.7 Hybrid GAN V5

### 实现思路

V5 把图像分成基础反射率和局部散斑强度：

```text
RGB Encoder ─► 身份 + 四尺度特征
  └─► RGBReflectivityGenerator
        ├─► clean 基础反射率图
        └─► 每个位置的 speckle scale

clean + 随机场 + speckle scale ─► sar_observation ─► 最终 SAR
```

判别器同时观察原始图、高通图和频谱图；训练还加入低频结构、统计、频谱、分类器原型和随机多样性。

### 可视化

从左到右为 RGB、真实 SAR、基础反射率、加入随机观测后的最终 SAR。

![Hybrid GAN V5](runs/v5_hybrid_xhh/validation_100.png)

### 评价

- 优点：开始明确区分“由 RGB 决定的主体结构”和“由采集随机性决定的散斑”。
- 问题：局部噪声尺度仍由带有 RGB 信息的解码特征预测，噪声路径可能携带车型轮廓。
- 图中有大片近乎纯黑孔洞、边缘发黑和局部过亮区域；基础图与最终图的差异不够像真实 SAR 采样变化。
- 验证分类率约 99.29%，但视觉上比 V1 更不自然，再次说明分类率不能替代真实度评价。

## 5.8 Continuous V1.1 / V1.2 真实域微调

### 实现思路

这一支线保留 V1 的 RGB 编码器、空间生成器和随机观测，从 V1 `best.pt` 开始小学习率微调。它降低教师分类器权重，并新增：

- 原始 SAR、高通 SAR、频谱 SAR 多域判别；
- 训练/验证固定划分；
- R1、EMA；
- 暗像素分布、暗轮廓、饱和度和 V1 参考锚点等约束。

V1.1 与 V1.2 使用相同脚本的不同迭代配置。当前 [train_continuous_spatial_v11.py](train_continuous_spatial_v11.py) 已包含后续保守 V1.2 逻辑，因此旧 V1.1 当时的精确源码状态没有单独冻结；应以各自 `config.json` 和 checkpoint 为实验记录。

### V1.1 可视化

从左到右为 RGB、真实 SAR、基础图、随机观测后的生成图。

![Continuous V1.1](runs/continuous_spatial_v11_xhh/validation_060.png)

### V1.2 可视化

![Continuous V1.2](runs/continuous_spatial_v12_xhh/validation_025.png)

### 评价

- 多域真实性约束确实改变了输出分布，但也破坏了 V1 原来紧凑的目标形态。
- 图中出现大块黑边、黑洞、粘连轮廓和过度平滑的主体，说明暗背景/高通/频谱约束权重之间不平衡。
- 判别器能约束“像不像这一批真实图的统计”，却不知道哪些暗区是合理阴影、哪些是生成伪影。
- 这条支线证明：不能通过不断叠加真实性 loss 自动得到更真实的 SAR。

## 5.9 Dual Component V1

### 实现思路

这个版本把最终 SAR 分成两个由网络学习的部分：

```text
RGB ─► 大型 RGB 编码器
        ├─► DenoisedSARGenerator ─► fake_clean
        └─► SARNoiseGenerator(clean, angle, random noise) ─► fake_log_noise

fake = fake_clean × exp(fake_log_noise)
```

真实 SAR 没有真正的 clean/noise 标签，因此代码用 Lee 风格滤波构造伪标签：

```text
real SAR ─► Lee 平滑 ─► real_clean
real SAR / real_clean ─► real_log_noise
```

然后使用三个判别器分别判断 clean、noise 和 full 是否真实。

### 可视化

![Dual Component V1](runs/dual_component_xhh/Buick_GL8_all_depressions.png)

### 评价

- 15°–45°主体连续性尚可，但出现很强的弧线、斜线和固定高亮轨迹。
- 60°几乎退化成整幅高噪声纹理，主体难以辨认。
- 根本问题是 Lee 分解不是物理真值：平滑残差中包含真实目标边缘和强散射结构，噪声生成器自然会学到车体轮廓。
- 三个判别器目标互相竞争：单独“干净图像真”“残差真”并不能保证相乘后的完整 SAR 真。
- 参数量约 3,767 万，复杂度明显增加，但视觉效果没有超过约 376 万参数的后续 V3，更没有超过简单 V1。

## 5.10 One-stage Wavelet

### 实现思路

这一版名称是一阶段，因为 clean 和 noise 由同一个共享解码器一次前向产生，但训练仍然使用 Lee 分解的伪 clean/noise 目标：

```text
RGB + 身份 + 几何
  └─► 共享 SPADE/FIR 解码器
        ├─► clean head
        └─► noise scale/bias + 随机场
              └─► log_noise

clean × exp(log_noise) ─► full SAR
```

三个判别器分别观察 clean、full 和 Haar 小波纹理能量。

### 可视化

![One-stage Wavelet](runs/one_stage_wavelet_xhh/Buick_GL8_all_depressions.png)

### 评价

- 比完全独立的双生成器共享了更多结构，计算流程更紧凑。
- 但小波判别器只知道高频能量是否相似，不知道高频亮点是否位于物理合理的散射位置。
- 结果把目标优化成固定弧线、车顶亮带和条纹，45°/60°最明显。
- 验证教师准确率从早期接近 100%下降到第120轮的 78.10%，说明长时间联合训练破坏了主体身份结构。

## 5.11 Dual Component V2

### 实现思路

V2 重点解决两个问题：使用全部 12 张 RGB，以及防止随机噪声被忽略。

```text
12 张 RGB ─► 共享编码器 ─► 每视图四尺度特征
目标方位角 ─► 环形视图注意力 ─► 选择/融合相邻视图
俯视角和方位角 ─► 逐尺度 Geometry-SPADE
                         └─► clean SAR

与车型无关的全分辨率随机场
  + 仅依赖俯视角/clean统计的有界参数
  └─► 相关乘性散斑 + 加性噪声
        └─► full SAR
```

训练分为 70 轮 clean、40 轮 noise、40 轮 joint。另有几何验证器约束车型、俯视角和方位角；噪声身份泄漏分类器尝试保证残差不包含车型。

### 完整 SAR 可视化

![Dual Component V2 full](runs/dual_component_v2_xhh/Buick_GL8_all_depressions_full.png)

### 单独噪声可视化

![Dual Component V2 noise](runs/dual_component_v2_xhh/Buick_GL8_all_depressions_noise.png)

### 评价

- 随机性目标基本实现：不同 seed 的噪声相关性为 `0.0157`，说明随机路径没有被忽略。
- 噪声身份泄漏准确率为 `6.92%`，明显低于直接携带完整车型信息，但仍高于40类随机水平 `2.5%`。
- 生成方位角验证指标看似优于真实图，但验证器同时参与训练，存在验证器捷径，不能据此宣称几何比真实 SAR 更准确。
- 真实特征余弦只有 `0.4453`；60°只有 `0.3589`，明显弱于 V1 的 `0.4878`。
- 15°噪声残差仍可看到主体轮廓；30°接近均匀细噪声；45°/60°噪声强度突然增大，说明不同俯视角没有学到统一、连续的成像规律。
- 12 个未三维配准的 RGB 二维特征直接融合，会造成重影和平均化；后期 joint 阶段又进一步破坏 clean 主体。

## 5.12 Continuous Spatial One-stage V3

### 实现思路

这一版重新回到 V1 的单生成器，并从 V1 epoch 100 权重初始化：

```text
12 张 RGB ─► 共享 V1 RGB Encoder
  ├─► 每视图特征
  └─► 聚合身份向量

目标方位角 ─► 环形查询，选权重最高的两个相邻视图
方位角/俯视角 ─► 8/16/32/64 尺度零初始化调制
V1 解码器 + 低通抗混叠 ─► base amplitude

固定三通道随机场
  ├─► 相关高斯乘性散斑，sigma∈[0.06,0.14]
  └─► Rayleigh 接收机底噪，eta∈[0.002,0.010]
        └─► 最终 SAR
```

模型只输出最终 SAR，不再训练 clean/noise 真值，不使用教师分类损失反向传播。一个共享判别器包含原图、半分辨率和 Fourier 三个分支。正式推理接口支持固定 seed，可复现且只改变细粒散斑。

### 固定 seed 连续角度图

![Continuous Spatial One-stage V3 fixed](runs/continuous_spatial_one_stage_v3_xhh/Buick_GL8_all_depressions_fixed.png)

### 多模型逐样本对比

列为 RGB、真实 SAR、V1、Wavelet、Dual V2、V3。

![多模型对比](runs/continuous_spatial_one_stage_v3_xhh/model_comparison.png)

### 评价

- 随机建模比双分量版本健康：seed 相关性 `0.0125`，低频 seed 差异 `0.0056`，白化残差车型泄漏 `2.24%`，接近40类随机水平。
- 推理速度达到目标：缓存 RGB 特征后，单角度约为 V1 的 `1.46×`，72角度批量约 `1.20×`。
- 原生 SAR 特征余弦 `0.5201`，优于 Dual V2，但仍低于 V1 的 `0.5595`。
- 最大问题是几何条件退化：生成方位角平均误差 `37.45°`，真实图为 `20.01°`；生成俯视角准确率 `58.16%`，真实图为 `85.08%`。
- 60°方位误差达到 `48.25°`。视觉上表现为重复车型模板、矩形边界、横竖条纹和角度变化不充分。
- 多视图融合虽然只选两个视图，但这些视图仍未做三维配准；零初始化模块在训练后逐渐改变 V1 主干，使原来的视觉优势被侵蚀。

---

## 6. 横向结果总结

### 6.1 关键定量结果

不同版本的评估器并不完全相同，下面只列可合理对照的审计数据，不能把所有训练 loss 放在同一尺度排序。

| 模型 | 真实 SAR 特征相似度 | 60°特征相似度 | 生成身份表现 | 主要结论 |
|---|---:|---:|---:|---|
| Continuous Spatial V1 | **0.5595** | **0.4878** | 原生分类器 100% | 当前特征和视觉最佳，但有分类捷径风险 |
| Dual Component V2 | 0.4453 | 0.3589 | 几何验证器身份 79.22% | 随机性好，完整图真实度和60°较差 |
| One-stage V3 | 0.5201（原生特征） | 0.4420 | 原生分类器 97.41% | 噪声健康，但角度控制和视觉条纹较差 |
| Latent Diffusion V3 | 未使用同一特征审计 | — | 生成数据→真实测试 2.93% | RGB 条件映射失败 |

### 6.2 各版本分别解决了什么

- 早期 Joint GAN：证明身份编码器可以与生成器联合训练。
- V1：证明多尺度 RGB 空间特征对目标外形有效。
- Style/Codebook：尝试表达“一张 RGB 对应多种合理 SAR”的不可确定性。
- Diffusion：尝试先学习真实 SAR latent，再做条件生成。
- V4/V5：尝试更强空间调制、真实域 latent、反射率与随机观测。
- V1.1/V1.2：尝试减少分类教师主导，加强真实域判别。
- Dual V1/Wavelet：尝试显式学习 clean 与 noise。
- Dual V2：解决 12 视图、随机噪声不可忽略和角度验证。
- One-stage V3：移除伪 clean/noise 标签，回到单生成器和有界随机成像。

这些实验不是没有价值。它们共同说明了三条很重要的经验：

1. SAR 噪声不能被当作任意可学习图像，否则会吸收车辆结构；
2. 未配准的 12 张 RGB 不能直接当成对齐特征平均；
3. 分类器、频谱和小波统计只能做辅助，权重过高就会生成“指标正确、视觉错误”的伪影。

---

## 7. 为什么 V1 的视觉效果最好

### 7.1 它把确定性主体和随机散斑分得恰到好处

V1 生成器负责车体的低频幅度和主要强散射结构，散斑由固定公式负责。生成器不需要同时猜测“什么是车”“什么是随机噪声”，任务更容易辨识。

### 7.2 单判别器比三个互相竞争的判别器稳定

V1 只要求最终图整体像真实 SAR。Dual/Wavelet 模型分别判 clean、noise、full，但 clean/noise 标签本身是人工分解的，因此判别得越认真，越可能把分解伪差当成真实规律。

### 7.3 最近 RGB 视图虽然简单，却避免了未配准多视图重影

当前每辆车的 12 张 RGB 只是不同角度图片，没有可靠相机内外参、三维表面对应关系和遮挡关系。V1 只使用最近视图，使主体边缘比较清楚。多视图模型直接融合二维特征后，车头、车尾和侧面特征会在同一特征图中冲突。

### 7.4 双线性上采样和适中的模型容量减少了棋盘纹

V1 从 4×4 逐级双线性上采样，再卷积生成 64×64 图，不使用容易产生棋盘格的转置卷积。模型容量也与当前数据量更匹配。

### 7.5 V1 的损失主要保护主体结构

较高的结构权重、特征簇中心和跨视图身份一致性，使车体始终位于中心并保持紧凑。后续高通、频谱、小波和暗区约束过强时，网络会优先满足统计量，而牺牲主体可读性。

---

## 8. V1 仍然存在的问题

### 8.1 车型模板与分类器捷径

生成分类率 100% 高于真实 92.34%。需要通过以下消融测试判断模型到底用了多少 RGB：

- 把 RGB 全部置黑，生成结果是否几乎不变；
- 打乱车型标签但保留 RGB，结果跟标签还是跟图像；
- 同车型不同 RGB 视角交换，主体结构是否发生合理变化；
- 不同车型 RGB 与标签互换，生成器依赖哪一个；
- 只保留轮廓、只保留颜色或遮挡局部，观察散射结构变化。

如果置黑 RGB 后仍能生成正确车型，说明模型主要记住了类别和角度模板，而不是做真正的图生图。

### 8.2 连续方位角可能只是纹理插值

V1 的相邻角度看起来连续，但连续并不等于几何正确。需要检查强散射中心是否随方位角平滑移动，不能只看整图像素变化。

### 8.3 60°明显较弱

60°特征相似度仅 0.4878，低于30°和45°。可视化中也有更强的竖向纹理和背景能量，说明俯视角调制不足，模型可能把60°当作一种噪声风格，而不是新的投影几何。

### 8.4 随机观测参数未经真实数据标定

V1 的散斑强度、相关核、增益、gamma 和接收机噪声范围主要是经验值。它看起来像 SAR，但未证明与这套数据的真实噪声统计一致。

### 8.5 RGB 与 SAR 弱配对形成任务上限

当前数据同一类别基本只有一个代表车辆及其12个RGB视图。网络很难证明自己在使用具体 RGB 个体信息，因为车型标签本身已经包含大量外形信息。若想做“同车型不同个体”的图生图，必须增加同类别多车辆实例，或者得到更严格的 RGB/SAR/角度对应关系。

---

## 9. 推荐的下一步：以 V1 为主干做最小改造，而不是再建一套大模型

建议下一版命名为 **Continuous Spatial V1-R（Realism-preserving revision）**。目标是保住 V1 的视觉形态，只修正已确认的问题。

### 阶段 A：先建立可信的评价基线

1. 冻结并备份 V1 `best.pt`，任何新版本都必须与它逐样本并排。
2. 使用 V3 已固定的 10,335/1,822 划分重新训练或微调 V1，正式 test 不参与选权重。
3. 固定 10–20 个车型、四个俯视角、24个方位角和固定随机种子，形成永久视觉审计集。
4. 增加黑 RGB、错标签、错视图、同类视图交换等消融图，直接测量 RGB 利用率。
5. 分类器只做独立审计，至少使用两个不同架构；生成器训练不再直接优化最终审计分类器。

### 阶段 B：保留 V1 主路径，只添加零初始化的小残差

核心结构建议为：

```text
最近 RGB 视图 ─► 原 V1 路径 ─────────────────► V1 base SAR

相邻第二视图 ─► 轻量特征编码
目标相对角度 ─► 置信度门控 ─► 小残差 ΔSAR ───┘

最终 base = V1 base + α·ΔSAR，α 从 0 开始且限制上限
```

具体规则：

- 目标正好是 0°、30°、…、330°时，默认保持原 V1 最近视图路径；
- 只有目标落在两个 RGB 视图之间时，才允许第二视图提供小残差；
- 不直接平均两个完整二维特征图；
- 门控残差零初始化，并限制幅度，防止训练初期破坏 V1；
- 以“目标角相对源视角的差值”调制各尺度，而不是只给一个全局角度向量。

### 阶段 C：保留解析随机成像，但用真实数据标定参数

不再训练完整噪声图，也不使用 Lee clean/noise 伪标签。建议：

1. 在真实 SAR 背景区和主体区分别统计局部均值—方差关系、空间自相关和径向功率谱；
2. 按俯视角拟合有界散斑强度和接收机底噪范围；
3. 噪声参数只能依赖俯视角和采集条件，不能依赖车型类别或 RGB 特征；
4. 随机 seed 只改变高频残差，4×下采样后的主体保持不变；
5. 继续做白化残差身份泄漏审计，目标保持在随机水平附近。

### 阶段 D：简化真实性约束

建议保留一个条件判别器，以原生 64×64 Patch 分支为主。可添加很弱的半分辨率或频谱分支，但不能让频谱分支主导。生成器损失优先级建议为：

1. 最终图条件对抗真实性；
2. 平移容忍的低频主体结构；
3. 幅度分位数和散射中心空间矩；
4. 判别器特征匹配；
5. 弱角度连续性；
6. 弱频谱统计。

明确不再使用：

- Lee 分解得到的 clean/noise 强监督；
- 独立可学习噪声图生成器；
- 三个互相竞争的 clean/noise/full 判别器；
- 小波纹理判别器主导训练；
- 冻结分类器交叉熵长期高权重反向传播；
- 未配准12视图全特征直接平均。

### 阶段 E：解决60°和角度控制

- 训练批次按四个俯视角均衡采样；
- 为每个俯视角设置小型、零初始化的逐尺度调制，而不是改变整个生成器；
- 使用与训练生成器完全独立的几何验证器做审计；
- 评估强散射点质心、主轴和方位角，而不仅是分类准确率；
- 对 `θ-5°、θ、θ+5°` 的低频结构使用二阶平滑，但允许高频散斑独立变化。

### 推荐训练顺序

```text
第 1 步：在固定划分上复现 V1，得到公平 V1 baseline
第 2 步：冻结 V1，只训练零初始化的角度/第二视图残差模块
第 3 步：解冻 V1 最后两层，小学习率联合微调
第 4 步：加入真实数据标定后的解析随机观测
第 5 步：只根据验证集视觉综合分数和独立审计选权重
第 6 步：最后一次运行正式 test 审计
```

每完成一步都与原 V1 并排。如果某一步使 V1 的紧凑目标、连续角度或背景自然度下降，就回退该模块，而不是继续用更多 loss 补救。

---

## 10. 最终建议

当前主线应回到 **Continuous Spatial V1**。下一步不是简单增加 epoch、参数量或判别器数量，而是完成三件更关键的事：

1. **证明模型确实利用 RGB。** 用置黑、错配和视图交换消融排除车型模板捷径。
2. **保住 V1 视觉主体，只以有界残差加入第二视图和逐尺度角度调制。**
3. **用真实 SAR 统计标定不可学习的随机成像公式。** 噪声保持随机且与身份无关。

如果这三步完成，下一版才有希望同时满足：看起来像真实 SAR、连续角度正确、噪声可随机采样、不会走分类器捷径，并维持 GAN 的快速单次前向推理。
