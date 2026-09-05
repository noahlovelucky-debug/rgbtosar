# UNSB-SAR：轮廓条件的无配对 SAR 桥接扩散

这是一条独立于旧 HiFC/GAN 和 conditional_sar_diffusion.py V2 的新实验主线。
目标不是把 RGB 当成 SAR 的逐像素输入，而是学习一个从“车辆轮廓端点”到
“SAR 目标分布端点”的随机桥。RGB 和 SAR 只通过车型、波段、极化和俯视角建立
弱配对，绝不比较 RGB 坐标 (h,w) 与 SAR 坐标 (h,w)。

## 为什么选择 UNSB

官方 cyclomon/UNSB（ICLR 2024，论文 Neural Schrödinger Bridge）把无配对图像翻译
写成两个域之间的随机桥，并用多阶段网络、目标域真实性和内容正则化保持语义。
这比 V2 的“从高斯噪声直接回归 SAR”更接近本任务：真实 SAR 可以作为目标域端点
样本，而不需要一个与 RGB 像素对应的目标图。

官方代码副本位于：

    external/UNSB/

本项目没有直接运行 UNSB 的 CycleGAN 数据读取器，因为上游假设 trainA/trainB
目录且没有车型/采集条件。这里复用了其桥接、successive refinement、SB energy
和 PatchNCE 思想，并用当前 JointROIDataset 和 torchrun/DDP 重写了数据和训练边界。

调研过的其它一手代码和论文：

* PDM-SAR：参考 SAR 强度、散斑和频谱评测，不直接复用其单域数据假设。
* DiffusionSat：参考 metadata 条件分路和 zero-initialized control adapter。
* GeoDiff-SAR：参考显式视角几何先验；当前数据没有 CAD/点云，不能直接使用其
  Stable Diffusion 3.5 底座。
* SAR-DisentDM：参考类别/方位分路和时步语义加权；它没有可直接复用的公开训练代码，
  且任务不是 RGB 无配对翻译。

## 数据和弱配对

训练入口仍然使用：

    ../A02/RGB/<class>/*.png
    ../A02/SOC_40classes_cut/train/<class>/*.tif + *.xml

每个 JointROIDataset 样本包含：

* rgb：主 RGB 视图，形状 [B,3,128,128]，由 RGBA PNG 合成背景后的 RGB；
* rgb_alt：同车型另一视图，形状 [B,3,128,128]，只用于全局身份 token；
* rgb_mask、rgb_alt_mask：PNG alpha 前景，形状 [B,1,128,128]，范围 [0,1]；
* roi：真实 SAR ROI，形状 [B,1,64,64]，范围 [-1,1]；
* meta/depression：目标 SAR 的方位、波段、极化和俯视角；
* rgb_angle：主 RGB 视角，用 sin/cos 传入桥接网络。

RGB 与 SAR 可以来自不同观测。数据集只保证车型和采集条件相容，因此这里不是
paired reconstruction。return_rgb_mask=True 才会创建一次性的 alpha 缓存：

    /tmp/rgb2sar_cache/joint_rgb_mask_cache_*.pt

## 模型流程

代码入口是 unsb_sar_bridge.py，G/D/E 训练入口是 train_conditional_sar_unsb.py。
G 的记录架构名为 unsb_sar_silhouette_bridge64_gde_v1。

### RGB 空间控制和身份 token

SpatialControlEncoder 输入 cat(rgb, alpha)，即 4x128x128，经过 stride-2 卷积得到：

    64x64 : 64 channels   control64
    32x32 : 128 channels  control32
    16x16 : 256 channels  control16
     8x8  : 512 channels  control8

四个 control map 通过 zero-initialized 1x1 adapter 注入 SAR transition U-Net。
主视图的 control 才进入空间路径；rgb_alt 只走全局池化路径，与主 token 平均。
这避免把两个不同视角的像素硬加到同一个 SAR 坐标。

encoder 另有 40 类 RGB-only head。它只用于 RGB 两视图的身份 CE 和 token cosine
一致性。40 类 logits/softmax posterior 不进入 SAR U-Net，native SAR 分类器也不
创建，因此没有冻结 SAR teacher 向生成器传递的 shortcut。

### 源端点：soft silhouette prior

alpha mask 缩放到 64x64 后，用 15x15 average pooling 得到平滑占据场：

    P_rgb = 0.35 * (2 * AvgPool15(alpha_64) - 1)

P_rgb 是与 SAR 同尺寸的 1 通道软轮廓端点。它包含前景和边界的低频形状，
不包含 RGB 颜色，也不读取任何配对 SAR 像素。

### 采集条件和源视角

目标条件 c 为 12 维：

    [sin(target_az), cos(target_az),
     dep15, dep30, dep45, dep60,
     band_X, band_KU,
     pol_HH, pol_HV, pol_VH, pol_VV]

网络另外计算目标方位 1--4 阶环形谐波（8 维）、源 RGB 视角 sin/cos（2 维）
和相对方位差 sin/cos(target-source)（2 维）。因此 0/360 度是连续的。

### Bridge U-Net

BridgeUNet 输入中间状态 x_t ([B,1,64,64]) 和连续时间 t∈[0,1]，输出同尺寸速度
v_theta(x_t,t,c)。

编码路径：

    x_t -> Conv 1->64 -> block64 + control64
        -> stride2 -> 128 -> block32 + control32
        -> stride2 -> 256 -> block16 + control16 + cross-attention
        -> stride2 -> 512 -> block8  + control8  + cross-attention

解码路径使用双线性上采样、卷积和对应 skip：

    8x8/512 -> 16x16/256 -> 32x32/128 -> 64x64/64 -> velocity(1)

每个 residual block 包含 GroupNorm、3x3 Conv、时间/条件 FiLM、第二个
GroupNorm/Conv、同尺度 zero-start control residual 和 shortcut 加法。
16x16、8x8 还用 SAR feature query 对 geometry control token 做 cross-attention。
这是一条多尺度轮廓路径，不是单一 global FiLM。

### 无配对随机桥

一个 batch 中，P_rgb 和真实 SAR y 仅按车型/采集条件弱匹配。随机采样 t 和 ε：

    sigma(t) = 0.12 * sqrt(t * (1-t))
    x_t       = (1-t) * P_rgb + t * y + sigma(t) * ε
    v_star    = stop_gradient(y - P_rgb)

y 可以是同车型、同波段/极化/俯视角的任意真实 ROI，不要求与 RGB 同一次观测。
推理时不读取真实 SAR；真实 y 只用于训练时的 target-domain bridge endpoint。

### G/D/E/PatchNCE 训练

生成器在 source endpoint 上做 5 步 successive refinement，得到 fake SAR：

    x_(k+1) = x_k + (1/5) * v_theta(x_k, (k+0.5)/5, c)

每个 batch 还生成第二条独立噪声轨迹。单一条件投影 PatchD 的输入是 SAR 图像和
detached identity/acquisition condition；real 与 fake 使用同条件，real 的打乱
condition 作为 wrong-pair negative。PatchD 不接 alpha 空间图。

BridgeEnergy 接收同一条件下两条轨迹的相邻状态，区分 positive trajectory 和
independent negative trajectory。它只作为 SB 正则，不从类别标签构造捷径。
BridgePatchEncoder 在 source soft silhouette 与 fake 的 32/16/8 特征之间做
PatchNCE；它约束的是源轮廓结构，不比较任意真实 SAR 的同坐标像素。

## Loss 和反传

生成器的核心目标只有 UNSB 的三项：

    L_D = hinge(real_score, fake_score)
          + 0.25 * hinge(wrong_condition_score)

    L_G = 1.0 * L_adv
          + 0.1 * L_SB
          + 1.0 * L_NCE

其中：

    L_adv = -mean(D(fake, detached_condition))
    L_SB  = 0.1 * (energy_contrast + mean((x_t - x_(t+1))^2))
    L_NCE = 多尺度 patch 对比交叉熵

训练脚本保留一个很小的 RGB-only 辅助项：

    L_identity = 0.5*CE(rgb_view1, class)
                + 0.5*CE(rgb_view2, class)
                + 0.25*(1-cos(z1,z2))

    L_total = L_G + 0.1*L_identity

梯度归属：

    L_adv -> G（D 参数冻结，只保留 fake 输入梯度）
    L_SB  -> G（E 参数冻结，只保留轨迹输入梯度）
    L_NCE -> G + PatchEncoder
    L_identity -> RGB encoder + RGB class head
    native SAR classifier -> 不创建，不加载，不参与 forward/backward
    TSTR classifier -> 训练结束后独立训练和评估，不进入训练图

这里没有 paired RGB/SAR L1、cycle consistency、bbox loss、SAR native class CE、
teacher feature loss 或 TSTR hypergradient。减少 loss 是为了让实验归因于一条
“无配对随机桥 + 多尺度轮廓条件”方法。

## 文件和命令

新增/使用文件：

    joint_data.py                  # 可选 alpha mask 输出
    unsb_sar_bridge.py             # G、D、E、PatchNCE、bridge rollout/sampler
    train_conditional_sar_unsb.py  # 单卡/DDP、G/D/E 更新、EMA、checkpoint
    render_unsb_sar_bridge.py      # 0..330 度条件 sweep
    test_unsb_sar_bridge.py        # shape、梯度、采样单测
    external/UNSB/                 # 官方 UNSB 参考代码，MIT

单卡 smoke：

    cd /data/newdata/A25_T37_down_大图/code
    CUDA_VISIBLE_DEVICES=0 python train_conditional_sar_unsb.py       --rgb-root ../A02/RGB       --sar-train-root ../A02/SOC_40classes_cut/train       --output runs/unsb_sar_gde_smoke       --epochs 1 --epoch-size 4 --batch-size 2 --workers 0       --base 8 --token-dim 32 --control-base 4       --discriminator-base 4 --energy-base 4 --patch-base 4       --bridge-steps 2 --sample-steps 2 --limit-train-batches 2 --no-amp

8 卡正式入口（先用 5 epoch pilot，再决定 120 epoch）：

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8       train_conditional_sar_unsb.py       --rgb-root ../A02/RGB       --sar-train-root ../A02/SOC_40classes_cut/train       --output runs/unsb_sar_bridge_all_conditions       --band all --polarization all --depression all       --epochs 5 --epoch-size 24000 --batch-size 8 --workers 4       --base 64 --token-dim 256 --control-base 32       --discriminator-base 32 --energy-base 16 --patch-base 16       --bridge-steps 5 --sample-steps 5 --preview-every 1 --save-every 1

生成 sweep：

    python render_unsb_sar_bridge.py       --checkpoint runs/unsb_sar_bridge_all_conditions/epoch_005.pt       --rgb-root ../A02/RGB --class-name Buick_GL8 --source-angle 0       --depression 30 --band X --polarization HH       --sample-steps 5 --output runs/unsb_sar_bridge_all_conditions/sweep.png

## 评测和 go/no-go

当前 smoke 已验证 alpha 读取、G/D/E/PatchNCE 反向、EMA、checkpoint 和 sweep
链路；它没有训练到可以与 HiFC 比较的视觉质量。

正式训练按以下顺序评测：

1. 真实/生成强度直方图、功率谱、稀疏散射点比例、连通分量和角度 sweep；
2. 三个独立随机初始化 SAR CNN 的纯生成训练到真实 X/HH 测试（TSTR）；
3. 真实 50%、10%、5%、1% 与生成数据混合训练，测试集严格只使用未见真实 X/HH；
4. source angle、target angle、class label 各自置乱，检查 geometry/identity/acquisition
   三条路径的独立效应；
5. 分别报告 X/KU、HH/HV/VH/VV、15/30/45/60 度，不能用 native teacher 训练准确率
   代替真实泛化。

首轮晋级条件建议相对于现有 V2：TSTR Top-1 至少提高 3 个百分点，四个俯视角中
最差层不下降超过 2 个百分点，同时 SAR feature distance、功率谱和散射统计不能
明显恶化。只有这些独立指标同时改善，才值得继续 120 epoch 和写论文。

## 公开来源

    https://github.com/cyclomon/UNSB
    https://arxiv.org/abs/2305.15086
    https://github.com/thomaskerdreux/PDM_SAR_InSAR_generation
    https://github.com/samar-khanna/DiffusionSat
    https://github.com/wxt117/GeoDiff-SAR
    https://ojs.aaai.org/index.php/AAAI/article/view/38165

