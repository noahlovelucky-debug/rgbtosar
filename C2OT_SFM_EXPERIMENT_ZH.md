# C2OT-SFM 实验说明

本实验针对无像素配对的 RGB->SAR 训练。RGB 只提供车型身份与多视图信息，SAR 训练样本提供真实分布和目标采集条件；不把 batch 中相同下标的 RGB/SAR 当作同一实例。

## 基线与新变量

基线是 `hifc_unpaired_conditioned_v1` 的 HiFC-inspired trainer。生成器、RGB identity encoder、条件 PatchGAN、LTC、geometry、学习率、EMA、数据 split 和训练长度均保持不变。唯一训练变量是 SFM 的实现：

```text
--sfm-mode batch              原始 itemwise SFM（默认）
--sfm-mode conditional_set_ot 条件原型白化 sliced-OT SFM
```

新方法不增加 native class CE，也不把真实测试集放进训练。

## C2OT-SFM 流程

训练开始时只扫描 GAN train split 的真实 SAR ROI，用冻结的 `SARClassifier64` 得到：

```text
group = class × band × polarization × depression
prototype[group] = mean/std(native embedding), mean/std(LTC signature)
```

共 40×2×4×4=1280 个条件组；缺失组回退到全局统计。prototype 保存为 `sfm_prototypes__all_all_all.pt`，并记录训练 split digest、teacher checkpoint 的大小/mtime 和 ROI 尺寸。

每个 G step：

1. fake 和当前真实 SAR 分别送入 native teacher 得到 embedding，并计算 15 维 LTC signature。
2. 对每张图减去自身条件组 prototype，再按 prototype 的逐维标准差白化。
3. 拼接 384 维 teacher residual 与 15 维 LTC residual，使用固定的 64 个随机投影。
4. 每个投影方向分别排序 fake/real 投影值，计算 `SmoothL1` 的一维 Wasserstein（sliced-OT）距离。排序意味着集合匹配，不依赖 batch 下标，也不比较 H×W 像素坐标。
5. 加入 fake 到条件 prototype 的 cosine anchor，保持车型/采集条件的粗粒度身份；保留原有共享判别器 feature-moment 项。

实现形式为：

```text
L_C2OT-SFM = L_sliced-Wasserstein
             + 0.25 L_prototype-anchor
             + 0.5 L_D-feature-moment
```

原训练总损失中的 `sfm_weight=2.0` 不变。native teacher、真实 embedding、prototype 都 detach；`native_gradient_mode=full` 只允许 fake 经过 teacher 表征回传到 E/G，teacher 参数本身始终冻结。

在 DDP 下，8 个 rank 的 fake/real residual set 先 all-gather 成 global batch=64。all-gather 的远端 fake 使用 detach，本 rank 的 fake 保留 autograd 路径；sliced 项乘 world size，以抵消 DDP 梯度平均，保持与原 global-mean 的梯度尺度一致。

## 其余损失与反传

```text
L_G = 1.0 L_adversarial
    + 1.0 L_rgb_identity
    + 2.0 L_LTC
    + 2.0 L_C2OT-SFM
    + 0.3 L_geometry
```

第 1 个 epoch 按原配置关闭 adversarial warmup。`L_rgb_identity` 使用 RGB 与 `rgb_alt` 的两次 class CE 加 identity cosine invariance；`L_LTC` 比较局部残差、对比度和 Haar signature 的 batch moments；`L_geometry` 用 frozen native auxiliary heads 监督 band/polarization/depression/azimuth。判别器仍是共享的 conditional projection PatchGAN，D step 只看 real 与 `fake.detach()`，并保留 wrong-class、wrong-condition 和 lazy R1。

## 启动命令

```bash
cd code
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  train_hifc_unpaired_sar_gan.py \
  --rgb-root ../A02/RGB \
  --sar-train-root ../A02/SOC_40classes_cut/train \
  --native-classifier-checkpoint server_results/sar_native64_multitask_v1/best.pt \
  --output runs/hifc_c2ot_sfm_full_ddp \
  --band all --polarization all --depression all \
  --epochs 120 --epoch-size 24000 --batch-size 8 --workers 2 \
  --native-gradient-mode full \
  --sfm-mode conditional_set_ot --sfm-projection-count 64 \
  --sfm-ltc-cost-weight 0.5 --sfm-anchor-weight 0.25 \
  --seed 20260830
```

当前服务器的 tmux 会话名为 `hifc_c2ot_sfm_full`，日志在 `code/runs/hifc_c2ot_sfm_full_ddp/run.log`。`config.json` 保存完整参数，`history.csv` 保存每 epoch 的 loss/teacher 诊断，`validation_*.png` 保存 RGB、real、clean、observed fake 对照。

## 验收标准

native class accuracy 仅作为诊断，不能用来选择 checkpoint。最终用冻结 GAN 生成的图训练独立 image-only CNN，在未参与生成训练的真实 X/HH test 上报告 Top-1、Top-5、四个 depression 分层、azimuth circular MAE 和 bootstrap CI。三 classifier seeds 的平均 Top-1 目标是相对旧 48.32% 至少提升 3 个百分点，Top-5 至少提升 2 个百分点；任何 seed 不下降超过 1 个百分点。混合真实+生成分类器结果只用于低资源诊断，不能替代 TSTR。
