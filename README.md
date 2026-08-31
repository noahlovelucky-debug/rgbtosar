# 身份优先的 RGB → SAR ROI 联合训练

## 当前最终模型：HiFC 无像素配对 RGB→SAR

当前已经完成并评估的推荐模型是 `hifc_unpaired_conditioned_v1`。它针对 RGB 车辆侧视图
与 SAR ROI 不同坐标、不同采集的情况，只按车型建立弱语义关系，把目标 SAR 的方位角、
俯视角、波段和极化作为 12 维条件；训练中不使用 RGB/SAR 像素级重建或平移对齐。

- [最终完整工作流、架构、loss、梯度路径和复现命令](HIFC_UNPAIRED_FINAL_WORKFLOW_ZH.md)
- [120 epoch / TSTR 48.32% checkpoint 的可复现实验包](repro/hifc_epoch120_tstr48/README_ZH.md)
- [下一阶段优化规划：A0 梯度断路、Meta-TSTR 和稀疏复散射场](HIFC_NEXT_OPTIMIZATION_PLAN_ZH.md)
- [最终训练曲线、流程图、TSTR 图表和原始指标](visualizations/hifc_unpaired_final/README.md)
- [最终模型代码](code/hifc_unpaired_sar_gan.py) 和 [训练入口](code/train_hifc_unpaired_sar_gan.py)

最终 GAN 已完成 120 epoch。独立 TSTR（生成 SAR 训练分类器、真实 X/HH 测试）三 seed
平均为 Top-1 `48.32%`、Top-5 `74.42%`；旧 V1 对照约为 `14.75%/39.06%`。这个
提升说明生成图包含了更多可迁移的真实 SAR 信息，但 native teacher 的近 100% 车型
准确率仍不能替代 TSTR，生成器的域差异和 shortcut 尚未完全消除。旧的
`train_joint_roi_gan.py`、V1/MT1 和各项消融记录继续保留，作为历史实验和对照，不代表
当前最终入口。

历史联合训练入口是 `train_joint_roi_gan.py`。它和早期 `train_bbox.py` 的关键区别是：

- RGB 车辆身份编码器不单独预训练、不冻结，而是在 GAN 每一步中联合更新；随机抽取同车
  另一视图做交叉熵和跨视角特征一致性约束；
- 身份交叉熵是主损失，编码器的 256 维身份特征直接输入生成器。生成损失到身份编码器的
  梯度按当前正确类别置信度逐步放开，使早期训练由身份识别主导；
- 将给定 `SOC_40classes.pth` 重配置并在 64×64 cut ROI 上完整微调；GAN 中冻结这份
  `best.pt`，同时约束生成 ROI 的分类结果和 512 维特征；
- 每个车型的真实 SAR ROI 特征中心从训练集计算并缓存，生成 ROI 向本车型簇中心靠近；
- 结构损失使用同车型、最邻近 30° RGB 方向的 SAR 样本，先在 ±4 像素内做监督目标平移
  对齐，再计算多尺度 L1、梯度和全局 SSIM；
- PatchGAN 判别器的真实性损失、特征匹配和 SAR 强度/边缘统计作为辅助项；生成器在
  前 10 轮输出无散斑结构，随后 5 轮逐步加入可微分乘性散斑，避免分类器利用棋盘格伪纹理。

## HiFC 风格无像素配对入口

论文公开了 HiFC-GAN 的方法，但没有找到作者公开的官方代码仓库。本项目新增了独立的
`hifc_unpaired_conditioned_v1`：保留浅层 LTC 局部纹理对比和深层 SFM 语义映射，RGB 与
SAR 只按车型建立弱语义关系，不做同坐标像素重建；目标 SAR 的方位、俯视角、波段和极化
作为条件输入。完整架构、loss、梯度路径和运行命令见
[`HIFC_UNPAIRED_ADAPTATION_ZH.md`](HIFC_UNPAIRED_ADAPTATION_ZH.md)。

本次正式 8 卡 DDP 续训的预览图、loss 曲线和指标快照见
[`visualizations/hifc_unpaired_ddp`](visualizations/hifc_unpaired_ddp/README.md)。

```bash
DATA_ROOT=/data/newdata/A25_T37_down_大图/A02 \
DEVICE=cuda:0 OUTPUT=runs/hifc_unpaired_all_conditions \
bash run_hifc_unpaired_all.sh
```

代码入口：`hifc_unpaired_sar_gan.py`、`train_hifc_unpaired_sar_gan.py`、
`render_hifc_unpaired_sar.py`；快速张量检查：`python test_hifc_unpaired_sar_gan.py`。

## 1. RGB_15 已暂停

当前联合训练直接读取原始 `RGB`，约定 `1.png=0°、2.png=30°、...、12.png=330°`；
SAR 方位角匹配每个车型实际存在的最近 RGB 视图，缺图时也不会中断。

三维脚本仅作为未采用的实验记录保留；现有重建未达到源视图重投影质量门槛，因此不会写入
`RGB_15`，也不会被训练入口调用。`interpolate_rgb15.py` 同样只保留为已拒绝的二维对照。

## 2. 把旧分类器重配置到原生 64×64

旧权重只用作初始化。权重里的 196 个 token 证明原输入为 14×14 patch，即 224×224。
14×14 的绝对位置编码被二维插值为 4×4，随后 backbone、
位置编码和分类头都在 `SOC_40classes_cut` 上更新；输入始终是直接读取的 64×64 ROI，
不回填到 128×128 或放大到 224×224：

```bash
python finetune_saratrx_64.py \
  --checkpoint "../分类器/SARatrX/model/SOC_40classes.pth" \
  --train-root "../amplitude 8-bit data_地距幅度8位数据.7z/SOC_40classes_cut/train" \
  --test-root "../amplitude 8-bit data_地距幅度8位数据.7z/SOC_40classes_cut/test" \
  --output server_results/saratrx_64_cut \
  --epochs 5 --batch-size 128 --device cuda:0
```

把旧 224 模型当教师、直接放大 cut ROI 评估的 Top-1 只有 18.68%，所以不做知识蒸馏。
原生 64 插值初始化的基线 Top-1 为 8.58%；第一阶段 5 轮达到 69.13%，再用较低
学习率续训 10 轮后达到 77.53% Top-1、95.62% Top-5。后续 GAN 使用的权重是
`server_results/saratrx_64_cut_stage2/best.pt`。

SARATR-X 对照在真实 cut test ROI 上的 Top-1 只有 77.53%，不能作为 GAN 的唯一真实性
判据。因此新增 `train_sar_classifier_64.py`：它只输入 64×64 SAR 强度图，波段、极化、
俯视角与方位仅作为辅助预测标签，绝不作为输入。该原生残差分类器的真实全域 test Top-1
为 94.49%，固定 X/HH/30° 域为 92.31%；权重为
`server_results/sar_native64_multitask_v1/best.pt`。

## 3. 在 SOC_40classes_cut 上联合训练

```bash
python train_joint_roi_gan.py \
  --rgb-root "../amplitude 8-bit data_地距幅度8位数据.7z/RGB" \
  --sar-root "../amplitude 8-bit data_地距幅度8位数据.7z/SOC_40classes_cut/train" \
  --saratrx-checkpoint server_results/saratrx_64_cut_stage2/best.pt \
  --native-classifier-checkpoint server_results/sar_native64_multitask_v1/best.pt \
  --output runs/joint_native_judge_gan \
  --epochs 30 --epoch-size 5000 --batch-size 32 --prototype-batch-size 128 \
  --band X --polarization HH --depression 30 --workers 4 --device cuda:0
```

默认 `--prototype-samples 0` 会使用每个车型的全部真实训练 ROI 计算簇中心，并缓存
为 `output/saratrx_prototypes.pt`。首次试跑可加 `--prototype-samples 64`，正式实验再
删除该参数。真实和生成样本都作为原生 64×64 ROI 送入新的图像分类器；SARATR-X 保留为
独立审计器。双分类器审计能降低生成器只投机单一固定分类器的风险。

主要输出：

- `history.csv`：RGB 身份准确率、生成 ROI 的 SARATR-X 准确率、簇余弦相似度及各项损失；
- `best.pt`：只在完整散斑阶段且 RGB/生成 SAR 身份准确率达标后，按结构、统计和特征
  匹配质量选择；
- `milestone_*.pt`：每 5 轮保存一次，供训练完成后的可审计选择；
- `selected.pt`、`selected.selection.json`：`select_joint_roi_checkpoint.py` 从完整散斑
  里程碑中选择的最终部署权重及选择依据；
- `latest.pt`：可通过 `--resume` 继续；
- `preview_*.png`：每行依次为输入 RGB、方向匹配的真实 SAR、生成 SAR。

训练后在独立 test ROI 上评估：

```bash
python evaluate_joint_roi_gan.py \
  --checkpoint runs/joint_identity_roi_gan/selected.pt \
  --prototype-cache runs/joint_identity_roi_gan/saratrx_prototypes.pt \
  --rgb-root "../amplitude 8-bit data_地距幅度8位数据.7z/RGB" \
  --sar-root "../amplitude 8-bit data_地距幅度8位数据.7z/SOC_40classes_cut/test" \
  --saratrx-checkpoint server_results/saratrx_64_cut_stage2/best.pt
```

评估同时报告 RGB 身份、生成/真实 ROI 的 SARATR-X top-1、类簇余弦、结构损失，
以及判别器对真实/生成 ROI 的平均分，并保存逐车型结果到 `test_metrics.json`。

权重默认优先级为身份识别（`rgb-id-weight=10`、`sar-class-weight=10`、跨视角一致性 2），然后是簇中心
（5）、结构（20）、SAR 统计（5）、真实性（2）和判别器特征匹配（5）。不要只增加分类权重：固定分类器
可能被高频伪纹理欺骗，簇中心和真实性损失正是用来限制这种捷径的。

完整流程可直接复现（若 64×64 分类器已存在则不会重复训练）：

```bash
cd /media/noah/5f655817-37a9-4ac4-9115-d0ba0dab1e2d/home/noah/workspace/DS_datasets/code
PYTHON_BIN=/home/noah/workspace/rgb2sar_direction_gan/.conda/bin/python bash run_joint_roi_gan.sh
```

本次固定 `X/HH/30°` 实验选择 epoch 30。独立 test 上循环取 5000 个样本时，RGB 身份
Top-1 为 100%，生成 SAR 的 64×64 SARATR-X Top-1 为 99.98%，类别簇余弦为
0.99498，结构损失为 0.60305。最终对照图在
`runs/joint_identity_roi_gan/rgb_to_sar_visualization.png`，每行依次为 RGB、匹配真实
SAR、生成 SAR。分类正确和簇相似不能单独证明物理真实性；当前结果用于 cut ROI 的
身份/方向可行性实验，尚不宣称完成全图边界融合或严格电磁散射仿真。

分类器迭代后，旧 GAN 在新分类器上的生成 Top-1 只有 61.94%，证实了旧单分类器约束的
投机风险。使用新分类器微调后，选择 epoch 10：新分类器生成 Top-1 为 99.56%，
SARATR-X 对照 Top-1 为 72.80%，结构误差为 0.5893。对应权重与可视化在
`runs/joint_native_judge_finetune_v1/selected.pt` 和
`runs/joint_native_judge_finetune_v1/rgb_to_sar_visualization.png`。

## 早期方向 CycleGAN demo

这是 ATRNet-STAR 数据的第一阶段 demo：每次只训练一个 RGB 离散方向，生成相同方位角域的 SAR。

## 方法选择

`RGB/<车型>/1.png...12.png` 是车型参考图；SAR 是不同波段、极化、俯视角、场景下的 128×128 幅度图。两者没有逐像素配准和一一对应关系，不能直接作为 pix2pix 的 paired 数据。本项目采用 class-matched unpaired CycleGAN：RGB 与 SAR 保证车型相同、方向角相近，但不假设像素对齐。

默认约定 `1.png=0°、2.png=30°、...、12.png=330°`。RGB 文件没有绝对角 metadata，因此必须用数据提供方说明或人工观察核验。若 `1.png=90°`，传 `--angle-offset 90`。SAR 按环形角距离筛选，默认 ±15°，0° 会正确包含 345°/350°/355° 等样本。

## 使用

在 `Z:\code` 执行：

```powershell
python analyze_data.py --rgb-root "Z:\amplitude 8-bit data_地距幅度8位数据.7z\RGB" --sar-root "Z:\amplitude 8-bit data_地距幅度8位数据.7z\SOC_40classes_cut\train"
python smoke_test.py

# 方向 a 暂取第 1 个 RGB 方向（默认 0°）
python train.py --rgb-root "Z:\amplitude 8-bit data_地距幅度8位数据.7z\RGB" --sar-root "Z:\amplitude 8-bit data_地距幅度8位数据.7z\SOC_40classes_cut\train" --rgb-index 1 --angle-tolerance 15 --output runs\angle_a --epochs 100 --batch-size 8

python infer.py --checkpoint runs\angle_a\latest.pt --input "Z:\amplitude 8-bit data_地距幅度8位数据.7z\RGB\Buick_GL8\1.png" --output runs\angle_a\generated_gl8.png
```

CPU 链路验证可在训练命令后加 `--tiny --image-size 64 --epoch-size 8 --epochs 1 --batch-size 1`。正式实验建议先固定成像条件，例如 `--band X --polarization HH --depression 30`，否则生成器还要同时拟合波段、极化和俯视角差异。

当前实现是“同车型、同方向域”的生成，并非同一次观测的物理严格重建。后续可训练 12 个模型，或把角度编码加入单一条件生成器。
## 当前主线：连续方位角 + 四俯视角的空间条件 ROI SAR 生成

本阶段固定 **X/HH**，不再只训练/验证 30° 俯视角。训练集同时包含 15°、30°、45°、60°，目标方位角以 `sin/cos` 连续条件输入；训练时对相邻 5° 方位角施加低频连续性约束，因此可渲染 7.5°、22.5° 等训练集中没有的方位角。极化和波段暂不扩展。

生成器不再只接收全局身份向量：RGB 编码器输出 64/32/16/8 像素四层特征，逐层注入 SAR 解码器（FPN/U-Net 式跳连），以保留车身轮廓、比例和源视角姿态。真实 SAR 以同一车型、目标方位角和俯视角的 ROI 弱配对，用于结构、统计与物理启发先验；不假设 RGB 与 SAR 像素严格配准。真实 SAR 标注框宽高会从生成条件中剔除，避免其成为车型捷径。

```bash
PYTHON_BIN=/home/noah/workspace/rgb2sar_direction_gan/.conda/bin/python \
bash run_continuous_spatial_roi_gan.sh
```

训练后，`test_audit.json` 会分别给出四个俯视角的真实/生成 ROI 64×64 分类准确率。独立连续渲染示例：

```bash
python render_continuous_spatial_sar.py \
  --gan-checkpoint runs/continuous_spatial_x_hh/best.pt \
  --rgb-root "../amplitude 8-bit data_地距幅度8位数据.7z/RGB" \
  --class-name Buick_GL8 --source-angle 0 \
  --target-angles "7.5,22.5,37.5,52.5" --output runs/interpolated_azimuths.png
```
