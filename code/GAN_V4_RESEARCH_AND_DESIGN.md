# v4 快速 RGB→SAR GAN：调研、基线与设计

## 为什么从 diffusion 切换为 GAN

在 Quadro RTX 6000、batch=1、包含 RGB 编码和 SAR 解码的本机基准中：

| 模型 | 延迟 |
|---|---:|
| v3 diffusion，1 步 | 4.98 ms |
| v3 diffusion，20 步 | 48.83 ms |
| v3 diffusion，50 步 | 117.47 ms |
| 旧单前向 GAN | 2.57 ms |
| v4 SPADE 生成器 | **5.13 ms** |

v4 只执行一次生成器前向传播，约 195 FPS；比 v3 的 50-step diffusion 快约 23 倍。

## 文献调研与基线选择

### 未采用 GigaGAN 作为直接代码基线

[GigaGAN](https://arxiv.org/abs/2303.05511) 展示了 GAN 在大规模文本到图像任务中快速采样的潜力，并报告 512px 图像约 0.13 秒推理。但它为超大规模训练数据和文本条件设计，直接套用到当前 12k 张、64×64、RGB→SAR 的数据集会导致模型容量与数据规模严重不匹配。因此它是“速度可行性”的参考，不是工程上的直接基线。

### 选用 SPADE 条件 GAN 作为直接基线

[SPADE](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html) 的关键思想是在生成器每一层利用空间条件图生成归一化仿射参数，而不是只在输入层拼接条件。它非常匹配本任务：RGB 外形、车身比例和视角是空间信息，不能被全局身份向量压缩成车型模板。

类别条件判别器借鉴 [BigGAN](https://arxiv.org/abs/1809.11096) 的条件生成思想；多尺度 PatchGAN 保持局部散射纹理约束。对于未严格配准的跨模态翻译，CUT/DCLGAN 的 patch contrastive 思路也值得用于 v4.1 的附加损失，而不应使用强逐像素 L1。[DCLGAN/CUT 比较](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Han_Dual_Contrastive_Learning_for_Unsupervised_Image-to-Image_Translation_CVPRW_2021_paper.html)

SAR 与光学跨模态研究也指出，GAN 翻译方法依赖配准关系；当前 RGB/SAR 没有严格像素配准，因此 v4 不把像素重建作为主目标。[SAR-to-optical reciprocal GAN](https://arxiv.org/abs/1901.03749)

## v4 架构

```
RGB 128×128 ─→ 4级 RGB-FPN (64/32/16/8)
                    │
类别 + [目标方位角、俯视角、RGB源视角] ─→ FiLM
                    │
随机 style z ─→ 4×4 latent ─→ SPADE ResBlocks ─→ SAR 64×64
                                      ↑ 每一级注入对应 RGB 空间特征

真实/生成 SAR + 类别 + 几何 ─→ 双尺度 Projection PatchGAN
```

### 相比旧 GAN 的具体改进

1. **SPADE 而非只用全局 identity 向量**：RGB FPN 特征在 8、16、32、64 输出尺度全部注入；车体轮廓不会在最初层被遗忘。
2. **判别器看见类别和几何条件**：旧判别器只能判断“像不像 SAR”，v4 的 projection discriminator 判断“是否是当前类别、方位角和俯视角下的 SAR”。
3. **错配真实 SAR hard negative**：真实图配错类别/几何也被当作负样本，阻止生成器用通用 SAR 模板骗过判别器。
4. **真实 SAR latent 流形约束**：冻结 v3 SAR autoencoder，以真实目标 latent 和 class×depression prototype 约束生成图；这是可微的真实 SAR 域约束，不依赖二次散斑模拟。
5. **较小的分类损失**：冻结真实 SAR 分类器只作为辅助，不再把分类准确率作为生成器的主导目标。
6. **部署优先**：生成器不使用 spectral normalization；谱归一化仅保留在判别器训练分支，减少推理延迟。

## 代码与训练

- `v4_spade_gan.py`：SPADE 生成器、RGB 条件 FPN、双尺度 projection PatchGAN。
- `train_v4_spade_gan.py`：严格 train/validation 训练、冻结 SAR latent 约束、错配负样本、AMP。
- 训练开始前会从原训练集固定划分 validation；原 test 集不参与 GAN 选权重。

建议先以 30 epoch 验证稳定性与速度，再根据 validation latent 距离、独立生成分类器和可视化决定是否扩展到 100 epoch。最终仍使用 generated-only classifier 在未参与选择的真实 test SAR 上审计。
