# BallShow TransReID 模型层文档 (Model Layer)

## 1. 算法架构与核心技术
本项目采用基于 Vision Transformer (ViT) 的 **TransReID** 架构作为核心特征提取器，专门针对篮球比赛中球员姿态变化大、遮挡严重的问题进行了深度优化。

### 核心特性
- **双分辨率特征融合 (Dual-ViT Ensemble)**
  - 系统同时加载两个不同输入分辨率的 ViT 模型（`256×128` 和 `384×128`）。
  - 在推理阶段（`reid_engine.py`），对同一图像分别提取特征，并使用 $L_2$ 正则化后进行加权融合（默认为 1:1 等权），显著提升了特征的鲁棒性和多尺度表达能力。
- **Jigsaw Patch Module (JPM)**
  - 引入 JPM 模块，通过将图像 patch 进行移位和打乱（Shift and Shuffle），强迫模型学习更加细粒度和鲁棒的局部特征，有效对抗比赛中的部分遮挡。
- **Side Information Embedding (SIE)**
  - 启用了 Camera ID 的 SIE（`SIE_CAMERA: True`），将摄像头视角信息编码到 Transformer 的输入位置编码中，缓解了不同机位造成的视角偏差。
- **Neck 与 Loss 优化**
  - **BNNeck**: 在特征层和分类层之间加入 Batch Normalization，使得度量学习（Triplet Loss/Circle Loss）和分类学习（Softmax Loss）的优化空间相互独立且稳定。
  - **联合损失**: 使用了 ID Loss (Softmax) + Metric Loss (Triplet/Circle) + Center Loss联合训练。

## 2. 训练与配置 (Config)
模型训练的配置文件主要位于 `configs/BallShow/vit_transreid_4090.yml`。

- **预训练权重**: 使用 ImageNet 预训练的 ViT-Base (`jx_vit_base_p16_224-80ecf9dd.pth`)
- **超参数设定**:
  - `BASE_LR`: 0.010 (配合 SGD 优化器和 Linear Warmup)
  - `IMS_PER_BATCH`: 72 (针对 RTX 4090 24GB 显存优化，384×128 分辨率下)
  - `MAX_EPOCHS`: 120
  - `CENTER_LOSS_WEIGHT`: 0.0005
- **数据增强**: Random Horizontal Flip (`PROB: 0.5`) + Random Erasing (`RE_PROB: 0.5`)

## 3. 推理阶段 (Inference)
在推理阶段（部署在 FastAPI 的 `reid_engine.py` 中）：
- `ReIDEngine` 类封装了完整的管道。
- 将图片转换为 `RGB` 格式并缩放至 `384x128`。
- 将图片分别送入两个独立加载的 ViT 模型，特征拼接/加权后使用 `F.normalize(p=2)` 进行 L2 归一化，输出 `768` 维度的特征向量。
- 检索时使用全矩阵乘法计算**余弦相似度** (Cosine Similarity)，并进行 Top-K 排序。

## 4. 目录结构
```text
TransReID-master/
├── configs/             # 模型配置文件 (YML)
│   └── BallShow/        # 针对本项目数据集优化的配置
├── datasets/            # 数据集加载与预处理逻辑
├── loss/                # 损失函数定义 (Triplet, Center, Circle 等)
├── model/               # TransReID 模型网络结构定义
├── processor/           # 训练 (do_train) 循环与测试逻辑
├── solver/              # 优化器与学习率调度器
├── train.py             # 分布式/单卡训练入口脚本
└── test.py              # 模型验证与评估脚本
```

## 5. 性能指标
当前部署的双模型配置在测试集上可达到：
- **Rank-1 Accuracy**: ~94.4%
- **mAP (Mean Average Precision)**: ~91.8%
