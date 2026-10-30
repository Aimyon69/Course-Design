# 多模态情感识别（CH-SIMS）

三路（文本 / 视觉 / 语音）中文情感三分类实验与实时演示的代码与实验记录。
三条支路都使用冻结的预训练编码器提取片段级特征，再比较拼接融合、标准 LMF
和质量门控 LMF；目标是可复现的实验、模态缺失/退化下的鲁棒性分析，
以及后续的实时系统。

## 数据与特征

- **数据集**：CH-SIMS 原始视频与 `label.csv`，官方划分
  train 1368 / valid 456 / test 457。
- **标签**：多模态连续标签 `label_M` 以 0 为阈值离散化为
  Negative / Neutral / Positive；仓库代码会断言 `annotation` 列与阈值规则一致。
- **编码器（全部冻结）**：

| 模态 | 编码器 | 特征 |
|---|---|---|
| 文本 | `bert-base-chinese` | `[CLS]`，768 维 |
| 视觉 | `openai/clip-vit-base-patch32` | 8 帧人脸均值，512 维 |
| 语音 | `TencentGameMate/chinese-wav2vec2-base` | 时序均值，768 维 |

数据目录约定（不随仓库上传）：

```
dataset/
├── label.csv
├── video_0001/0001.mp4
├── video_0001/0002.mp4
└── ...
```

特征缓存默认写到 `outputs/features/`，同样不入库（体积大，且受数据集许可限制）。

## 仓库结构

```
code/
├── common/chsims.py                 # CH-SIMS 统一读取、划分、标签规则
├── text/
│   ├── extract_text_features.py     # BERT [CLS] 特征
│   └── train_text_baseline.py
├── vision/
│   ├── extract_vis_features.py      # 人脸检测 + CLIP 特征
│   └── train_vis_baseline.py
├── audio/
│   ├── extract_audio_features.py    # 解码 + Wav2Vec2 特征 + 质量指标
│   └── train_audio_baseline.py
└── fusion/
    ├── fusion_data.py               # 三路对齐、z-score、质量向量
    ├── train_fusion_baseline.py     # 拼接 + MLP / 标准 LMF
    ├── build_perturbed_features.py  # 原始输入扰动缓存（噪声/遮挡/替换）
    ├── train_fusion_gated.py        # 质量门控 LMF + 扰动训练 + 鲁棒性评测
    ├── analyze_gated.py             # 单 seed 结果表
    └── analyze_seeds.py             # 多 seed 均值±标准差
doc/                                 # 方案、阶段记录、总结
```

## 环境

```bash
conda create -n host python=3.10 -y
conda activate host
pip install -r requirements.txt
```

实验在 WSL + CUDA（RTX 5090）上完成；CPU 也可运行小规模推理，
但特征提取会明显变慢。`ffmpeg` 由 `imageio-ffmpeg` 自带，无需系统安装。

## 复现流程

### 1. 提取三路特征

```bash
python code/text/extract_text_features.py
python code/vision/extract_vis_features.py
python code/vision/extract_vis_features.py --finalize
python code/audio/extract_audio_features.py
python code/audio/extract_audio_features.py --finalize
```

### 2. 单模态基线

```bash
python code/text/train_text_baseline.py
python code/vision/train_vis_baseline.py
python code/audio/train_audio_baseline.py
```

### 3. 拼接融合与标准 LMF

```bash
python code/fusion/train_fusion_baseline.py --method both --seed 42
```

### 4. 质量门控 LMF 与鲁棒性实验

```bash
# 训练增强缓存（offset 0）
python code/fusion/build_perturbed_features.py --modality text
python code/fusion/build_perturbed_features.py --modality audio
python code/fusion/build_perturbed_features.py --modality vision

# 评测缓存（offset 1，换随机实现，避免泄漏）
python code/fusion/build_perturbed_features.py --modality text --splits test --seed-offset 1
python code/fusion/build_perturbed_features.py --modality audio --splits test --seed-offset 1
python code/fusion/build_perturbed_features.py --modality vision --splits test --seed-offset 1

# 四个对照 × 三个 seed
for s in 42 1 7; do
  out=outputs/results/fusion_gated
  [ $s -ne 42 ] && out=$out/seed$s
  python code/fusion/train_fusion_gated.py \
    --methods lmf lmf_pert lmf_gate lmf_gate_pert \
    --seed $s --patience 30 --epochs 200 --out $out
done

python code/fusion/analyze_gated.py
python code/fusion/analyze_seeds.py
```

完整命令、消融和逐条件表见
[`doc/实验记录_质量门控LMF.md`](doc/实验记录_质量门控LMF.md)。

## 主要结果

test Macro-F1（457 条）。第 4 阶段单 seed；第 5 阶段为 seed 42/1/7 均值。

| 方法 | 干净 | 视觉遮挡 60% | 语音 0 dB | 文本 30% 替换 |
|---|---|---|---|---|
| 拼接 + MLP | **0.6673** | 0.4967 | **0.6479** | **0.5692** |
| 标准 LMF | 0.6032 | 0.5281 | 0.5906 | 0.5183 |
| LMF + 门控 + 扰动训练 | 0.5834 | **0.5731** | 0.5745 | 0.4663 |

- 门控 + 扰动训练在**视觉退化**和**音视频组合退化**下稳定优于标准 LMF：
  occ60 绝对 F1 +0.045，退化幅度从 7.5 个百分点降到约 1.0 个百分点；
  组合退化 +0.047。
- 文本 / 语音退化下没有优势；干净输入低约 0.020。
- 关键负结果：门控值几乎不随质量变化，逐样本最优门控与 `face_frac` 不相关。
  收益来自 `present` 掩码、模态 dropout 和受扰训练，而不是质量自适应降权。
  详见 [`doc/质量门控LMF_阶段总结.md`](doc/质量门控LMF_阶段总结.md)。

## 实验记录

- [`doc/多模态实时情感识别_稳妥实施方案.md`](doc/多模态实时情感识别_稳妥实施方案.md)
- [`doc/实验记录_文本模态.md`](doc/实验记录_文本模态.md)
- [`doc/实验记录_视觉模态.md`](doc/实验记录_视觉模态.md)
- [`doc/实验记录_语音模态.md`](doc/实验记录_语音模态.md)
- [`doc/实验记录_融合基线.md`](doc/实验记录_融合基线.md)
- [`doc/实验记录_质量门控LMF.md`](doc/实验记录_质量门控LMF.md)
- [`doc/质量门控LMF_阶段总结.md`](doc/质量门控LMF_阶段总结.md)

## 说明

- `dataset/`、`outputs/features/` 和模型权重（`*.pt`）不纳入版本控制；
  需要按上面的命令在本地重新生成。
- CH-SIMS 的使用受其原始许可约束，请从官方渠道获取数据。
- 当前结果是小样本（test 457 条）上的单数据集结论，seed/早停波动约 ±0.02；
  逐条件差异应结合 `multi_seed_summary.md` 中的标准差阅读。
