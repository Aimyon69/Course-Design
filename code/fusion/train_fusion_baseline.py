# -*- coding: utf-8 -*-
"""三路拼接融合基线与标准 LMF 复现。

实验设置（方案第 4–5 周）：
    - 输入：文本 768 + 视觉 512 + 语音 768 的冻结片段级特征，按 sample_id 严格对齐；
    - 两种融合方法使用同一训练/验证/测试划分、同一三路编码器、同一训练预算；
    - 每个模态在训练集上做 z-score 标准化（均值/方差只取自 train）；
    - 类别加权交叉熵、Adam lr 1e-3 / wd 1e-4、batch 128、
      按 valid Macro-F1 早停（patience 10）、seed 42，与单模态基线一致。

方法：
    concat —— 三路特征直接拼接为 2048 维，接 2048→64→3 MLP（dropout 0.3）。
              隐层 64 与 LMF 的约 13 万参数大致匹配，避免用更大的分类头
              把差异归因于参数量。
    lmf    —— Low-rank Multimodal Fusion（Liu et al., ACL 2018），按官方实现：
              每路先过预融合子网（BatchNorm → Dropout → 3 层 ReLU MLP），
              输入尾部拼接常数 1，再乘模态专属低秩因子 (rank, hidden+1, n_classes)，
              三路结果逐元素相乘、按 rank 加权求和并加偏置，直接输出三类 logits。
              因子与融合权重用 xavier_normal 初始化，融合偏置置零。

    LMF 的文本输入在本项目中是 bert-base-chinese 的池化 [CLS] 向量而非词序列，
    因此文本子网与音视频子网一样用 MLP（官方实现中文本分支为 LSTM）。
    hidden 维度按 valid 选取为 text 64 / vision 16 / audio 64；MMSA 的 CH-SIMS
    配置针对其预提取特征（audio 33 / video 709 维），直接套用到本项目的
    CLIP 512 与 Wav2Vec2 768 维特征上不稳定。

用法：
    python code/fusion/train_fusion_baseline.py            # 依次跑 concat 与 lmf
    python code/fusion/train_fusion_baseline.py --method lmf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_data import load_fusion_features

SEED = 42
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
MODALITIES = ('text', 'vision', 'audio')


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConcatFusion(nn.Module):
    """早期拼接：三路特征 concat 后接 MLP 分类头。"""

    def __init__(self, dims, hidden=64, n_classes=3, dropout=0.3):
        super().__init__()
        in_dim = sum(dims.values())
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, feats):
        return self.net(torch.cat([feats[m] for m in MODALITIES], dim=1))


class SubNet(nn.Module):
    """LMF 预融合子网（原论文 audio/video SubNet 的结构）。

    BatchNorm1d → Dropout → Linear+ReLU ×3。
    """

    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(dropout)
        self.l1 = nn.Linear(in_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.drop(self.norm(x))
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return torch.relu(self.l3(x))


class LMFFusion(nn.Module):
    """标准 LMF：预融合子网 + 模态专属低秩因子 + 逐元素乘法 + rank 加权求和。

    z_m = W_m [1; h_m] ∈ R^{rank × n_classes}   （[1; h] 为尾部补常数 1）
    h   = sum_r w_r * prod_m z_{m,r} + b
    y   = h（直接作为三类 logits）
    """

    def __init__(self, dims, hidden_dims, rank=4, n_classes=3, dropout=0.2):
        super().__init__()
        self.rank = rank
        self.hidden_dims = hidden_dims
        self.subnets = nn.ModuleList([
            SubNet(dims[m], hidden_dims[m], dropout) for m in MODALITIES])
        self.factors = nn.ParameterList([
            nn.Parameter(torch.empty(rank, hidden_dims[m] + 1, n_classes))
            for m in MODALITIES])
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.empty(1, n_classes))
        for p in self.factors:
            nn.init.xavier_normal_(p)
        nn.init.xavier_normal_(self.fusion_weights)
        nn.init.zeros_(self.fusion_bias)

    def forward(self, feats):
        hs = [sub(feats[m]) for sub, m in zip(self.subnets, MODALITIES)]
        ones = torch.ones(hs[0].size(0), 1, device=hs[0].device, dtype=hs[0].dtype)
        # [B, hidden+1] @ [rank, hidden+1, n_classes] -> [rank, B, n_classes]
        zs = [torch.matmul(torch.cat([ones, h], dim=1), f)
              for h, f in zip(hs, self.factors)]
        zy = zs[0] * zs[1] * zs[2]                  # [rank, B, n_classes]
        out = torch.einsum('rbc,r->bc', zy, self.fusion_weights.squeeze(0))
        return out + self.fusion_bias               # [B, n_classes]


def standardize(data):
    """按 train 统计量标准化每个模态；返回新的 splits（不改原字典）。"""
    stats = {}
    for m in MODALITIES:
        x = data['splits']['train']['feats'][m]
        stats[m] = (x.mean(0), x.std(0).clamp_min(1e-6))
    out = {}
    for mode in ('train', 'valid', 'test'):
        src = data['splits'][mode]
        feats = {m: (src['feats'][m] - stats[m][0]) / stats[m][1]
                 for m in MODALITIES}
        out[mode] = {**src, 'feats': feats}
    return out, stats


def summarize(y_true, y_pred):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'macro_f1': round(f1_score(y_true, y_pred, average='macro'), 4),
        'per_class_f1': dict(zip(CLASS_NAMES, [
            round(v, 4) for v in f1_score(y_true, y_pred, average=None)])),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def train_one(method, data, dims, args, dev):
    set_seed(args.seed)
    if method == 'concat':
        model = ConcatFusion(dims, hidden=args.hidden, dropout=args.dropout).to(dev)
        config = {'hidden': args.hidden, 'dropout': args.dropout}
    elif method == 'lmf':
        hidden_dims = dict(zip(MODALITIES, args.lmf_hidden))
        model = LMFFusion(dims, hidden_dims=hidden_dims, rank=args.rank,
                          dropout=args.lmf_dropout).to(dev)
        config = {'hidden_dims': hidden_dims, 'rank': args.rank,
                  'dropout': args.lmf_dropout}
    else:
        raise ValueError(method)

    tr, va, te = data['train'], data['valid'], data['test']
    tr_feats = {m: tr['feats'][m].to(dev) for m in MODALITIES}
    va_feats = {m: va['feats'][m].to(dev) for m in MODALITIES}
    te_feats = {m: te['feats'][m].to(dev) for m in MODALITIES}
    ytr, yva, yte = tr['label3'].to(dev), va['label3'], te['label3']
    counts = torch.bincount(ytr, minlength=3).float()
    weight = counts.sum() / (3.0 * counts)
    crit = nn.CrossEntropyLoss(weight=weight.to(dev))
    if method == 'lmf' and args.factor_lr:
        factor_params = (list(model.factors)
                         + [model.fusion_weights, model.fusion_bias])
        fids = {id(p) for p in factor_params}
        opt = torch.optim.Adam([
            {'params': [p for p in model.parameters() if id(p) not in fids],
             'lr': args.lr},
            {'params': factor_params, 'lr': args.factor_lr},
        ], weight_decay=args.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.wd)
    print(f'[{method}] 参数 {sum(p.numel() for p in model.parameters()):,} | '
          f'类别计数 {counts.tolist()} | 损失权重 {[round(w, 3) for w in weight.tolist()]}')

    best_f1, best_state, best_ep, left = -1.0, None, 0, args.patience
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(ytr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = crit(model({m: tr_feats[m][idx] for m in MODALITIES}),
                        ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(va_feats)
            pred = pred.argmax(1).cpu().numpy()
        va_f1 = f1_score(yva.numpy(), pred, average='macro')
        if va_f1 > best_f1:
            best_f1, best_ep, left = va_f1, ep, args.patience
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            left -= 1
        if ep % 10 == 0 or left == 0:
            print(f'  epoch {ep:>3} valid macro-F1 = {va_f1:.4f} '
                  f'(best {best_f1:.4f})')
        if left == 0:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(te_feats)
        pred_te = logits.argmax(1).cpu().numpy()
    result = summarize(yte.numpy(), pred_te)
    result['best_valid_macro_f1'] = round(best_f1, 4)
    result['best_epoch'] = best_ep
    return model, result, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['concat', 'lmf', 'both'], default='both')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=64, help='concat 隐层维度')
    ap.add_argument('--dropout', type=float, default=0.3, help='concat dropout')
    ap.add_argument('--rank', type=int, default=4, help='LMF 秩')
    ap.add_argument('--lmf-hidden', type=int, nargs=3, default=[64, 16, 64],
                    metavar=('TEXT', 'VISION', 'AUDIO'),
                    help='LMF 各模态预融合 hidden 维度')
    ap.add_argument('--lmf-dropout', type=float, default=0.2)
    ap.add_argument('--factor-lr', type=float, default=None,
                    help='LMF 因子单独学习率；默认与 --lr 相同')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2]
                                         / 'outputs' / 'results' / 'fusion_baseline'))
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw = load_fusion_features()
    data, norm_stats = standardize(raw)
    dims = raw['dims']
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ['concat', 'lmf'] if args.method == 'both' else [args.method]
    comparison = {}
    for method in methods:
        model, result, config = train_one(method, data, dims, args, dev)
        payload = {
            'task': f'CH-SIMS 三模态融合（{method}）',
            'method': method,
            'config': config,
            'seed': args.seed,
            'split': {k: len(data[k]['sample_id']) for k in ('train', 'valid', 'test')},
            'feature_dims': dims,
            'feature_models': raw['model_names'],
            'n_params': sum(p.numel() for p in model.parameters()),
            **result,
        }
        with open(out_dir / f'{method}_result.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        np.savetxt(out_dir / f'{method}_confusion_matrix.csv',
                   np.array(result['confusion_matrix']), fmt='%d', delimiter=',')
        torch.save({
            'method': method, 'config': config, 'dims': dims,
            'feature_models': raw['model_names'],
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'norm': {m: {'mean': norm_stats[m][0].cpu(),
                         'std': norm_stats[m][1].cpu()} for m in MODALITIES},
        }, out_dir / f'{method}_best.pt')
        comparison[method] = {k: payload[k] for k in
                              ('accuracy', 'macro_f1', 'per_class_f1',
                               'best_valid_macro_f1', 'best_epoch', 'n_params')}
        print(f'[{method}] test: Acc {result["accuracy"]:.4f} '
              f'Macro-F1 {result["macro_f1"]:.4f} | 混淆矩阵 {result["confusion_matrix"]}')

    with open(out_dir / 'comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print('结果已保存到:', out_dir)


if __name__ == '__main__':
    main()
