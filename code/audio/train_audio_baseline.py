# -*- coding: utf-8 -*-
"""语音单模态分类基线：冻结 chinese-wav2vec2-base 时序均值特征 + MLP 分类头。

与文本 / 视觉基线（code/text、code/vision）完全同一套训练协议：
    训练集标准化、类别加权交叉熵、valid Macro-F1 早停（patience 10）、
    test 上报 Accuracy / Macro-F1 / 混淆矩阵；另附 sklearn LogisticRegression 对照。
仅输入维度不同（768），保证模态间可比。

用法：
    python code/audio/train_audio_baseline.py [--feat outputs/features/audio_cls.pt]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

SEED = 42
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLPHead(nn.Module):
    def __init__(self, dim=768, hidden=256, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def summarize(y_true, y_pred):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'macro_f1': round(f1_score(y_true, y_pred, average='macro'), 4),
        'per_class_f1': dict(zip(CLASS_NAMES, [
            round(v, 4) for v in f1_score(y_true, y_pred, average=None)])),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', default=str(Path(__file__).resolve().parents[2]
                                            / 'outputs' / 'features' / 'audio_cls.pt'))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2]
                                         / 'outputs' / 'results' / 'audio_baseline'))
    args = ap.parse_args()
    set_seed()

    pack = torch.load(args.feat, weights_only=False)
    splits = pack['splits']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    Xtr, ytr = splits['train']['feat'], splits['train']['label3']
    Xva, yva = splits['valid']['feat'], splits['valid']['label3']
    Xte, yte = splits['test']['feat'], splits['test']['label3']

    mean, std = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xva, Xte = ((Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std)
    Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    Xva_d, yva_np = Xva.to(dev), yva.numpy()
    Xte_d = Xte.to(dev)

    counts = torch.bincount(ytr, minlength=3).float()
    w = counts.sum() / (3.0 * counts)
    print('训练集类别计数:', counts.tolist(), '损失权重:', [round(x, 3) for x in w.tolist()])

    model = MLPHead().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(weight=w.to(dev))

    best_va_f1, best_state, patience_left, best_ep = -1.0, None, args.patience, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva_d).argmax(1).cpu().numpy()
        va_f1 = f1_score(yva_np, pred, average='macro')
        if va_f1 > best_va_f1:
            best_va_f1, best_state, patience_left, best_ep = va_f1, {
                k: v.clone() for k, v in model.state_dict().items()}, args.patience, ep
        else:
            patience_left -= 1
        if ep % 10 == 0 or patience_left == 0:
            print(f'epoch {ep:>3}  valid macro-F1 = {va_f1:.4f}  (best {best_va_f1:.4f})')
        if patience_left == 0:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te = model(Xte_d).argmax(1).cpu().numpy()
    mlp = summarize(yte, pred_te)
    mlp['best_valid_macro_f1'] = round(best_va_f1, 4)
    mlp['best_epoch'] = best_ep

    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(Xtr.cpu().numpy(), ytr.cpu().numpy())
    pred_lr = lr.predict(Xte.cpu().numpy())
    logreg = summarize(yte, pred_lr)

    # 质量信号只做描述统计与正确率对照，不参与本基线训练；供后续门控实验参考
    q = splits['test']
    sf = q['speech_frac'].numpy()
    correct = (pred_te == yte.numpy()).astype(float)
    q_low, q_high = sf <= np.median(sf), sf > np.median(sf)
    quality = {
        'speech_frac_mean': round(float(sf.mean()), 4),
        'speech_frac_median': round(float(np.median(sf)), 4),
        'snr_db_median': round(float(q['snr_db'].median()), 2),
        'dur_sec_mean': round(float(q['dur_sec'].mean()), 3),
        'point_biserial_speech_frac_vs_correct': round(float(
            np.corrcoef(sf, correct)[0, 1]), 4),
        'accuracy_low_speech_frac_half': round(float(correct[q_low].mean()), 4),
        'accuracy_high_speech_frac_half': round(float(correct[q_high].mean()), 4),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'task': 'CH-SIMS 语音单模态三分类（chinese-wav2vec2-base 冻结特征，时序均值池化）',
        'feature': {'model': pack.get('model'), 'dim': pack.get('dim'),
                    'pooling': pack.get('pooling')},
        'seed': SEED,
        'split': {'train': len(Xtr), 'valid': len(Xva), 'test': len(Xte)},
        'mlp_head': mlp,
        'logistic_regression': logreg,
        'test_quality_diagnostics': quality,
    }
    with open(out_dir / 'audio_baseline_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savetxt(out_dir / 'confusion_matrix_mlp.csv', np.array(mlp['confusion_matrix']),
               fmt='%d', delimiter=',')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('结果已保存到:', out_dir)


if __name__ == '__main__':
    main()
