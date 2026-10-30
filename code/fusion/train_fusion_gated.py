# -*- coding: utf-8 -*-
"""质量门控 LMF：结构改进 + 扰动训练的消融与鲁棒性实验。

四个对照（同一划分、同一编码器、同一训练预算）：

    lmf            标准 LMF（干净数据训练）
    lmf_pert       LMF + 扰动训练（无门控）
    lmf_gate       LMF + 质量门控（干净数据训练）
    lmf_gate_pert  LMF + 质量门控 + 扰动训练（完整方案）

门控（方案第 3 节）：
    每一路先得到低秩投影 p_m（[rank, B, C]），再计算标量 g_m ∈ [0,1]：
        p'_m = 1 + g_m * (p_m - 1)
    g_m=0 时该路成为逐元素乘法的单位元，g_m=1 时退化为原始 LMF。
    g_m 由该路质量向量经小 MLP + sigmoid 得到；模态缺失时 present=0，直接 g_m=0。

质量向量（训练、评测、演示同口径）：
    text    [1]（存在标记另行用 present 表示；离线实验中没有可用的 ASR 置信度）
    vision  [face_frac]
    audio   [speech_frac, snr_db/60]

扰动训练（原始输入上加扰，再走冻结编码器，见 build_perturbed_features.py）：
    vision  人脸遮挡 20% / 30% / 40%
    audio   白噪声 SNR 10 / 5 dB
    text    15% 汉字随机替换
    每个模态以 p_miss=0.08 置为缺失（特征置零、present=0）；
    否则以 p_pert=0.40 从该模态的训练扰动中随机取一个条件。

评测条件包含训练中见过的强度，也包含未见过强度（SNR 0 dB、遮挡 60%、
文本 10/20/30%）以及缺失与组合扰动。评测扰动使用不同随机种子生成，
避免与训练增强共用同一噪声实现。

用法：
    python code/fusion/train_fusion_gated.py --seed 42
    python code/fusion/train_fusion_gated.py --methods lmf_gate_pert
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
from train_fusion_baseline import (ConcatFusion, LMFFusion, SubNet, set_seed,
                                   summarize)

ROOT = Path(__file__).resolve().parents[2]
PERT_DIR = ROOT / 'outputs' / 'features' / 'perturbed'
EVAL_DIR = ROOT / 'outputs' / 'features' / 'perturbed_eval'
MODALITIES = ('text', 'vision', 'audio')
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']

TRAIN_VARIANTS = {
    'text': ('corr15',),
    'vision': ('occ20', 'occ30', 'occ40'),
    'audio': ('snr10', 'snr5'),
}
# 评测需要全部条件；训练用不到的强度在这里也要能加载
ALL_VARIANTS = {
    'text': ('corr10', 'corr15', 'corr20', 'corr30'),
    'vision': ('occ20', 'occ30', 'occ40', 'occ60'),
    'audio': ('snr10', 'snr5', 'snr0'),
}

EVAL_CONDITIONS = [
    ('clean', {}),
    ('audio_snr10', {'audio': 'snr10'}),
    ('audio_snr5', {'audio': 'snr5'}),
    ('audio_snr0', {'audio': 'snr0'}),
    ('vision_occ20', {'vision': 'occ20'}),
    ('vision_occ40', {'vision': 'occ40'}),
    ('vision_occ60', {'vision': 'occ60'}),
    ('text_corr10', {'text': 'corr10'}),
    ('text_corr20', {'text': 'corr20'}),
    ('text_corr30', {'text': 'corr30'}),
    ('missing_text', {'text': 'missing'}),
    ('missing_vision', {'vision': 'missing'}),
    ('missing_audio', {'audio': 'missing'}),
    ('missing_text+vision', {'text': 'missing', 'vision': 'missing'}),
    ('missing_text+audio', {'text': 'missing', 'audio': 'missing'}),
    ('missing_vision+audio', {'vision': 'missing', 'audio': 'missing'}),
    ('audio_snr5+vision_occ40',
     {'audio': 'snr5', 'vision': 'occ40'}),
]


# --------------------------------------------------------------------------- #
# 质量门控
# --------------------------------------------------------------------------- #
class QualityGate(nn.Module):
    """质量向量 → 标量门控 g ∈ [0,1]。"""

    def __init__(self, in_dim, hidden=8, bias_init=2.0, init_std=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # 末层零初始化会让第一层在第一步拿不到梯度，质量编码器几乎不学习；
        # init_std>0 时改用小随机初始化，保留 g≈sigmoid(bias) 的初值但允许
        # 质量相关的梯度从训练开始就回传到第一层。
        if init_std > 0:
            nn.init.normal_(self.net[2].weight, std=init_std)
        else:
            nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, bias_init)

    def forward(self, q):
        return torch.sigmoid(self.net(q)).squeeze(-1)


class GatedLMFFusion(LMFFusion):
    """在标准 LMF 的低秩投影上加入可退化门控。"""

    GATE_DIMS = {'text': 1, 'vision': 1, 'audio': 2}

    def __init__(self, dims, hidden_dims, rank=4, n_classes=3, dropout=0.2,
                 gate_hidden=8, gate_init=0.0):
        super().__init__(dims, hidden_dims, rank=rank,
                         n_classes=n_classes, dropout=dropout)
        self.gates = nn.ModuleDict({
            m: QualityGate(self.GATE_DIMS[m], gate_hidden, init_std=gate_init)
            for m in MODALITIES})

    def forward(self, feats, quality, present, return_gates=False):
        hs = [sub(feats[m]) for sub, m in zip(self.subnets, MODALITIES)]
        ones = torch.ones(hs[0].size(0), 1,
                          device=hs[0].device, dtype=hs[0].dtype)
        zs = [torch.matmul(torch.cat([ones, h], dim=1), f)
              for h, f in zip(hs, self.factors)]
        gates = {}
        gated = []
        for m, z in zip(MODALITIES, zs):
            g = present[m] * self.gates[m](quality[m])     # [B]
            gates[m] = g
            gated.append(1.0 + g.view(1, -1, 1) * (z - 1.0))
        zy = gated[0] * gated[1] * gated[2]
        out = torch.einsum('rbc,r->bc', zy,
                           self.fusion_weights.squeeze(0)) + self.fusion_bias
        return (out, gates) if return_gates else out


def build_model(method, dims, args):
    hidden_dims = dict(zip(MODALITIES, args.lmf_hidden))
    if method in ('lmf', 'lmf_pert'):
        return LMFFusion(dims, hidden_dims=hidden_dims, rank=args.rank,
                         dropout=args.lmf_dropout)
    if method in ('lmf_gate', 'lmf_gate_pert'):
        return GatedLMFFusion(dims, hidden_dims=hidden_dims, rank=args.rank,
                              dropout=args.lmf_dropout,
                              gate_hidden=args.gate_hidden,
                              gate_init=args.gate_init)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def standardize_splits(data):
    stats = {}
    for m in MODALITIES:
        x = data['splits']['train']['feats'][m]
        stats[m] = (x.mean(0), x.std(0).clamp_min(1e-6))
    out = {}
    for mode in ('train', 'valid', 'test'):
        src = data['splits'][mode]
        out[mode] = {
            **src,
            'feats': {m: (src['feats'][m] - stats[m][0]) / stats[m][1]
                      for m in MODALITIES},
            'quality': {
                'text': torch.ones(len(src['sample_id']), 1),
                'vision': src['quality']['vision']['face_frac'].unsqueeze(1),
                'audio': torch.stack([
                    src['quality']['audio']['speech_frac'],
                    src['quality']['audio']['snr_db'] / 60.0], dim=1),
            },
        }
    return out


def load_variant_raw(modality, cond, mode, ids, prefer_eval=False):
    """加载扰动缓存并按 ids 对齐，返回未标准化的特征与质量向量。"""
    primary = EVAL_DIR if prefer_eval else PERT_DIR
    fallback = PERT_DIR if prefer_eval else EVAL_DIR
    path = primary / f'{modality}_{cond}.pt'
    if not path.exists():
        path = fallback / f'{modality}_{cond}.pt'
    pack = torch.load(path, weights_only=False)
    sub = pack['splits'][mode]
    idx = {sid: i for i, sid in enumerate(sub['sample_id'])}
    order = [idx[s] for s in ids]
    feat = sub['feat'][order]
    if modality == 'text':
        q = torch.ones(len(ids), 1)
    elif modality == 'vision':
        q = sub['face_frac'][order].unsqueeze(1)
    elif modality == 'audio':
        q = torch.stack([sub['speech_frac'][order],
                         sub['snr_db'][order] / 60.0], dim=1)
    else:
        raise ValueError(modality)
    return feat, q


def load_variant(modality, cond, mode, ids, stats, prefer_eval=False):
    """加载扰动缓存，并用干净 train 统计量标准化。"""
    feat, q = load_variant_raw(modality, cond, mode, ids, prefer_eval)
    mean, std = stats[modality]
    return (feat - mean) / std, q


def build_eval_variants(split_data, stats):
    """test 用不同随机实现的评测缓存。"""
    ids = split_data['sample_id']
    bank = {}
    for m in MODALITIES:
        bank[m] = {}
        for cond in ALL_VARIANTS[m]:
            feat, q = load_variant(m, cond, 'test', ids, stats,
                                   prefer_eval=True)
            bank[m][cond] = {'feat': feat, 'q': q}
    return bank


def build_train_bank(split_data, stats, variants):
    """每个模态堆叠 [干净, 扰动1, ...] 方便按样本向量化采样。"""
    ids = split_data['sample_id']
    bank = {}
    for m in MODALITIES:
        feats = [split_data['feats'][m]]
        qs = [split_data['quality'][m]]
        for cond in variants[m]:
            f, q = load_variant(m, cond, 'train', ids, stats)
            feats.append(f)
            qs.append(q)
        bank[m] = {'feat': torch.stack(feats),   # [K+1, N, d]
                   'q': torch.stack(qs)}          # [K+1, N, k]
    return bank


# --------------------------------------------------------------------------- #
# 训练 / 评测
# --------------------------------------------------------------------------- #
def sample_train_batch(bank, idx, p_pert, p_miss, dev):
    feats, quality, present = {}, {}, {}
    b = len(idx)
    for m in MODALITIES:
        stack = bank[m]['feat']
        qstack = bank[m]['q']
        k = stack.size(0) - 1                  # 扰动条件数
        choice = torch.full((b,), k, dtype=torch.long, device=dev)
        pert = torch.rand(b, device=dev) < p_pert
        n_pert = int(pert.sum().item())
        if k > 0 and n_pert:
            choice[pert] = torch.randint(0, k, (n_pert,), device=dev)
        feat = stack[choice, idx]
        q = qstack[choice, idx]
        miss = torch.rand(b, device=dev) < p_miss
        if miss.any():
            feat = feat.clone()
            q = q.clone()
            feat[miss] = 0.0
            q[miss] = 0.0
        feats[m] = feat
        quality[m] = q
        present[m] = (~miss).float()
    return feats, quality, present


def eval_condition(clean, variants, mods, dev):
    feats, quality, present = {}, {}, {}
    for m in MODALITIES:
        if mods.get(m) == 'missing':
            feats[m] = torch.zeros_like(clean['feats'][m]).to(dev)
            quality[m] = torch.zeros_like(clean['quality'][m]).to(dev)
            present[m] = torch.zeros(len(clean['feats'][m]), device=dev)
        elif m in mods:
            v = variants[m][mods[m]]
            feats[m] = v['feat'].to(dev)
            quality[m] = v['q'].to(dev)
            present[m] = torch.ones(len(clean['feats'][m]), device=dev)
        else:
            feats[m] = clean['feats'][m].to(dev)
            quality[m] = clean['quality'][m].to(dev)
            present[m] = torch.ones(len(clean['feats'][m]), device=dev)
    return feats, quality, present


@torch.no_grad()
def predict(model, method, feats, quality, present):
    if method.startswith('lmf_gate'):
        return model(feats, quality, present, return_gates=True)
    return model(feats), None


def train_one(method, data, stats, args, dev):
    set_seed(args.seed)
    dims = {m: data['train']['feats'][m].shape[1] for m in MODALITIES}
    model = build_model(method, dims, args).to(dev)
    gated = method.startswith('lmf_gate')
    augmented = method.endswith('_pert')

    tr, va, te = data['train'], data['valid'], data['test']
    variants = {m: (TRAIN_VARIANTS[m] if m in args.pert_modalities else ())
                for m in MODALITIES}
    train_bank = build_train_bank(tr, stats, variants) if augmented else None
    if train_bank is not None:
        for m in MODALITIES:
            train_bank[m]['feat'] = train_bank[m]['feat'].to(dev)
            train_bank[m]['q'] = train_bank[m]['q'].to(dev)
    tr_clean = {m: tr['feats'][m].to(dev) for m in MODALITIES}
    tr_q = {m: tr['quality'][m].to(dev) for m in MODALITIES}
    tr_present = {m: torch.ones(len(tr['sample_id']), device=dev)
                  for m in MODALITIES}
    va_clean = {'feats': {m: va['feats'][m].to(dev) for m in MODALITIES},
                'quality': {m: va['quality'][m].to(dev) for m in MODALITIES}}
    te_clean = {'feats': {m: te['feats'][m].to(dev) for m in MODALITIES},
                'quality': {m: te['quality'][m].to(dev) for m in MODALITIES}}
    eval_variants = build_eval_variants(te, stats)
    va_present = {m: torch.ones(len(va['sample_id']), device=dev)
                  for m in MODALITIES}

    ytr = tr['label3'].to(dev)
    yva, yte = va['label3'], te['label3']
    counts = torch.bincount(ytr, minlength=3).float()
    weight = counts.sum() / (3.0 * counts)
    crit = nn.CrossEntropyLoss(weight=weight.to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    print(f'[{method}] 参数 {sum(p.numel() for p in model.parameters()):,} | '
          f'gated={gated} augmented={augmented} | '
          f'类别计数 {counts.tolist()}')

    best_f1, best_state, best_ep, left = -1.0, None, 0, args.patience
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(ytr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            if augmented:
                feats, q, present = sample_train_batch(
                    train_bank, idx, args.p_pert, args.p_miss, dev)
            else:
                feats = {m: tr_clean[m][idx] for m in MODALITIES}
                q = {m: tr_q[m][idx] for m in MODALITIES}
                present = {m: tr_present[m][idx] for m in MODALITIES}
            opt.zero_grad()
            logits = (model(feats, q, present) if gated else model(feats))
            loss = crit(logits, ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if gated:
                logits, _ = model(va_clean['feats'], va_clean['quality'],
                                  va_present, return_gates=True)
            else:
                logits = model(va_clean['feats'])
            pred = logits.argmax(1).cpu().numpy()
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
    results = {}
    with torch.no_grad():
        for name, mods in EVAL_CONDITIONS:
            feats, q, present = eval_condition(te_clean, eval_variants, mods, dev)
            logits, gates = predict(model, method, feats, q, present)
            pred = logits.argmax(1).cpu().numpy()
            res = summarize(yte.numpy(), pred)
            res['n'] = int(len(yte))
            if gates is not None:
                res['gate_mean'] = {m: round(float(gates[m].mean()), 4)
                                    for m in MODALITIES}
            results[name] = res
            if name in ('clean', 'audio_snr0', 'vision_occ60',
                        'text_corr30', 'missing_text'):
                print(f'  [{method}/{name}] Acc {res["accuracy"]:.4f} '
                      f'Macro-F1 {res["macro_f1"]:.4f}')
    result = {
        'method': method,
        'seed': args.seed,
        'gated': gated,
        'augmented': augmented,
        'n_params': sum(p.numel() for p in model.parameters()),
        'best_valid_macro_f1': round(best_f1, 4),
        'best_epoch': best_ep,
        'config': {'hidden_dims': dict(zip(MODALITIES, args.lmf_hidden)),
                   'rank': args.rank, 'lmf_dropout': args.lmf_dropout,
                   'gate_hidden': args.gate_hidden if gated else None,
                   'gate_init': args.gate_init if gated else None,
                   'pert_modalities': (list(args.pert_modalities)
                                       if augmented else None),
                   'p_pert': args.p_pert if augmented else 0.0,
                   'p_miss': args.p_miss if augmented else 0.0},
        'conditions': results,
    }
    return model, result


# --------------------------------------------------------------------------- #
# 干净的拼接基线在受扰条件下的参考表现
# --------------------------------------------------------------------------- #
def eval_concat_reference(data, dev):
    ckpt_path = (ROOT / 'outputs' / 'results' / 'fusion_baseline'
                 / 'concat_best.pt')
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = ConcatFusion(ckpt['dims'], hidden=ckpt['config']['hidden'],
                         dropout=ckpt['config']['dropout']).to(dev)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    norm = ckpt['norm']
    te = data['test']
    ids = te['sample_id']
    clean_feats = {
        m: (data['splits']['test']['feats'][m] - norm[m]['mean'])
           / norm[m]['std'] for m in MODALITIES}
    clean = {'feats': clean_feats, 'quality': te['quality']}
    variants = {}
    for m in MODALITIES:
        variants[m] = {}
        for cond in ALL_VARIANTS[m]:
            raw, q = load_variant_raw(m, cond, 'test', ids, prefer_eval=True)
            variants[m][cond] = {
                'feat': (raw - norm[m]['mean']) / norm[m]['std'], 'q': q}
    yte = te['label3'].numpy()
    results = {}
    with torch.no_grad():
        for name, mods in EVAL_CONDITIONS:
            feats, q, present = eval_condition(clean, variants, mods, dev)
            pred = model(feats).argmax(1).cpu().numpy()
            results[name] = {**summarize(yte, pred), 'n': int(len(yte))}
    return {'method': 'concat_reference',
            'n_params': sum(p.numel() for p in model.parameters()),
            'conditions': results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+',
                    default=['lmf', 'lmf_pert', 'lmf_gate', 'lmf_gate_pert'],
                    choices=['lmf', 'lmf_pert', 'lmf_gate', 'lmf_gate_pert'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--lmf-hidden', type=int, nargs=3, default=[64, 16, 64],
                    metavar=('TEXT', 'VISION', 'AUDIO'))
    ap.add_argument('--lmf-dropout', type=float, default=0.2)
    ap.add_argument('--gate-hidden', type=int, default=8)
    ap.add_argument('--gate-init', type=float, default=0.0,
                    help='门控末层权重初始化标准差；0 表示零初始化')
    ap.add_argument('--p-pert', type=float, default=0.40)
    ap.add_argument('--p-miss', type=float, default=0.08)
    ap.add_argument('--pert-modalities', nargs='+', default=['text', 'vision', 'audio'],
                    choices=['text', 'vision', 'audio'],
                    help='参与扰动增强的模态；未列出的模态只用干净特征')
    ap.add_argument('--eval-concat', action='store_true', default=True)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'results'
                                         / 'fusion_gated'))
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_fusion_features()
    stats = {m: (raw['splits']['train']['feats'][m].mean(0),
                 raw['splits']['train']['feats'][m].std(0).clamp_min(1e-6))
             for m in MODALITIES}
    data = standardize_splits(raw)
    data['splits'] = raw['splits']          # concat 参考评测需要原始特征

    summary = {}
    for method in args.methods:
        model, result = train_one(method, data, stats, args, dev)
        with open(out_dir / f'{method}_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        np.savetxt(out_dir / f'{method}_confusion_matrix.csv',
                   np.array(result['conditions']['clean']['confusion_matrix']),
                   fmt='%d', delimiter=',')
        torch.save({'method': method, 'config': result['config'],
                    'dims': {m: data['train']['feats'][m].shape[1]
                             for m in MODALITIES},
                    'state_dict': {k: v.cpu() for k, v in
                                   model.state_dict().items()},
                    'norm': {m: {'mean': stats[m][0].cpu(),
                                 'std': stats[m][1].cpu()}
                             for m in MODALITIES}},
                   out_dir / f'{method}_best.pt')
        summary[method] = {
            'n_params': result['n_params'],
            'best_valid_macro_f1': result['best_valid_macro_f1'],
            'best_epoch': result['best_epoch'],
            'conditions': {name: {'accuracy': r['accuracy'],
                                  'macro_f1': r['macro_f1'],
                                  'gate_mean': r.get('gate_mean')}
                           for name, r in result['conditions'].items()},
        }

    if args.eval_concat:
        ref = eval_concat_reference(data, dev)
        if ref:
            with open(out_dir / 'concat_reference.json', 'w',
                      encoding='utf-8') as f:
                json.dump(ref, f, ensure_ascii=False, indent=2)
            summary['concat_reference'] = {
                'n_params': ref['n_params'],
                'conditions': {name: {'accuracy': r['accuracy'],
                                      'macro_f1': r['macro_f1']}
                               for name, r in ref['conditions'].items()},
            }

    with open(out_dir / 'robustness.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('结果已保存到:', out_dir)


if __name__ == '__main__':
    main()
