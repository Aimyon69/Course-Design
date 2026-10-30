# -*- coding: utf-8 -*-
"""汇总多个 seed 的 robustness.json，输出均值 ± 标准差表格。

目录约定：
    outputs/results/fusion_gated/robustness.json          seed 42
    outputs/results/fusion_gated/seed1/robustness.json    seed 1
    outputs/results/fusion_gated/seed7/robustness.json    seed 7

用法：python code/fusion/analyze_seeds.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs' / 'results' / 'fusion_gated'
SEED_FILES = [
    ('42', BASE / 'robustness.json'),
    ('1', BASE / 'seed1' / 'robustness.json'),
    ('7', BASE / 'seed7' / 'robustness.json'),
]

METHOD_NAMES = {
    'lmf': 'LMF',
    'lmf_pert': 'LMF+扰动',
    'lmf_gate': 'LMF+门控',
    'lmf_gate_pert': 'LMF+门控+扰动',
    'concat_reference': '拼接+MLP',
}
GROUPS = {
    '语音退化': ('audio_snr10', 'audio_snr5', 'audio_snr0'),
    '视觉退化': ('vision_occ20', 'vision_occ40', 'vision_occ60'),
    '文本退化': ('text_corr10', 'text_corr20', 'text_corr30'),
    '单模态缺失': ('missing_text', 'missing_vision', 'missing_audio'),
    '双模态缺失': ('missing_text+vision', 'missing_text+audio',
                   'missing_vision+audio'),
}


def load_all():
    runs = []
    for seed, path in SEED_FILES:
        if path.exists():
            runs.append((seed, json.load(open(path, encoding='utf-8'))))
    return runs


def cell(values):
    if not values:
        return '—'
    if len(values) == 1:
        return f'{values[0]:.4f}'
    return f'{statistics.mean(values):.4f}±{statistics.pstdev(values):.4f}'


def main():
    runs = load_all()
    print('使用的 seed：' + ', '.join(s for s, _ in runs) + '\n')
    methods = [m for m in METHOD_NAMES
               if all(m in data for _, data in runs)]
    if not methods:
        print('没有可汇总的结果')
        return
    conds = list(runs[0][1][methods[0]]['conditions'])

    print('## 各条件 Macro-F1（多 seed 均值±总体标准差）\n')
    print('| 方法 | ' + ' | '.join(conds) + ' |')
    print('|---|' + '---|' * len(conds))
    for m in methods:
        row = [METHOD_NAMES[m]]
        for c in conds:
            vals = [d[m]['conditions'][c]['macro_f1'] for _, d in runs
                    if d[m]['conditions'][c]['macro_f1'] is not None]
            row.append(cell(vals))
        print('| ' + ' | '.join(row) + ' |')

    print('\n## 分组平均（先组内平均，再跨 seed 平均±标准差）\n')
    print('| 方法 | 干净 | ' + ' | '.join(GROUPS) + ' |')
    print('|---|' + '---|' * (len(GROUPS) + 1))
    for m in methods:
        row = [METHOD_NAMES[m]]
        for c in ('clean', *GROUPS):
            per_seed = []
            for _, d in runs:
                cd = d[m]['conditions']
                if c == 'clean':
                    v = cd['clean']['macro_f1']
                    if v is not None:
                        per_seed.append(v)
                else:
                    vals = [cd[n]['macro_f1'] for n in GROUPS[c]
                            if n in cd and cd[n]['macro_f1'] is not None]
                    if vals:
                        per_seed.append(sum(vals) / len(vals))
            row.append(cell(per_seed))
        print('| ' + ' | '.join(row) + ' |')

    print('\n## 配对差：门控+扰动 − LMF（越正越好）\n')
    if 'lmf' in methods and 'lmf_gate_pert' in methods:
        print('| 条件 | 绝对 F1 差 | 相对干净 F1 的退化差（百分点） |')
        print('|---|---|---|')
        for c in conds:
            if c == 'clean':
                continue
            diffs, drop_diffs = [], []
            for _, d in runs:
                a = d['lmf']['conditions']
                b = d['lmf_gate_pert']['conditions']
                if a[c]['macro_f1'] is None or b[c]['macro_f1'] is None:
                    continue
                diffs.append(b[c]['macro_f1'] - a[c]['macro_f1'])
                da = 100 * (a[c]['macro_f1'] - a['clean']['macro_f1']) / a['clean']['macro_f1']
                db = 100 * (b[c]['macro_f1'] - b['clean']['macro_f1']) / b['clean']['macro_f1']
                drop_diffs.append(db - da)
            print(f'| {c} | {cell(diffs)} | {cell(drop_diffs)} |')

    print('\n## 配对差：门控+扰动 − 扰动训练（隔离结构改进）\n')
    if 'lmf_pert' in methods and 'lmf_gate_pert' in methods:
        print('| 条件 | 绝对 F1 差 | 相对退化差（百分点） |')
        print('|---|---|---|')
        for c in conds:
            if c == 'clean':
                continue
            diffs, drop_diffs = [], []
            for _, d in runs:
                a = d['lmf_pert']['conditions']
                b = d['lmf_gate_pert']['conditions']
                if a[c]['macro_f1'] is None or b[c]['macro_f1'] is None:
                    continue
                diffs.append(b[c]['macro_f1'] - a[c]['macro_f1'])
                da = 100 * (a[c]['macro_f1'] - a['clean']['macro_f1']) / a['clean']['macro_f1']
                db = 100 * (b[c]['macro_f1'] - b['clean']['macro_f1']) / b['clean']['macro_f1']
                drop_diffs.append(db - da)
            print(f'| {c} | {cell(diffs)} | {cell(drop_diffs)} |')

    print('\n## 门控均值（跨 seed，仅门控方法）\n')
    for m in ('lmf_gate', 'lmf_gate_pert'):
        if m not in methods:
            continue
        print(f'### {METHOD_NAMES[m]}\n')
        print('| 条件 | text | vision | audio |')
        print('|---|---|---|---|')
        for c in conds:
            cells = []
            for mod in ('text', 'vision', 'audio'):
                vals = []
                for _, d in runs:
                    gm = d[m]['conditions'][c].get('gate_mean')
                    if gm and gm.get(mod) is not None:
                        vals.append(gm[mod])
                cells.append(f'{statistics.mean(vals):.3f}' if vals else '—')
            print(f'| {c} | ' + ' | '.join(cells) + ' |')
        print()


if __name__ == '__main__':
    main()
