# -*- coding: utf-8 -*-
"""把 fusion_gated/robustness.json 整理成 Markdown 表格，便于写实验记录。

用法：python code/fusion/analyze_gated.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'outputs' / 'results' / 'fusion_gated' / 'robustness.json'

METHOD_NAMES = {
    'lmf': 'LMF',
    'lmf_pert': 'LMF + 扰动训练',
    'lmf_gate': 'LMF + 质量门控',
    'lmf_gate_pert': 'LMF + 门控 + 扰动训练',
    'concat_reference': '拼接 + MLP（干净训练参考）',
}

GROUPS = {
    '语音退化': ('audio_snr10', 'audio_snr5', 'audio_snr0'),
    '视觉退化': ('vision_occ20', 'vision_occ40', 'vision_occ60'),
    '文本退化': ('text_corr10', 'text_corr20', 'text_corr30'),
    '单模态缺失': ('missing_text', 'missing_vision', 'missing_audio'),
    '双模态缺失': ('missing_text+vision', 'missing_text+audio',
                   'missing_vision+audio'),
}


def fmt(v):
    return '—' if v is None else f'{v:.4f}'


def main():
    data = json.load(open(RESULT, encoding='utf-8'))
    methods = [m for m in METHOD_NAMES if m in data]
    conds = list(data[methods[0]]['conditions'])

    print('## 干净/受扰 test Macro-F1\n')
    header = '| 方法 | ' + ' | '.join(conds) + ' |'
    print(header)
    print('|' + '---|' * (len(conds) + 1))
    for m in methods:
        row = [METHOD_NAMES[m]]
        for c in conds:
            row.append(fmt(data[m]['conditions'][c]['macro_f1']))
        print('| ' + ' | '.join(row) + ' |')

    print('\n## 相对干净条件的 Macro-F1 变化\n')
    print(header)
    print('|' + '---|' * (len(conds) + 1))
    for m in methods:
        clean = data[m]['conditions']['clean']['macro_f1']
        row = [METHOD_NAMES[m]]
        for c in conds:
            v = data[m]['conditions'][c]['macro_f1']
            row.append('—' if c == 'clean' else f'{v - clean:+.4f}')
        print('| ' + ' | '.join(row) + ' |')

    print('\n## 门控均值\n')
    gated = [m for m in methods if data[m]['conditions']['clean'].get('gate_mean')]
    if gated:
        print('| 方法 | 条件 | text | vision | audio |')
        print('|---|---|---|---|---|')
        for m in gated:
            for c in conds:
                gm = data[m]['conditions'][c].get('gate_mean')
                if gm:
                    print(f'| {METHOD_NAMES[m]} | {c} | {gm["text"]:.3f} | '
                          f'{gm["vision"]:.3f} | {gm["audio"]:.3f} |')

    print('\n## 分组平均 Macro-F1（组内条件等权）\n')
    print('| 方法 | 干净 | ' + ' | '.join(GROUPS) + ' |')
    print('|---|' + '---|' * (len(GROUPS) + 1))
    for m in methods:
        c = data[m]['conditions']
        row = [METHOD_NAMES[m], fmt(c['clean']['macro_f1'])]
        for group, names in GROUPS.items():
            vals = [c[n]['macro_f1'] for n in names
                    if n in c and c[n]['macro_f1'] is not None]
            row.append(f'{sum(vals) / len(vals):.4f}' if vals else '—')
        print('| ' + ' | '.join(row) + ' |')

    print('\n## 相对 LMF 的 Macro-F1 变化（同条件配对差）\n')
    base = data['lmf']['conditions'] if 'lmf' in data else None
    if base:
        others = [m for m in methods if m != 'lmf']
        print('| 条件 | ' + ' | '.join(METHOD_NAMES[m] for m in others) + ' |')
        print('|---|' + '---|' * len(others))
        for c in conds:
            b = base[c]['macro_f1']
            row = [c]
            for m in others:
                v = data[m]['conditions'][c]['macro_f1']
                row.append('—' if v is None or b is None else f'{v - b:+.4f}')
            print('| ' + ' | '.join(row) + ' |')

    print('\n## 改进主张检查：门控是否减小退化幅度\n')
    print('Δ = 100 × (受扰 F1 − 干净 F1) / 干净 F1，越接近 0 越稳。\n')
    print('| 条件 | LMF | LMF+扰动 | 门控+扰动 | 门控−LMF | 门控−扰动 |')
    print('|---|---|---|---|---|---|')
    for c in conds:
        if c == 'clean':
            continue
        cells = []
        drops = {}
        for m in ('lmf', 'lmf_pert', 'lmf_gate_pert'):
            if m not in data:
                cells.append('—')
                continue
            clean = data[m]['conditions']['clean']['macro_f1']
            v = data[m]['conditions'][c]['macro_f1']
            if v is None or not clean:
                cells.append('—')
                drops[m] = None
                continue
            d = 100 * (v - clean) / clean
            drops[m] = d
            cells.append(f'{d:+.1f}%')
        if drops.get('lmf') is not None and drops.get('lmf_gate_pert') is not None:
            a = f'{drops["lmf_gate_pert"] - drops["lmf"]:+.1f}%'
        else:
            a = '—'
        if drops.get('lmf_pert') is not None and drops.get('lmf_gate_pert') is not None:
            b = f'{drops["lmf_gate_pert"] - drops["lmf_pert"]:+.1f}%'
        else:
            b = '—'
        print(f'| {c} | ' + ' | '.join(cells) + f' | {a} | {b} |')


if __name__ == '__main__':
    main()
