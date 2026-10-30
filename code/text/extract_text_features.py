# -*- coding: utf-8 -*-
"""文本模态特征提取：bert-base-chinese（冻结）→ CLS 向量。

按进度汇报的约定：
    文本 → tokenizer 编码 → bert-base-chinese 前向 → 取 [CLS] 向量，d = 768。
    全部参数冻结，仅做前向推理；特征缓存为 .pt，供分类基线与后续 LMF 融合共用。

用法：
    python code/text/extract_text_features.py [--out outputs/features/text_cls.pt]
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.chsims import load_chsims, split_by_mode

MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 64   # CH-SIMS 最长转写 47 字符，64 足够


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2]
                                          / 'outputs' / 'features' / 'text_cls.pt'))
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    df = load_chsims()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    with torch.no_grad():
        for mode, sub in split_by_mode(df).items():
            texts = sub['text'].tolist()
            feats = []
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i:i + args.batch_size]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors='pt').to(args.device)
                cls = model(**enc).last_hidden_state[:, 0]  # [B, 768]
                feats.append(cls.cpu())
            data[mode] = {
                'sample_id': sub['sample_id'].tolist(),
                'feat': torch.cat(feats),                       # [N, 768]
                'label3': torch.tensor(sub['label3'].values),     # [N]
                'label_m': torch.tensor(sub['label_m'].values, dtype=torch.float32),
            }
            print(f'{mode:>5}: {len(texts)} 条, feat {data[mode]["feat"].shape}')
    torch.save({'model': MODEL_NAME, 'pooling': 'cls', 'dim': 768,
                'splits': data}, out_path)
    print('已保存:', out_path)


if __name__ == '__main__':
    main()
