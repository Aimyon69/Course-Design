# -*- coding: utf-8 -*-
"""CH-SIMS 统一数据接口。

label.csv 列（无表头，经 MMSA-FET dataset.py 官方解析代码确认）：
    video_id, clip_id, text, label, label_T, label_A, label_V, annotation, mode
其中：
    label   = 多模态连续标签（-1.0 ~ 1.0，步长 0.2），即 label_M
    label_T/A/V = 文本/语音/视觉单模态连续标签
    annotation  = 三类标签 Negative / Neutral / Positive
                规则（阈值 0，与 MMSA-FET 一致，已验证 2281 条 0 不一致）：
                    label < 0 -> Negative；label == 0 -> Neutral；label > 0 -> Positive
    mode    = 官方划分 train / valid / test（1368 / 456 / 457）

本模块只依赖 pandas，供所有模态的脚本共用。

目录约定（2026-10-26 起）：
    dataset/label.csv          标注文件
    dataset/video_XXXX/YYYY.mp4 原始视频（CH-SIMS Raw）
    doc/                       方案与汇报文档
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LABEL2ID = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / 'dataset' / 'label.csv'
RAW_VIDEO_ROOT = PROJECT_ROOT / 'dataset'


@dataclass
class ChsimsSample:
    sample_id: str      # video_0001_0001
    video_id: str
    clip_id: str
    text: str
    label_m: float      # 多模态连续标签
    label_t: float
    label_a: float
    label_v: float
    label3: int         # 0=Negative 1=Neutral 2=Positive（annotation 列）
    split: str          # train / valid / test


def load_chsims(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """读取 label.csv，返回统一样本表（DataFrame）。"""
    csv_path = Path(csv_path)
    rows = [r for r in csv.reader(open(csv_path, encoding='utf-8')) if len(r) >= 9]
    df = pd.DataFrame(rows, columns=[
        'video_id', 'clip_id', 'text', 'label', 'label_T', 'label_A', 'label_V',
        'annotation', 'mode'])
    df['sample_id'] = df['video_id'] + '_' + df['clip_id']
    df['label_m'] = df['label'].astype(float)
    df['label_t'] = df['label_T'].astype(float)
    df['label_a'] = df['label_A'].astype(float)
    df['label_v'] = df['label_V'].astype(float)
    df['label3'] = df['annotation'].map(LABEL2ID)

    # 阈值规则自检：annotation 必须等于 label_m 按阈值 0 的离散化
    pred = df['label_m'].map(lambda v: 0 if v < 0 else (2 if v > 0 else 1))
    n_bad = int((pred != df['label3']).sum())
    assert n_bad == 0, f'annotation 与 label_m 阈值规则不一致: {n_bad} 条'

    # 划分合法性自检
    assert set(df['mode']) <= {'train', 'valid', 'test'}
    return df[['sample_id', 'video_id', 'clip_id', 'text', 'label_m',
               'label_t', 'label_a', 'label_v', 'label3', 'mode']]


def split_by_mode(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {m: df[df['mode'] == m].reset_index(drop=True)
            for m in ('train', 'valid', 'test')}


if __name__ == '__main__':
    df = load_chsims()
    print('总样本:', len(df))
    for m, sub in split_by_mode(df).items():
        dist = sub['label3'].map(ID2LABEL).value_counts().to_dict()
        print(f'{m:>5}: {len(sub)} 条, 标签分布 {dist}')
