# -*- coding: utf-8 -*-
"""三模态融合实验的统一特征加载与对齐。

读取文本 / 视觉 / 语音三个阶段产出的缓存：
    outputs/features/text_cls.pt   (bert-base-chinese CLS, 768)
    outputs/features/vis_cls.pt    (CLIP-ViT-B/32 人脸帧均值, 512)
    outputs/features/audio_cls.pt  (chinese-wav2vec2-base 时序均值, 768)

所有缓存必须来自同一 label.csv / 同一官方划分。本模块按 sample_id 严格对齐三路，
避免依赖各文件内部行序；标签一致性也会断言检查。
"""
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_FILES = {
    'text': PROJECT_ROOT / 'outputs' / 'features' / 'text_cls.pt',
    'vision': PROJECT_ROOT / 'outputs' / 'features' / 'vis_cls.pt',
    'audio': PROJECT_ROOT / 'outputs' / 'features' / 'audio_cls.pt',
}
SPLITS = ('train', 'valid', 'test')


def load_fusion_features(feature_files: dict[str, Path] | None = None) -> dict:
    """返回 {'dims': {...}, 'model_names': {...}, 'splits': {...}}。

    splits[mode] = {
        'sample_id': list[str],
        'feats': {modality: Tensor[N, d_m]},
        'X': Tensor[N, sum(dims)],      # 按 text/vision/audio 顺序拼接
        'label3': Tensor[N],
        'label_m': Tensor[N],
    }
    """
    files = feature_files or FEATURE_FILES
    packs = {m: torch.load(p, weights_only=False) for m, p in files.items()}

    for m, pack in packs.items():
        for mode in SPLITS:
            ids = pack['splits'][mode]['sample_id']
            assert len(ids) == len(set(ids)), f'{m}/{mode} 存在重复 sample_id'
            assert set(ids) == set(packs['text']['splits'][mode]['sample_id']), \
                f'{m}/{mode} 与 text 的样本集合不一致'

    dims = {m: int(packs[m]['dim']) for m in packs}
    model_names = {m: packs[m].get('model') for m in packs}
    out = {'dims': dims, 'model_names': model_names, 'splits': {}}
    for mode in SPLITS:
        ids = packs['text']['splits'][mode]['sample_id']
        feats = {}
        quality = {}
        labels = None
        for m in ('text', 'vision', 'audio'):
            sub = packs[m]['splits'][mode]
            idx = {sid: i for i, sid in enumerate(sub['sample_id'])}
            order = [idx[s] for s in ids]
            feats[m] = sub['feat'][order]
            if m == 'vision' and 'face_frac' in sub:
                quality['vision'] = {'face_frac': sub['face_frac'][order]}
            elif m == 'audio' and 'speech_frac' in sub:
                quality['audio'] = {
                    'speech_frac': sub['speech_frac'][order],
                    'snr_db': sub['snr_db'][order],
                }
            y = sub['label3'][order]
            if labels is None:
                labels = y
            else:
                assert torch.equal(labels, y), f'{m}/{mode} 标签与 text 不一致'
        out['splits'][mode] = {
            'sample_id': ids,
            'feats': feats,
            'quality': quality,
            'X': torch.cat([feats[m] for m in ('text', 'vision', 'audio')], dim=1),
            'label3': labels,
            'label_m': packs['text']['splits'][mode]['label_m'],
        }
    return out


if __name__ == '__main__':
    data = load_fusion_features()
    print('dims:', data['dims'])
    print('models:', data['model_names'])
    for mode in SPLITS:
        d = data['splits'][mode]
        print(f'{mode:>5}: {len(d["sample_id"])} 条, X {tuple(d["X"].shape)}, '
              f'标签 {torch.bincount(d["label3"]).tolist()}')
