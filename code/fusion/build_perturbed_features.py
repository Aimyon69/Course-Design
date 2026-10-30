# -*- coding: utf-8 -*-
"""为质量门控与鲁棒性实验生成固定强度的受扰特征缓存。

三类扰动都在**原始输入**上施加，再走与干净特征完全相同的冻结编码器链路，
保证训练增强与评测使用同一口径：

    audio  —— 16 kHz 波形叠加白噪声，固定 SNR ∈ {10, 5, 0} dB，
              重新计算 speech_frac / snr_db / rms_db，再过 Wav2Vec2 均值池化；
    vision —— 人脸框内随机位置贴灰度矩形（遮挡面积占比 0.2 / 0.3 / 0.4 / 0.6），
              在遮挡后的帧上重新做人脸检测（face_frac 随遮挡下降），再过 CLIP；
    text   —— 按比例随机替换汉字（10% / 15% / 20% / 30%），重新过 bert-base-chinese。

说明：
    - 扰动只用于训练增强与鲁棒性评测，不改变官方划分与标签；
    - 所有随机过程使用 sample_id / 条件派生的固定种子，可复现；
    - 缺失模态不需要缓存：训练/评测时把特征置零、present=0、质量置零即可。

用法：
    python code/fusion/build_perturbed_features.py --modality audio
    python code/fusion/build_perturbed_features.py --modality text
    python code/fusion/build_perturbed_features.py --modality vision   # 约 30 分钟
    python code/fusion/build_perturbed_features.py --modality all --device cuda
"""
import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import (AutoModel, AutoTokenizer, CLIPModel,
                          Wav2Vec2Model)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.chsims import RAW_VIDEO_ROOT, load_chsims, split_by_mode
from audio.extract_audio_features import (SAMPLE_RATE, audio_quality,
                                          decode_audio, encode_batch)
from vision.extract_vis_features import (FACE_SIZE, T_FRAMES, crop_face,
                                         detect_face, sample_frames,
                                         to_clip_input)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs' / 'features' / 'perturbed'
EVAL_DIR = ROOT / 'outputs' / 'features' / 'perturbed_eval'
AUDIO_SNRS = (10, 5, 0)
VIS_SEVERITIES = (0.2, 0.3, 0.4, 0.6)
TEXT_RATES = (0.10, 0.15, 0.20, 0.30)
ALL_SPLITS = ('train', 'valid', 'test')


def out_dir_for(seed_offset: int) -> Path:
    """offset=0 为训练增强缓存；offset>0 为评测用、不同随机实现的缓存。"""
    return OUT_DIR if seed_offset == 0 else EVAL_DIR


# --------------------------------------------------------------------------- #
# 文本
# --------------------------------------------------------------------------- #
def corrupt_text(text: str, rate: float, pool: list[str], seed: str) -> str:
    rng = random.Random(seed)
    chars = list(text)
    idx = [i for i, c in enumerate(chars) if '一' <= c <= '鿿']
    if not idx or rate <= 0:
        return text
    k = max(1, int(round(rate * len(idx))))
    for i in rng.sample(idx, min(k, len(idx))):
        choices = [c for c in pool if c != chars[i]]
        chars[i] = rng.choice(choices)
    return ''.join(chars)


def build_text(device: str, batch_size: int, splits, seed_offset: int):
    from text.extract_text_features import MAX_LEN, MODEL_NAME
    df = load_chsims()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    pool = sorted({c for t in df['text'].tolist() for c in t
                   if '一' <= c <= '鿿'})
    print(f'[text] 字符池 {len(pool)} 个，random seed offset {seed_offset}')

    out = {}
    out_dir = out_dir_for(seed_offset)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rate in TEXT_RATES:
        cond = f'corr{int(round(rate * 100))}'
        out[cond] = {'splits': {}}
        for mode, sub in split_by_mode(df).items():
            if mode not in splits:
                continue
            texts = [corrupt_text(t, rate, pool,
                                  f'{sid}|{rate}|{seed_offset}')
                     for sid, t in zip(sub['sample_id'], sub['text'])]
            feats = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    enc = tok(texts[i:i + batch_size], padding=True,
                              truncation=True, max_length=MAX_LEN,
                              return_tensors='pt').to(device)
                    feats.append(model(**enc).last_hidden_state[:, 0].cpu())
            out[cond]['splits'][mode] = {
                'sample_id': sub['sample_id'].tolist(),
                'feat': torch.cat(feats),
                'label3': torch.tensor(sub['label3'].values),
                'label_m': torch.tensor(sub['label_m'].values, dtype=torch.float32),
                'present': torch.ones(len(sub)),
                'text': texts,
            }
            print(f'  [text/{cond}/{mode}] {len(texts)} 条')
        out[cond].update({'modality': 'text', 'model': MODEL_NAME,
                          'dim': 768, 'condition': {'type': 'char_corruption',
                                                    'rate': rate},
                          'seed_offset': seed_offset})
        torch.save(out[cond], out_dir / f'text_{cond}.pt')
    print(f'[text] 完成，{len(TEXT_RATES)} 个条件，offset={seed_offset}')


# --------------------------------------------------------------------------- #
# 语音
# --------------------------------------------------------------------------- #
def add_white_noise(wav: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    power = float(np.mean(wav.astype(np.float64) ** 2))
    npow = float(np.mean(noise.astype(np.float64) ** 2))
    noise *= np.sqrt(power / (10 ** (snr_db / 10.0)) / max(npow, 1e-12))
    return (wav + noise).astype(np.float32)


def build_audio(device: str, batch_size: int, splits, seed_offset: int):
    from audio.extract_audio_features import MODEL_NAME
    df = load_chsims()
    split_map = split_by_mode(df)
    sids = [s for mode in splits for s in split_map[mode]['sample_id']]
    out_dir = out_dir_for(seed_offset)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / 'audio_partial.pt'
    store = {}
    for snr in AUDIO_SNRS:
        store[f'snr{snr}'] = {'feat': {}, 'speech_frac': {}, 'snr_db': {},
                              'rms_db': {}, 'dur_sec': {}}
    if partial_path.exists():
        store = torch.load(partial_path, weights_only=False)

    todo = [s for s in sids
            if any(s not in store[f'snr{snr}']['feat'] for snr in AUDIO_SNRS)]
    if not todo:
        print('[audio] 无待处理样本')
        return
    print(f'[audio] 待处理 {len(todo)} 条，加载 {MODEL_NAME} ...')
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    for k, sid in enumerate(todo):
        vid, cid = sid.rsplit('_', 1)
        wav = decode_audio(RAW_VIDEO_ROOT / vid / f'{cid}.mp4')
        if wav is None:
            for snr in AUDIO_SNRS:
                store[f'snr{snr}']['feat'][sid] = torch.zeros(768)
                store[f'snr{snr}']['speech_frac'][sid] = 0.0
                store[f'snr{snr}']['snr_db'][sid] = 0.0
                store[f'snr{snr}']['rms_db'][sid] = -80.0
                store[f'snr{snr}']['dur_sec'][sid] = 0.0
            continue
        noisies = [add_white_noise(
                       wav, snr,
                       seed=1000 * j + zlib_crc(sid) + 100000 * seed_offset)
                   for j, snr in enumerate(AUDIO_SNRS)]
        feats = encode_batch(model, noisies, device)
        for snr, noisy, feat in zip(AUDIO_SNRS, noisies, feats):
            q = audio_quality(noisy)
            d = store[f'snr{snr}']
            d['feat'][sid] = feat
            d['speech_frac'][sid] = q['speech_frac']
            d['snr_db'][sid] = q['snr_db']
            d['rms_db'][sid] = q['rms_db']
            d['dur_sec'][sid] = len(wav) / SAMPLE_RATE
        if (k + 1) % 50 == 0:
            torch.save(store, partial_path)
            print(f'  [audio] {k+1}/{len(todo)}，用时 {time.time()-t0:.0f}s')
    torch.save(store, partial_path)
    print(f'[audio] 提取完成，{len(todo)} 条，用时 {time.time()-t0:.0f}s')


# --------------------------------------------------------------------------- #
# 视觉
# --------------------------------------------------------------------------- #
def zlib_crc(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode('utf-8')) & 0x7FFFFFFF


def _pseudo_bbox(frame: np.ndarray) -> tuple:
    h, w = frame.shape[:2]
    bw, bh = int(w * 0.5), int(h * 0.5)
    return (w // 2 - bw // 2, h // 2 - bh // 2, bw, bh)


def occlude_frame(frame: np.ndarray, bbox: tuple, frac: float, seed: str):
    """在 bbox 内随机位置贴一块灰度矩形，面积约占 bbox 的 frac。"""
    x, y, w, h = [int(v) for v in bbox]
    rng = random.Random(seed)
    ow = max(1, min(w, int(round(w * np.sqrt(frac)))))
    oh = max(1, min(h, int(round(frac * w * h / ow))))
    cx = x + rng.randint(0, max(0, w - ow))
    cy = y + rng.randint(0, max(0, h - oh))
    out = frame.copy()
    v = rng.choice([0, 32, 64, 96, 128, 160, 192, 224])
    cv2.rectangle(out, (cx, cy), (min(cx + ow, frame.shape[1] - 1),
                                  min(cy + oh, frame.shape[0] - 1)),
                  (v, v, v), -1)
    return out


def build_vision(device: str, splits, seed_offset: int):
    from vision.extract_vis_features import MODEL_NAME
    df = load_chsims()
    split_map = split_by_mode(df)
    sids = [s for mode in splits for s in split_map[mode]['sample_id']]
    out_dir = out_dir_for(seed_offset)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / 'vision_partial.pt'
    store = {f'occ{int(round(s * 100))}': {'feat': {}, 'face_frac': {}}
             for s in VIS_SEVERITIES}
    if partial_path.exists():
        store = torch.load(partial_path, weights_only=False)

    cond_names = list(store)
    todo = [s for s in sids
            if any(s not in store[c]['feat'] for c in cond_names)]
    if not todo:
        print('[vision] 无待处理样本')
        return
    print(f'[vision] 待处理 {len(todo)} 条，加载 {MODEL_NAME} ...')
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    for k, sid in enumerate(todo):
        vid, cid = sid.rsplit('_', 1)
        frames = sample_frames(RAW_VIDEO_ROOT / vid / f'{cid}.mp4', T_FRAMES)
        if not frames:
            for c in cond_names:
                store[c]['feat'][sid] = torch.zeros(512)
                store[c]['face_frac'][sid] = 0.0
            continue
        base = [detect_face(f) or _pseudo_bbox(f) for f in frames]
        for c in cond_names:
            frac = int(c[3:]) / 100.0
            crops, flags = [], []
            for i, f in enumerate(frames):
                g = occlude_frame(f, base[i], frac,
                                  f'{sid}|{c}|{i}|{seed_offset}')
                crop, ok = crop_face(g)
                crops.append(crop)
                flags.append(ok)
            x = to_clip_input(crops).to(device)
            with torch.no_grad():
                vo = model.vision_model(pixel_values=x)
                emb = model.visual_projection(vo.pooler_output)
            store[c]['feat'][sid] = emb.mean(0).cpu()
            store[c]['face_frac'][sid] = float(np.mean(flags))
        if (k + 1) % 50 == 0:
            torch.save(store, partial_path)
            print(f'  [vision] {k+1}/{len(todo)}，'
                  f'用时 {time.time()-t0:.0f}s')
    torch.save(store, partial_path)
    print(f'[vision] 提取完成，{len(todo)} 条，用时 {time.time()-t0:.0f}s')


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def finalize_audio(out_dir: Path, splits, seed_offset: int):
    store = torch.load(out_dir / 'audio_partial.pt', weights_only=False)
    df = load_chsims()
    for snr in AUDIO_SNRS:
        cond = f'snr{snr}'
        d = store[cond]
        splits_out = {}
        for mode, sub in split_by_mode(df).items():
            if mode not in splits:
                continue
            ids = sub['sample_id'].tolist()
            splits_out[mode] = {
                'sample_id': ids,
                'feat': torch.stack([d['feat'][s] for s in ids]),
                'label3': torch.tensor(sub['label3'].values),
                'label_m': torch.tensor(sub['label_m'].values, dtype=torch.float32),
                'speech_frac': torch.tensor([d['speech_frac'][s] for s in ids]),
                'snr_db': torch.tensor([d['snr_db'][s] for s in ids]),
                'rms_db': torch.tensor([d['rms_db'][s] for s in ids]),
                'dur_sec': torch.tensor([d['dur_sec'][s] for s in ids]),
            }
        torch.save({'modality': 'audio', 'model': 'TencentGameMate/chinese-wav2vec2-base',
                    'dim': 768, 'condition': {'type': 'white_noise', 'snr_db': snr},
                    'seed_offset': seed_offset, 'splits': splits_out},
                   out_dir / f'audio_{cond}.pt')
        print(f'[audio/{cond}] 已汇总 offset={seed_offset}')


def finalize_vision(out_dir: Path, splits, seed_offset: int):
    store = torch.load(out_dir / 'vision_partial.pt', weights_only=False)
    df = load_chsims()
    for sev in VIS_SEVERITIES:
        cond = f'occ{int(round(sev * 100))}'
        d = store[cond]
        splits_out = {}
        for mode, sub in split_by_mode(df).items():
            if mode not in splits:
                continue
            ids = sub['sample_id'].tolist()
            splits_out[mode] = {
                'sample_id': ids,
                'feat': torch.stack([d['feat'][s] for s in ids]),
                'label3': torch.tensor(sub['label3'].values),
                'label_m': torch.tensor(sub['label_m'].values, dtype=torch.float32),
                'face_frac': torch.tensor([d['face_frac'][s] for s in ids]),
            }
        torch.save({'modality': 'vision', 'model': 'openai/clip-vit-base-patch32',
                    'dim': 512, 'condition': {'type': 'face_occlusion',
                                              'area_frac': sev},
                    'seed_offset': seed_offset, 'splits': splits_out},
                   out_dir / f'vision_{cond}.pt')
        print(f'[vision/{cond}] 已汇总 offset={seed_offset}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modality', choices=['audio', 'vision', 'text', 'all'],
                    default='all')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--splits', choices=['all', 'test', 'train'], default='all',
                    help='评测缓存只生成 test；训练增强缓存可只生成 train')
    ap.add_argument('--seed-offset', type=int, default=0,
                    help='0=训练增强缓存；>0=评测用不同随机实现，写入 perturbed_eval/')
    ap.add_argument('--finalize', action='store_true')
    args = ap.parse_args()
    if args.splits == 'all':
        splits = ALL_SPLITS
    else:
        splits = (args.splits,)
    out_dir = out_dir_for(args.seed_offset)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        if args.modality in ('audio', 'all'):
            finalize_audio(out_dir, splits, args.seed_offset)
        if args.modality in ('vision', 'all'):
            finalize_vision(out_dir, splits, args.seed_offset)
        return

    if args.modality in ('text', 'all'):
        build_text(args.device, args.batch_size, splits, args.seed_offset)
    if args.modality in ('audio', 'all'):
        build_audio(args.device, args.batch_size, splits, args.seed_offset)
        finalize_audio(out_dir, splits, args.seed_offset)
    if args.modality in ('vision', 'all'):
        build_vision(args.device, splits, args.seed_offset)
        finalize_vision(out_dir, splits, args.seed_offset)


if __name__ == '__main__':
    main()
