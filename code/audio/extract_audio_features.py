# -*- coding: utf-8 -*-
"""语音模态特征提取：16 kHz 单声道波形（冻结 Wav2Vec2）→ 时序特征均值池化。

按进度汇报约定：
    wav 音频 → 16 kHz 重采样 → Wav2Vec2 提取音频时序特征 → 全局池化得到固定向量。
    编码器权重全部冻结，仅做前向推理；单向量维度 d_a = 768，供单模态基线与后续
    拼接 / LMF 融合共用同一条链路。

模型：
    TencentGameMate/chinese-wav2vec2-base（中文预训练 Wav2Vec2，未做 ASR 微调）。
    该 checkpoint 的 feature extractor 配置 do_normalize=false，因此直接输入原始
    16 kHz 波形，与预训练口径一致。

质量信号（供后续质量门控使用，训练、评测、演示三端口径一致）：
    以 25 ms 窗 / 10 ms 帧移计算帧级 RMS；取 10% 分位为噪声底、90% 分位为语音电平：
      speech_frac —— RMS 高于“噪声底 + 20% 动态范围”的帧比例，近似语音活动比例；
      snr_db      —— 20*log10(语音电平 / 噪声底)，截断到 [0, 60] dB；
      rms_db      —— 全片段 RMS 电平。
    这三个量与波形同步计算，不依赖文本或 ASR。

音频解码：
    环境未安装系统 ffmpeg，使用 imageio-ffmpeg 自带的 ffmpeg 可执行文件，
    从 mp4 直接解出 16 kHz 单声道 float32 PCM。

用法（支持断点续跑；全部 2281 条约几分钟）：
    python code/audio/extract_audio_features.py
    python code/audio/extract_audio_features.py --finalize   # 汇总缓存
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from transformers import Wav2Vec2Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.chsims import load_chsims, split_by_mode, RAW_VIDEO_ROOT

MODEL_NAME = 'TencentGameMate/chinese-wav2vec2-base'
SAMPLE_RATE = 16000
FEAT_DIM = 768
FRAME_MS = 25
HOP_MS = 10
SNR_MIN_DB, SNR_MAX_DB = 0.0, 60.0

PARTIAL = Path(__file__).resolve().parents[2] / 'outputs' / 'features' / 'audio_partial.pt'
FINAL = Path(__file__).resolve().parents[2] / 'outputs' / 'features' / 'audio_cls.pt'

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def decode_audio(video_path: Path, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """mp4 → 16 kHz 单声道 float32 波形；失败返回 None。"""
    cmd = [FFMPEG, '-v', 'error', '-i', str(video_path), '-vn',
           '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-']
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        print(f'  ! ffmpeg 异常 {video_path}: {type(e).__name__} {e}')
        return None
    if proc.returncode != 0:
        print(f'  ! ffmpeg 失败 {video_path}: {proc.stderr[:160]!r}')
        return None
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        return None
    return wav


def audio_quality(wav: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """帧级能量统计，返回 speech_frac / snr_db / rms_db。"""
    frame = max(int(sr * FRAME_MS / 1000), 1)
    hop = max(int(sr * HOP_MS / 1000), 1)
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = wav[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)

    noise = float(np.percentile(rms, 10))
    level = float(np.percentile(rms, 90))
    if level <= noise * 1.001:            # 近似静音 / 无动态范围
        speech_frac, snr_db = 0.0, 0.0
    else:
        thr = noise + 0.2 * (level - noise)
        speech_frac = float(np.mean(rms > thr))
        snr_db = float(np.clip(20.0 * np.log10((level + 1e-8) / (noise + 1e-8)),
                               SNR_MIN_DB, SNR_MAX_DB))
    rms_db = float(20.0 * np.log10(np.sqrt(np.mean(wav.astype(np.float64) ** 2)) + 1e-8))
    return {'speech_frac': speech_frac, 'snr_db': snr_db, 'rms_db': rms_db}


def conv_out_len(n: int, config) -> int:
    """按 conv_kernel / conv_stride 推算时序输出长度。"""
    for k, s in zip(config.conv_kernel, config.conv_stride):
        n = (n - k) // s + 1
    return max(n, 1)


@torch.no_grad()
def encode_batch(model, waves: list[np.ndarray], device: str) -> list[torch.Tensor]:
    """等长填充一批波形，按各自有效长度做均值池化。"""
    lens = [len(w) for w in waves]
    max_len = max(lens)
    x = torch.zeros(len(waves), max_len)
    mask = torch.zeros(len(waves), max_len, dtype=torch.long)
    for i, w in enumerate(waves):
        x[i, :len(w)] = torch.from_numpy(w)
        mask[i, :len(w)] = 1
    hidden = model(x.to(device), attention_mask=mask.to(device)).last_hidden_state
    out_lens = [conv_out_len(n, model.config) for n in lens]
    return [hidden[i, :out_lens[i]].mean(0).cpu() for i in range(len(waves))]


@torch.no_grad()
def extract(device: str, batch_size: int):
    df = load_chsims()
    sids = sorted(df['sample_id'].tolist())

    store = {'feat': {}, 'speech_frac': {}, 'snr_db': {}, 'rms_db': {}, 'dur_sec': {}}
    if PARTIAL.exists():
        store = torch.load(PARTIAL, weights_only=False)
    todo = [s for s in sids if s not in store['feat']]
    if not todo:
        print(f'无待处理样本（已完成 {len(store["feat"])}）')
        return

    print(f'待处理 {len(todo)} 条，加载 {MODEL_NAME} ...')
    t0 = time.time()
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'模型就绪（{time.time()-t0:.1f}s）')

    wavs, ok_ids = [], []
    for k, sid in enumerate(todo):
        vid, cid = sid.rsplit('_', 1)
        wav = decode_audio(RAW_VIDEO_ROOT / vid / f'{cid}.mp4')
        if wav is None:
            # 解码失败：零向量 + 质量 0，保证缓存完整；复跑时由 --redo 重试
            store['feat'][sid] = torch.zeros(FEAT_DIM)
            q = {'speech_frac': 0.0, 'snr_db': 0.0, 'rms_db': -80.0}
            store['speech_frac'][sid], store['snr_db'][sid] = q['speech_frac'], q['snr_db']
            store['rms_db'][sid], store['dur_sec'][sid] = q['rms_db'], 0.0
            continue
        q = audio_quality(wav)
        store['speech_frac'][sid] = q['speech_frac']
        store['snr_db'][sid] = q['snr_db']
        store['rms_db'][sid] = q['rms_db']
        store['dur_sec'][sid] = len(wav) / SAMPLE_RATE
        wavs.append(wav)
        ok_ids.append(sid)
        if len(wavs) == batch_size:
            for sid2, feat in zip(ok_ids, encode_batch(model, wavs, device)):
                store['feat'][sid2] = feat
            wavs, ok_ids = [], []
        if (k + 1) % 50 == 0:
            torch.save(store, PARTIAL)
            print(f'  解码 {k+1}/{len(todo)}，累计特征 {len(store["feat"])}，'
                  f'用时 {time.time()-t0:.0f}s')
    if wavs:
        for sid2, feat in zip(ok_ids, encode_batch(model, wavs, device)):
            store['feat'][sid2] = feat

    torch.save(store, PARTIAL)
    n_ok = sum(1 for v in store['dur_sec'].values() if v > 0)
    print(f'完成：{len(store["feat"])} 条，其中有效音频 {n_ok} 条，'
          f'总用时 {time.time()-t0:.0f}s')


def finalize():
    store = torch.load(PARTIAL, weights_only=False)
    df = load_chsims()
    missing = [s for s in df['sample_id'] if s not in store['feat']]
    assert not missing, f'缺少 {len(missing)} 条特征，如 {missing[:3]}'
    data = {}
    for mode, sub in split_by_mode(df).items():
        ids = sub['sample_id'].tolist()
        data[mode] = {
            'sample_id': ids,
            'feat': torch.stack([store['feat'][s] for s in ids]),
            'label3': torch.tensor(sub['label3'].values),
            'label_m': torch.tensor(sub['label_m'].values, dtype=torch.float32),
            'speech_frac': torch.tensor([store['speech_frac'][s] for s in ids]),
            'snr_db': torch.tensor([store['snr_db'][s] for s in ids]),
            'rms_db': torch.tensor([store['rms_db'][s] for s in ids]),
            'dur_sec': torch.tensor([store['dur_sec'][s] for s in ids]),
        }
        print(f'{mode:>5}: {len(sub)} 条, feat {data[mode]["feat"].shape}, '
              f'时长 {data[mode]["dur_sec"].mean():.2f}s, '
              f'speech_frac 均值 {data[mode]["speech_frac"].mean():.2f}, '
              f'SNR 中位 {data[mode]["snr_db"].median():.1f} dB')
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': MODEL_NAME, 'pooling': 'mean_over_time',
                'dim': FEAT_DIM, 'sample_rate': SAMPLE_RATE,
                'quality': ['speech_frac', 'snr_db', 'rms_db'],
                'splits': data}, FINAL)
    print('已保存:', FINAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--redo', action='store_true',
                    help='删除缓存后从头提取（用于修复解码失败样本）')
    args = ap.parse_args()

    if args.finalize:
        finalize()
        return
    if args.redo and PARTIAL.exists():
        PARTIAL.unlink()
    extract(args.device, args.batch_size)


if __name__ == '__main__':
    main()
