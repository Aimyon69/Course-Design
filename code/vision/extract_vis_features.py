# -*- coding: utf-8 -*-
"""视觉模态特征提取：CLIP ViT-B/32（冻结）→ 人脸帧 [CLS] 向量。

按进度汇报约定：
    预处理人脸 RGB 图像 [3,224,224] → CLIP-ViT-B/32 前向 → [CLS] 向量，d_v = 512；
    一段视频均匀采样 T 帧，得到 [T,512]，全局平均池化为固定向量，供 LMF 使用。
    全部参数冻结，仅做前向推理。

人脸预处理：
    Haar 多层级联（frontal_alt2 / frontal_default / profile，半分辨率检测），
    取最大人脸、外扩 40% 边距、resize 224×224；检不出时回退为中心裁剪，
    并记录 face_frac（检出帧比例），作为后续质量门控的视觉质量信号。

用法（分块执行，便于在受限时长内跑完 2281 条）：
    python code/vision/extract_vis_features.py --start 0 --end 300
    python code/vision/extract_vis_features.py --finalize   # 汇总缓存
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.chsims import load_chsims, split_by_mode, RAW_VIDEO_ROOT

MODEL_NAME = 'openai/clip-vit-base-patch32'
T_FRAMES = 8
FACE_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

PARTIAL = Path(__file__).resolve().parents[2] / 'outputs' / 'features' / 'vis_partial.pt'
FINAL = Path(__file__).resolve().parents[2] / 'outputs' / 'features' / 'vis_cls.pt'

_CASCADES = None


def cascades():
    global _CASCADES
    if _CASCADES is None:
        d = cv2.data.haarcascades
        _CASCADES = [cv2.CascadeClassifier(d + n) for n in (
            'haarcascade_frontalface_alt2.xml',
            'haarcascade_frontalface_default.xml',
            'haarcascade_profileface.xml')]
    return _CASCADES


def detect_face(frame: np.ndarray) -> tuple | None:
    """返回 (x, y, w, h)（原图坐标）或 None。"""
    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    for c in cascades():
        faces = c.detectMultiScale(small, 1.1, 3, minSize=(36, 36))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return x * 2, y * 2, w * 2, h * 2
    return None


def crop_face(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    det = detect_face(frame)
    if det is None:  # 回退：中心 80% 裁剪
        h, w = frame.shape[:2]
        x, y = int(w * 0.1), int(h * 0.1)
        crop = frame[y:h - y, x:w - x]
        return crop, False
    x, y, w, h = det
    m = int(0.4 * max(w, h))
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, frame.shape[1]), min(y + h + m, frame.shape[0])
    return frame[y0:y1, x0:x1], True


def sample_frames(video_path: Path, t: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, n - 1, t).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def to_clip_input(crops: list[np.ndarray]) -> torch.Tensor:
    """不等大的人脸裁剪 → 统一 224×224 → normalized float tensor [N,3,224,224]。"""
    crops = [cv2.resize(c, (FACE_SIZE, FACE_SIZE)) for c in crops]
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
    return (x - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.finalize:
        finalize()
        return

    df = load_chsims()
    sids = sorted(df['sample_id'].tolist())
    start = args.start or 0
    end = min(args.end or len(sids), len(sids))
    todo = [s for s in sids[start:end]]

    store = {'feat': {}, 'face_frac': {}}
    if PARTIAL.exists():
        store = torch.load(PARTIAL, weights_only=False)
    todo = [s for s in todo if s not in store['feat']]
    if not todo:
        print(f'[{start}:{end}] 无待处理样本（已完成 {len(store["feat"])}）')
        return

    model = CLIPModel.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_face = 0
    n_frames = 0
    for k, sid in enumerate(todo):
        vid, cid = sid.rsplit('_', 1)
        frames = sample_frames(RAW_VIDEO_ROOT / vid / f'{cid}.mp4', T_FRAMES)
        if not frames:
            print(f'  ! 解码失败 {sid}，置零向量')
            store['feat'][sid] = torch.zeros(512)
            store['face_frac'][sid] = 0.0
            continue
        crops, flags = [], []
        for f in frames:
            c, ok = crop_face(f)
            crops.append(c)
            flags.append(ok)
        x = to_clip_input(crops).to(args.device)
        # 等价于 get_image_features：vision [CLS] → 投影到 512
        vo = model.vision_model(pixel_values=x)
        emb = model.visual_projection(vo.pooler_output)  # [N,512]
        store['feat'][sid] = emb.mean(0).cpu()          # 全局平均池化
        store['face_frac'][sid] = float(np.mean(flags))
        n_face += sum(flags)
        n_frames += len(flags)
        if (k + 1) % 50 == 0:
            torch.save(store, PARTIAL)
            print(f'[{start}:{end}] {k+1}/{len(todo)} '
                  f'人脸检出率 {n_face/max(n_frames,1):.1%}')
    torch.save(store, PARTIAL)
    print(f'[{start}:{end}] 完成 {len(todo)} 条，累计 {len(store["feat"])}，'
          f'本块人脸检出率 {n_face/max(n_frames,1):.1%}')


def finalize():
    store = torch.load(PARTIAL, weights_only=False)
    df = load_chsims()
    missing = [s for s in df['sample_id'] if s not in store['feat']]
    assert not missing, f'缺少 {len(missing)} 条特征，如 {missing[:3]}'
    data = {}
    for mode, sub in split_by_mode(df).items():
        data[mode] = {
            'sample_id': sub['sample_id'].tolist(),
            'feat': torch.stack([store['feat'][s] for s in sub['sample_id']]),
            'label3': torch.tensor(sub['label3'].values),
            'label_m': torch.tensor(sub['label_m'].values, dtype=torch.float32),
            'face_frac': torch.tensor(
                [store['face_frac'][s] for s in sub['sample_id']]),
        }
        print(f'{mode:>5}: {len(sub)} 条, feat {data[mode]["feat"].shape}, '
              f'平均人脸检出率 {data[mode]["face_frac"].mean():.1%}')
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': MODEL_NAME, 'pooling': f'mean_over_{T_FRAMES}_frames',
                'dim': 512, 'splits': data}, FINAL)
    print('已保存:', FINAL)


if __name__ == '__main__':
    main()
