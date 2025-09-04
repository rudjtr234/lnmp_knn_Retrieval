#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN 최근접 거리 기반 META/NON-META 타일 판정 + 슬라이드 집계(JSON 출력)
- 배경(non-meta) 임베딩으로 임계치 tau를 지정(분위수 q%)
- 각 타일의 '배경까지의 최근접 kNN 거리'가 tau보다 크면 META, 작으면 NON-META
- 슬라이드 집계: META 타일 비율이 RATIO 이상이고, META 타일 수가 MIN_META_COUNT 이상이면 슬라이드 META
- 최종 출력: [{ "id": "<slide>.svs", "metastasis": 0/1 }, ...]
"""

import os, re, json
from pathlib import Path
from typing import List, Tuple
from visualize_result import make_overlay_slide
import numpy as np
import torch
from PIL import Image, ImageFile
from chromadb import PersistentClient
from tqdm import tqdm

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

import faiss   # GPU KNN 라이브러리

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 설정
# =========================
BASE_ROOT = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data")
BASE_ROOT_2 = Path("/home/mts/ssd_16tb/member/jks/lnmp_RAG/inference/result")
BASE_ROOT_3 = Path("/home/mts/ssd_16tb/member/jks/lnmp_RAG/inference")
VERSION_DIR = BASE_ROOT / "V1.0.0"
ROOT_DIR = VERSION_DIR
TEST_IDS_TXT = BASE_ROOT_3 / "test_ids.txt"
DB_PATH  = BASE_ROOT / "vectordb/lnmp_RAG_embedding_db_v0.2.0"

COLLECTION_NAME_BG   = "lnmp_non_meta_tile_embeddings_UNI2"

K_NEIGHBORS     = 5
Q_PERCENT       = 99.99
MAX_BG          = None
SLIDE_RATIO_THR = 0.005
MIN_META_COUNT  = 10

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SLIDE_RE = re.compile(r"(BC_\d+_\d+)")

# =========================
# 디바이스 & 모델
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)
if torch.cuda.is_available():
    print("   - GPU count:", torch.cuda.device_count())

timm_kwargs = {
    "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
    "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667*2,
    "num_classes": 0, "no_embed_class": True,
    "mlp_layer": SwiGLUPacked, "act_layer": torch.nn.SiLU,
    "reg_tokens": 8, "dynamic_img_size": True,
}

model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
if torch.cuda.device_count() > 1:
    print("✅ Multi-GPU DataParallel enabled")
    model = torch.nn.DataParallel(model)

model = model.to(device)
model.eval()
data_cfg = resolve_data_config({}, model=model)
transform = create_transform(**data_cfg)

# =========================
# 유틸
# =========================
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def read_test_ids(path: Path) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def list_tiles(slide_dir: Path) -> List[Path]:
    return sorted([p for p in slide_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])

@torch.no_grad()
def embed_images(img_list, batch_size=256):
    embs = []
    for i in range(0, len(img_list), batch_size):
        batch = img_list[i:i+batch_size]
        x = torch.stack([transform(img) for img in batch]).to(device)

        if isinstance(model, torch.nn.DataParallel):
            feats = model.module.forward_features(x)
        else:
            feats = model.forward_features(x)

        if feats.ndim == 3:
            feats = feats[:, 0, :]  # CLS 토큰
        elif feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))  # GAP

        feats = l2_normalize(feats, dim=-1)
        embs.append(feats.detach().cpu())

    return torch.cat(embs, dim=0).numpy().astype(np.float32)

def build_bg_matrix_from_chroma(client, collection_name: str, max_bg: int = None):
    col = client.get_or_create_collection(collection_name)
    cnt = col.count()
    if cnt == 0:
        raise RuntimeError(f"[배경 임베딩 누락] 컬렉션 '{collection_name}' 이 비어있습니다.")
    limit = cnt if max_bg is None else min(cnt, max_bg)
    got = col.get(include=["embeddings"], limit=limit)
    return np.array(got["embeddings"], dtype=np.float32)

def batched_search(index, embs: np.ndarray, k: int, batch_size: int = 64):
    all_dist, all_idx = [], []
    for i in range(0, len(embs), batch_size):
        batch = embs[i:i+batch_size]
        batch = batch.reshape(batch.shape[0], -1)

        if batch.shape[1] != index.d:
            raise ValueError(f"[Shape Mismatch] batch dim={batch.shape[1]}, index dim={index.d}")

        d, idx = index.search(batch.astype(np.float32), k)
        all_dist.append(d)
        all_idx.append(idx)

    return np.vstack(all_dist), np.vstack(all_idx)

def calibrate_tau_with_bg_faiss_gpu(
    X_bg: np.ndarray, k: int, q_percent: float, batch_size: int = 16, device: int = 0
):
    """τ 계산은 GPU 한 장에서만 (안정적)"""
    d = X_bg.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, device, cpu_index)

    gpu_index.add(X_bg.astype(np.float32))

    distances, _ = batched_search(gpu_index, X_bg, k+1, batch_size=batch_size)
    scores_bg = distances[:, 1]
    tau = float(np.percentile(scores_bg, q_percent))
    return tau, scores_bg

def slide_decision(meta_count: int, total_tiles: int, ratio_thr: float, min_meta: int) -> str:
    ratio = (meta_count / max(total_tiles, 1))
    return "META" if (meta_count >= min_meta and ratio >= ratio_thr) else "NON_META"

# =========================
# 메인
# =========================
def main():
    print("=== LNMP_RAG: META/NON-META 추론(FAISS-KNN-τ, 5-NN 만장일치, Single GPU) ===")

    # -------------------------
    # 테스트 슬라이드 목록
    # -------------------------
    test_ids = read_test_ids(TEST_IDS_TXT)
    print(f"- 테스트 슬라이드 수: {len(test_ids)}")

    # -------------------------
    # 배경 임베딩 로드
    # -------------------------
    client = PersistentClient(path=str(DB_PATH))
    X_bg = build_bg_matrix_from_chroma(client, COLLECTION_NAME_BG, MAX_BG)
    print(f"- 배경 임베딩 로드 완료 (shape={X_bg.shape})")

    # -------------------------
    # τ 계산 (GPU0 단일 카드에서만)
    # -------------------------
    tau, bg_scores = calibrate_tau_with_bg_faiss_gpu(
        X_bg, k=1, q_percent=Q_PERCENT, batch_size=16, device=0
    )
    print(f"- tau (Q={Q_PERCENT}%): {tau:.6f}")

    # -------------------------
    # 추론용 인덱스 (GPU0 단일 카드)
    # -------------------------
    d = X_bg.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # GPU0 한 장만 사용
    index.add(X_bg.astype(np.float32))

    # -------------------------
    # 슬라이드별 추론
    # -------------------------
    per_slide = []

    for sid in tqdm(test_ids, desc="Slides"):
        slide_dir = ROOT_DIR / sid / sid
        if not slide_dir.exists():
            alt_dir = ROOT_DIR / sid
            if alt_dir.exists():
                slide_dir = alt_dir

        tiles = list_tiles(slide_dir)
        if len(tiles) == 0:
            per_slide.append({"id": f"{sid}.svs", "metastasis": 0})
            continue

        # 타일 로드
        imgs = []
        for tp in tiles:
            try:
                imgs.append(Image.open(tp).convert("RGB"))
            except:
                continue

        # === GPU 임베딩 ===
        embs = embed_images(imgs, batch_size=256)

        # === FAISS KNN 검색 ===
        distances, _ = batched_search(index, embs, K_NEIGHBORS, batch_size=16)
        scores = distances[:, 1:K_NEIGHBORS+1]

        # ✅ 5개 모두 tau보다 커야 META
        is_meta = np.all(scores > tau, axis=1)

        # 슬라이드 집계
        meta_cnt = int(np.sum(is_meta))
        all_cnt = len(is_meta)
        slide_pred = slide_decision(meta_cnt, all_cnt, SLIDE_RATIO_THR, MIN_META_COUNT)

        per_slide.append({
            "id": f"{sid}.svs",
            "metastasis": 1 if slide_pred == "META" else 0
        })

        print(f"  - {sid}: tiles={all_cnt}, META={meta_cnt} "
              f"({(meta_cnt/all_cnt*100):.3f}%), -> {slide_pred}")

        # 🔥 Overlay JPG 저장
        make_overlay_slide(
            slide_id=sid,
            tiles=tiles,
            is_meta=is_meta,
            tile_size=512,
            out_dir=BASE_ROOT_2 / "wsi_image"
        )

    # -------------------------
    # 결과 JSON 저장
    # -------------------------
    out_json = BASE_ROOT_2 / f"lnmp_predictions_v.0.1.0.json"
    with open(out_json, "w") as f:
        json.dump(per_slide, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 저장 완료: {out_json}")


if __name__ == "__main__":
    main()
