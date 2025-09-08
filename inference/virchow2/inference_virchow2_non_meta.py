#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN 최근접 거리 기반 META/NON-META 타일 판정 + 슬라이드 집계(JSON 출력, Virchow2 전용)
- 배경(non-meta) 임베딩으로 임계치 tau를 지정(분위수 q%)
- 각 타일의 '배경까지의 최근접 kNN 거리'가 tau보다 크면 META, 작으면 NON-META
- 슬라이드 집계: META 타일 비율이 RATIO 이상이고, META 타일 수가 MIN_META_COUNT 이상이면 슬라이드 META
- 최종 출력: [{ "id": "<slide>.svs", "metastasis": 0/1 }, ...]
"""

import os, re, json
from pathlib import Path
from typing import List
from visualize_result import make_overlay_slide
import numpy as np
import torch
from PIL import Image, ImageFile
from chromadb import PersistentClient
from tqdm import tqdm
from timm.layers import SwiGLUPacked
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import faiss   # GPU KNN 라이브러리

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 설정
# =========================
BASE_ROOT = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data")
BASE_ROOT_2 = Path("/home/mts/ssd_16tb/member/jks/lnmp_knn_Retrieval/inference/result")
BASE_ROOT_3 = Path("/home/mts/ssd_16tb/member/jks/lnmp_knn_Retrieval/inference/")
VERSION_DIR = BASE_ROOT / "V1.0.1"
ROOT_DIR = VERSION_DIR
TEST_IDS_TXT = BASE_ROOT_3 / "label/answer_label.txt"
DB_PATH  = BASE_ROOT / "vectordb/lnmp_RAG_embedding_db_v0.3.0"

# 👉 Virchow2 전용 배경 DB
COLLECTION_NAME_BG   = "lnmp_non_meta_tile_embeddings_Virchow2"

K_NEIGHBORS     = 5
Q_PERCENT       = 99.99
MAX_BG          = None
SLIDE_RATIO_THR = 0.005
MIN_META_COUNT  = 10

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SLIDE_RE = re.compile(r"(BC_\d+_\d+)")

# =========================
# 디바이스 & 모델 (Virchow2, Multi-GPU for embedding)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)
if torch.cuda.is_available():
    print("   - GPU count:", torch.cuda.device_count())

# Virchow2 timm 모델 로드 (공식 권장 방식)
base_model = timm.create_model(
    "hf-hub:paige-ai/Virchow2",
    pretrained=True,
    num_classes=0,
    mlp_layer=SwiGLUPacked,       # 반드시 지정
    act_layer=torch.nn.SiLU,      # 반드시 지정
).to(device).eval()

# ✅ DataParallel 적용 (GPU 0,1,2 사용)
if torch.cuda.device_count() > 1:
    print("✅ Multi-GPU DataParallel enabled (using GPU 0,1,2 for embedding)")
    model = torch.nn.DataParallel(base_model, device_ids=[0, 1, 2])
else:
    model = base_model

# Virchow2 전처리 transform
data_cfg = resolve_data_config(model.module.pretrained_cfg if isinstance(model, torch.nn.DataParallel) else model.pretrained_cfg, model=model)
transform = create_transform(**data_cfg)


@torch.no_grad()
def embed_images(img_list, batch_size=256):
    """
    Virchow2 기반 타일 임베딩 추출 (Multi-GPU DataParallel 지원)
    출력: numpy array (N, 2560)
         (CLS 토큰 1280 + 패치 평균 1280 → concat → normalize)
    """
    embs = []
    for i in range(0, len(img_list), batch_size):
        batch = img_list[i:i+batch_size]
        if not batch:
            continue

        x = torch.stack([transform(img) for img in batch]).to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16):  # Mixed precision 권장
            out = model(x)  # (B, 261, 1280)  → DataParallel이면 자동으로 분산 처리

        cls_tok = out[:, 0]                # (B, 1280)
        patch_toks = out[:, 5:]            # (B, 256, 1280), 앞 4개는 register tokens 제외
        pooled_patch = patch_toks.mean(1)  # (B, 1280)

        z = torch.cat([cls_tok, pooled_patch], dim=-1)  # (B, 2560)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)

        embs.append(z.cpu().numpy().astype("float32"))

    if not embs:
        return np.empty((0, 2560), dtype=np.float32)
    return np.concatenate(embs, axis=0)

# =========================
# 유틸
# =========================
def read_test_ids(path: Path) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def list_tiles(slide_dir: Path) -> List[Path]:
    return sorted([p for p in slide_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])

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
        d, idx = index.search(batch.astype(np.float32), k)
        all_dist.append(d)
        all_idx.append(idx)
    return np.vstack(all_dist), np.vstack(all_idx)

def calibrate_tau_with_bg_faiss_gpu(
    X_bg: np.ndarray, k: int, q_percent: float, batch_size: int = 16, device: int = 0
):
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
    print("=== LNMP_KNN (Virchow2): META/NON-META 추론 (FAISS-KNN-τ, 5-NN 만장일치, Single GPU) ===")

    # 테스트 슬라이드 목록
    test_ids = read_test_ids(TEST_IDS_TXT)
    print(f"- 테스트 슬라이드 수: {len(test_ids)}")

    # 배경 임베딩 로드
    client = PersistentClient(path=str(DB_PATH))
    X_bg = build_bg_matrix_from_chroma(client, COLLECTION_NAME_BG, MAX_BG)
    print(f"- 배경 임베딩 로드 완료 (shape={X_bg.shape})")

    # τ 계산
    tau, bg_scores = calibrate_tau_with_bg_faiss_gpu(
        X_bg, k=1, q_percent=Q_PERCENT, batch_size=16, device=0
    )
    print(f"- tau (Q={Q_PERCENT}%): {tau:.6f}")

    # 추론용 인덱스
    d = X_bg.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    index.add(X_bg.astype(np.float32))

    # 슬라이드별 추론
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

        # GPU 임베딩
        embs = embed_images(imgs, batch_size=256)

        # FAISS KNN 검색
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

        # Overlay JPG 저장
        make_overlay_slide(
            slide_id=sid,
            tiles=tiles,
            is_meta=is_meta,
            tile_size=512,
            out_dir=BASE_ROOT_2 / "wsi_image_virchow2"
        )

    # 결과 JSON 저장
    out_json = BASE_ROOT_2 / f"lnmp_predictions_v0.3.0.json"
    with open(out_json, "w") as f:
        json.dump(per_slide, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 저장 완료: {out_json}")


if __name__ == "__main__":
    main()

