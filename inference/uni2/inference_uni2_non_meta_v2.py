


# -*- coding: utf-8 -*-

"""
KNN 최근접 거리 기반 META/NON-META 타일 판정 + 슬라이드 집계(JSON 출력, UNI2-DDP 개선 버전)
- HNSW 기반 ANN으로 global τ 계산 (저장/불러오기 지원)
- 모든 rank에서 임베딩 계산 + 슬라이드별 진행상황 출력
- Rank 0에서만 최종 결과 합쳐서 JSON 저장
"""

import os, json, time
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image, ImageFile
from chromadb import PersistentClient
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import faiss

from visualize_result import make_overlay_slide

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 설정
# =========================
BASE_ROOT   = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data")
BASE_ROOT_2 = Path("/home/mts/ssd_16tb/member/jks/lnmp_knn_Retrieval/inference/result")
BASE_ROOT_3 = Path("/home/mts/ssd_16tb/member/jks/lnmp_knn_Retrieval/inference")
VERSION_DIR = BASE_ROOT / "V1.0.1"
ROOT_DIR    = VERSION_DIR
TEST_IDS_TXT = BASE_ROOT_3 / "label/answer_label.txt"
DB_PATH     = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/vectordb/lnmp_knn_embedding_db_v0.1.1")
COLLECTION_NAME_BG = "lnmp_tile_embeddings_UNI2_nonmeta_300"

K_NEIGHBORS     = 5
Q_LOW           = 99
Q_HIGH          = 99.99
MAX_BG          = None
SLIDE_RATIO_THR = 0.005
MIN_META_COUNT  = 10
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".bmp"}

# τ 저장 경로
TAU_FILE   = BASE_ROOT_2 / "tau_params.json"
INDEX_FILE = BASE_ROOT_2 / "hnsw_index.faiss"

# =========================
# DDP Setup
# =========================
from datetime import timedelta

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(hours=24)   # ← 12시간으로 늘림
    )
    torch.cuda.set_device(rank)

def cleanup():
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"[Rank Cleanup] NCCL 종료 오류 무시: {e}", flush=True)

# =========================
# 유틸 함수
# =========================
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def read_test_ids(path: Path) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def list_tiles(slide_dir: Path) -> List[Path]:
    return sorted([p for p in slide_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])

@torch.no_grad()
def embed_images(img_list, model, transform, device, batch_size=256):
    embs = []
    for i in range(0, len(img_list), batch_size):
        batch = img_list[i:i+batch_size]
        x = torch.stack([transform(img) for img in batch]).to(device, non_blocking=True)
        feats = model.module.forward_features(x) if isinstance(model, DDP) else model.forward_features(x)
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        elif feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        feats = l2_normalize(feats, dim=-1)
        embs.append(feats.detach().cpu())
    return torch.cat(embs, dim=0).numpy().astype(np.float32)

def build_bg_matrix_from_chroma(client, collection_name: str, max_bg: int = None):
    col = client.get_or_create_collection(collection_name)
    cnt = col.count()
    if cnt == 0:
        raise RuntimeError(f"[배경 임베딩 누락] '{collection_name}' 비어있음")
    limit = cnt if max_bg is None else min(cnt, max_bg)
    got = col.get(include=["embeddings"], limit=limit)
    return np.array(got["embeddings"], dtype=np.float32)

def calibrate_tau_with_bg_faiss_hnsw(X_bg: np.ndarray, k: int,
                                     q_percent_high: float, q_percent_low: float,
                                     M: int = 32, efSearch: int = 64):
    d = X_bg.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efSearch = efSearch
    index.verbose = True
    index.add(X_bg.astype(np.float32))
    print(f"[HNSW] Index 구축 완료: {len(X_bg)} vectors, dim={d}")

    dists, _ = index.search(X_bg, k+1)
    top1_dist = dists[:, 1]
    tau_high = float(np.percentile(top1_dist, q_percent_high))
    tau_low  = float(np.percentile(top1_dist, q_percent_low))
    return tau_low, tau_high, top1_dist, index

def slide_decision(meta_count: int, total_tiles: int, ratio_thr: float, min_meta: int) -> str:
    ratio = (meta_count / max(total_tiles, 1))
    return "META" if (meta_count >= min_meta and ratio >= ratio_thr) else "NON_META"

# =========================
# Main Worker
# =========================
def main_worker(rank, world_size):
    setup(rank, world_size)

    # === 모델 ===
    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667*2,
        "num_classes": 0, "no_embed_class": True,
        "mlp_layer": SwiGLUPacked, "act_layer": torch.nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": True,
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    model.eval()
    data_cfg = resolve_data_config({}, model=model.module)
    transform = create_transform(**data_cfg)

    # === Rank0: τ 계산 또는 불러오기 ===
    if rank == 0:
        if TAU_FILE.exists() and INDEX_FILE.exists():
            with open(TAU_FILE, "r") as f:
                tau = json.load(f)
            tau_low, tau_high = tau["tau_low"], tau["tau_high"]
            hnsw_index = faiss.read_index(str(INDEX_FILE))
            print(f"[Rank0] τ 불러옴: tau_low={tau_low:.6f}, tau_high={tau_high:.6f}")
        else:
            client = PersistentClient(path=str(DB_PATH))
            X_bg = build_bg_matrix_from_chroma(client, COLLECTION_NAME_BG, MAX_BG)
            tau_low, tau_high, _, hnsw_index = calibrate_tau_with_bg_faiss_hnsw(
                X_bg, k=1, q_percent_high=Q_HIGH, q_percent_low=Q_LOW,
                M=32, efSearch=64
            )
            with open(TAU_FILE, "w") as f:
                json.dump({"tau_low": tau_low, "tau_high": tau_high}, f, indent=2)
            faiss.write_index(hnsw_index, str(INDEX_FILE))
            print(f"[Rank0] τ 저장됨 + HNSW index 저장됨: {TAU_FILE}, {INDEX_FILE}")
    else:
        tau_low, tau_high, hnsw_index = None, None, None

    # === τ 값 브로드캐스트 ===
    tau_low = torch.tensor([tau_low if tau_low is not None else 0.0], device=rank)
    tau_high = torch.tensor([tau_high if tau_high is not None else 0.0], device=rank)
    dist.broadcast(tau_low, src=0)
    dist.broadcast(tau_high, src=0)
    tau_low, tau_high = tau_low.item(), tau_high.item()

    # === 테스트 슬라이드 분배 ===
    test_ids = read_test_ids(TEST_IDS_TXT)
    my_ids = test_ids[rank::world_size]
    per_slide = []
    for idx, sid in enumerate(my_ids, 1):
        slide_dir = ROOT_DIR / sid / sid
        if not slide_dir.exists():
            alt_dir = ROOT_DIR / sid
            if alt_dir.exists():
                slide_dir = alt_dir
        tiles = list_tiles(slide_dir)
        if len(tiles) == 0:
            per_slide.append({"id": f"{sid}.svs", "embs": None, "tiles": []})
            continue
        imgs = [Image.open(tp).convert("RGB") for tp in tiles if tp.exists()]
        embs = embed_images(imgs, model, transform, rank, batch_size=256)
        print(f"[Rank {rank}] {idx}/{len(my_ids)} 슬라이드 처리 완료: {sid} (tiles={len(embs)})", flush=True)
        per_slide.append({"id": f"{sid}.svs", "embs": embs, "tiles": tiles})

    # === 결과 수집 ===
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, per_slide)

    if rank == 0:
        merged = []
        for r in gather_list:
            for entry in r:
                sid = entry["id"]
                if entry["embs"] is None:
                    merged.append({"id": sid, "metastasis": 0})
                    continue
                embs = entry["embs"]
                tiles = entry["tiles"]

                # === global HNSW index에서 Top-K 검색
                dists, _ = hnsw_index.search(embs.astype(np.float32), K_NEIGHBORS)
                scores = dists

                # === META/NON-META 판정 ===
                is_meta = []
                for row in scores:
                    if np.all(row >= tau_high):
                        is_meta.append(True)
                    elif np.all(row <= tau_low):
                        is_meta.append(False)
                    else:
                        is_meta.append(np.all(row > tau_high))
                is_meta = np.array(is_meta)

                meta_cnt = int(np.sum(is_meta))
                all_cnt = len(is_meta)
                slide_pred = slide_decision(meta_cnt, all_cnt, SLIDE_RATIO_THR, MIN_META_COUNT)
                merged.append({"id": sid, "metastasis": 1 if slide_pred == "META" else 0})
                print(f"[Rank0] 최종판정 - {sid}: tiles={all_cnt}, META={meta_cnt} "
                      f"({(meta_cnt/all_cnt*100):.3f}%), -> {slide_pred}", flush=True)
                make_overlay_slide(
                    slide_id=sid.replace(".svs", ""),
                    tiles=tiles,
                    is_meta=is_meta,
                    tile_size=512,
                    out_dir=BASE_ROOT_2 / "wsi_image"
                )

        out_json = BASE_ROOT_2 / f"lnmp_predictions_v0.2.0.json"
        with open(out_json, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 전체 결과 저장 완료: {out_json}")

    cleanup()

# =========================
# 엔트리포인트
# =========================
def main():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    main_worker(rank, world_size)

if __name__ == "__main__":
    main()
