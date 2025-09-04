#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNMP_RAG: Non-metastasis 전용 타일 임베딩 (Virchow2, timm)
- 입력: IDS_TXT(슬라이드 ID 목록; non-meta만)
- 라벨: 고정 "nonmetastasis"
"""

import json
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from chromadb import PersistentClient

# ======================
# ✅ Config
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/V1.0.0")
IDS_TXT  = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/non-meta/non_meta_dirs.txt")

CHROMA_PATH     = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/vectordb/lnmp_RAG_embedding_db_v0.3.0")
COLLECTION_NAME = "lnmp_non_meta_tile_embeddings_Virchow2"

IMG_EXTS  = (".jpg", ".jpeg", ".png")
ADD_BATCH = 5000
BATCH_SIZE = 128
FIX_LABEL = "nonmetastasis"

DELETE_BEFORE_INSERT = True

# ======================
# ✅ Helpers
# ======================
def list_tiles_recursive(root: Path) -> List[Path]:
    exts = {e.lower() for e in IMG_EXTS}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def resolve_slide_root(slide_id: str) -> Path:
    base = ROOT_DIR / slide_id
    nested = base / slide_id
    return nested if nested.is_dir() else base

def load_id_list(txt_path: Path) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    ids = sorted(set(ids))
    print(f"✅ IDS 로드: {len(ids)}개 from {txt_path}")
    return ids

def chunked(seq, n):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

# ======================
# ✅ Model (Virchow2) & Transform
# ======================
model = timm.create_model(
    "hf-hub:paige-ai/Virchow2",
    pretrained=True,
    num_classes=0,
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU,
).to(DEVICE).eval()

transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

@torch.inference_mode()
def encode_paths(paths: List[Path]) -> np.ndarray:
    embs = []
    for batch in chunked(paths, BATCH_SIZE):
        imgs = []
        for p in batch:
            try:
                with Image.open(p) as im:
                    imgs.append(transform(im.convert("RGB")))
            except Exception as e:
                print(f"[WARN] 이미지 로드 실패: {p} ({e})")
        if not imgs:
            continue

        x = torch.stack(imgs, dim=0).to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)   # (B, 261, 1280)

        cls_tok = out[:, 0]          # (B, 1280)
        patch_toks = out[:, 5:]      # (B, 256, 1280)
        pooled_patch = patch_toks.mean(1)  # (B, 1280)

        z = torch.cat([cls_tok, pooled_patch], dim=-1)  # (B, 2560)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)

        embs.append(z.cpu().numpy().astype("float32"))

    if not embs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(embs, axis=0)

# ======================
# ✅ ChromaDB
# ======================
client = PersistentClient(path=str(CHROMA_PATH))
coll = client.get_or_create_collection(COLLECTION_NAME)
print(f"🗃️ Collection ready: {COLLECTION_NAME} @ {CHROMA_PATH}")

def flush_add(ids, embs, metas):
    if not ids:
        return 0
    coll.add(ids=ids, embeddings=embs, metadatas=metas)
    n = len(ids)
    ids.clear(); embs.clear(); metas.clear()
    return n

def embed_and_store(slide_id: str):
    slide_root = resolve_slide_root(slide_id)
    if not slide_root.is_dir():
        print(f"[SKIP] 경로 없음: {slide_root}")
        return 0

    tile_paths = list_tiles_recursive(slide_root)
    if not tile_paths:
        print(f"[SKIP] 타일 없음: {slide_root}")
        return 0

    if DELETE_BEFORE_INSERT:
        try:
            coll.delete(where={"slide_id": slide_id})
        except Exception:
            pass

    embs = encode_paths(tile_paths)
    if embs.size == 0:
        print(f"[SKIP] 임베딩 실패/없음: {slide_root}")
        return 0

    add_ids, add_embs, add_metas = [], [], []
    total = 0
    for p, e in zip(tile_paths, embs):
        rel = p.relative_to(slide_root).as_posix()
        tid = f"{slide_id}/{rel}"
        add_ids.append(tid)
        add_embs.append(e.tolist())
        add_metas.append({
            "slide_id": slide_id,
            "tile_name": p.name,
            "tile_path": str(p),
            "label": FIX_LABEL,
            "source": "non-meta",
        })
        if len(add_ids) >= ADD_BATCH:
            total += flush_add(add_ids, add_embs, add_metas)

    total += flush_add(add_ids, add_embs, add_metas)
    print(f"📊 {slide_id} 타일 수: {len(tile_paths)} @ {slide_root}")
    return total

# ======================
# ✅ Main
# ======================
if __name__ == "__main__":
    slide_ids = load_id_list(IDS_TXT)
    print(f"📂 대상 슬라이드 수: {len(slide_ids)}")

    grand_total = 0
    for sid in tqdm(slide_ids, desc="Embedding non-meta slides (Virchow2)"):
        try:
            grand_total += embed_and_store(sid)
            print(f"✅ 저장 완료: {sid} (label={FIX_LABEL})")
        except Exception as e:
            print(f"❌ 오류: {sid} → {e}")

    print(f"✅ 전체 완료: 총 {grand_total} 타일 적재 (label={FIX_LABEL})")
