#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNMP_RAG: Tile-level embedding with UNI2-h (timm)
- non_meta_ids_300.json 에 적힌 non-metastasis 슬라이드만 임베딩
- 라벨은 전부 "nonmetastasis" 고정
- GPU 3,4번만 사용 (DataParallel)
- 배치 단위로 임베딩 처리 (batch_size=64)
"""

import os
import json
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import timm
from chromadb import PersistentClient
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ======================
# ✅ Config
# ======================
# GPU 3,4번만 사용하도록 고정
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 슬라이드 디렉토리 루트
ROOT_DIR = Path("/data/143/airflow_shared/mnt/dataset/breast/images/patches/V1.0.0")

# non-meta ID JSON (리스트)
IDS_JSON = Path("/data/143/airflow_shared/mnt/dataset/breast/cv/lnmp/V1.0.0-build.lnmp/20x/non_meta_ids_300.json")

# ✅ VectorDB 저장 위치 (144 SSD)
CHROMA_PATH = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/vectordb/lnmp_knn_embedding_db_v0.1.1")
COLLECTION_NAME = "lnmp_tile_embeddings_UNI2_nonmeta_300"

# 이미지 확장자 허용
IMG_EXTS = (".jpg", ".jpeg", ".png")

# ChromaDB add 배치 크기
ADD_BATCH = 5000

# 모델 입력 배치 크기
BATCH_SIZE = 256  # VRAM 상황에 맞게 조절

# ======================
# ✅ Model (UNI2-h) & Transform
# ======================
timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}
# timm 모델 생성
model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    **timm_kwargs,
)

# transform은 DataParallel 감싸기 전에 timm 모델에서 가져와야 함
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# 여러 GPU 병렬 사용 (GPU 3,4번)
if torch.cuda.device_count() > 1:
    print(f"✅ {torch.cuda.device_count()} GPUs 사용: {list(range(torch.cuda.device_count()))}")
    model = torch.nn.DataParallel(model)

model = model.to(DEVICE)
model.eval()

# ======================
# ✅ ChromaDB
# ======================
chroma_client = PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ======================
# ✅ Utils
# ======================
def list_tiles_recursive(root: Path) -> list[Path]:
    exts = {e.lower() for e in IMG_EXTS}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def load_id_list_json(json_path: Path) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} is not a list of IDs")
    return [str(x).strip() for x in data if x]

@torch.inference_mode()
def get_embeddings_batch(img_paths: list[Path]) -> np.ndarray:
    imgs = []
    for p in img_paths:
        with Image.open(p) as im:
            imgs.append(transform(im.convert("RGB")))
    tensor = torch.stack(imgs).to(DEVICE)

    use_amp = (DEVICE.type == "cuda")
    if use_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            feats = model(tensor)  # [batch, 1536]
    else:
        feats = model(tensor)

    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.detach().cpu().numpy()

def embed_and_store(slide_root: Path, slide_id: str, label: str):
    files = list_tiles_recursive(slide_root)
    if not files:
        print(f"⚠️ 타일 없음: {slide_root}")
        return

    collection.delete(where={"slide_id": slide_id})

    embeddings, ids, metadatas = [], [], []
    for i in tqdm(range(0, len(files), BATCH_SIZE), desc=f"Embedding {slide_id}", leave=False):
        batch_paths = files[i:i+BATCH_SIZE]
        batch_embs = get_embeddings_batch(batch_paths)

        for p, emb in zip(batch_paths, batch_embs):
            rel = p.relative_to(slide_root)
            ids.append(f"{slide_id}_{rel.as_posix()}")
            embeddings.append(emb.tolist())
            metadatas.append({
                "slide_id": slide_id,
                "tile_name": rel.name,
                "tile_path": str(p),
                "label": label,
            })

        if len(ids) >= ADD_BATCH:
            collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
            embeddings, ids, metadatas = [], [], []

    if ids:
        collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    print(f"📊 {slide_id} 타일 수: {len(files)} @ {slide_root}")

# ======================
# ✅ Main
# ======================
if __name__ == "__main__":
    # 1) 임베딩 대상 슬라이드
    ids = load_id_list_json(IDS_JSON)

    # 2) 실제 존재하는 폴더만 필터
    exist_ids = [sid for sid in ids if (ROOT_DIR / sid).is_dir()]
    miss = set(ids) - set(exist_ids)
    if miss:
        print(f"⚠️ 존재하지 않는 폴더 {len(miss)}개 스킵: {sorted(list(miss))[:5]} ...")

    print(f"임베딩 대상 슬라이드: {len(exist_ids)} (from {IDS_JSON.name})")

    # 3) 슬라이드별 임베딩 (라벨 전부 nonmetastasis)
    for slide_id in exist_ids:
        label = "nonmetastasis"
        slide_dir = ROOT_DIR / slide_id
        try:
            embed_and_store(slide_dir, slide_id, label)
            print(f"✅ 저장 완료: {slide_id} ({label})")
        except Exception as e:
            print(f"❌ 오류 발생: {slide_id} → {e}")
