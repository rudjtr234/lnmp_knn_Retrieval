#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNMP_RAG: Tile-level embedding with UNI2-h (timm)
- 라벨은 test_report.json의 GT만 사용: "metastasis" | "nonmetastasis"
- IDS_TXT에 적힌 슬라이드 디렉토리만 임베딩
"""

import os
import json
from pathlib import Path
from typing import List, Dict
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 슬라이드 디렉토리 루트 (V1.0.0 바로 아래에 슬라이드 폴더들이 있어야 함)
ROOT_DIR = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/V1.0.0")

# 임베딩할 슬라이드 ID 목록 (한 줄에 하나의 디렉토리명) - 예: train_ids.txt 혹은 test_ids.txt
IDS_TXT = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/train_ids.txt")

# ✅ test_report (GT) 경로: JSON만 사용
# 예: {"BC_01_0001": "metastasis", "BC_01_0002": "nonmetastasis", ...}
TEST_REPORT_JSON = Path("/home/mts/ssd_16tb/member/jks/lnmp_RAG/embedding/test_report.json")

# ChromaDB (persistent)
CHROMA_PATH = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/vectordb/lnmp_RAG_embedding_db_v0.1.0")
COLLECTION_NAME = "lnmp_tile_embeddings_UNI2"

# 이미지 확장자 허용
IMG_EXTS = (".jpg", ".jpeg", ".png")


# 이미지 확장자 허용 (이미 있다면 그대로 사용)
IMG_EXTS = (".jpg", ".jpeg", ".png")

def list_tiles_recursive(root: Path) -> list[Path]:
    """root 이하 모든 하위 디렉토리에서 png/jpg 타일을 재귀적으로 수집"""
    exts = {e.lower() for e in IMG_EXTS}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def resolve_slide_root(slide_id: str) -> Path:
    """
    슬라이드 루트 결정:
    1) 기본: ROOT_DIR/slide_id
    2) 같은 이름 하위 폴더가 있으면: ROOT_DIR/slide_id/slide_id (예: .../BC_01_0527/BC_01_0527)
    """
    base = ROOT_DIR / slide_id
    nested = base / slide_id
    return nested if nested.is_dir() else base


# ChromaDB add 배치 크기 (타일 수 많을 때 메모리 안정화)
ADD_BATCH = 5000

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
model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    **timm_kwargs,
).to(DEVICE)
model.eval()

# 권장 전처리
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# ======================
# ✅ ChromaDB
# ======================
chroma_client = PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ======================
# ✅ Utils
# ======================




def load_id_list(txt_path: Path) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _norm_label(s: str) -> str:
    s = str(s).strip().lower()
    if s in {"1", "metastasis", "meta", "mets", "positive", "pos"}:
        return "metastasis"
    if s in {"0", "nonmetastasis", "non-metastasis", "non_meta", "nonmeta",
             "negative", "neg", "normal"}:
        return "nonmetastasis"
    return s  # 예상 밖 값은 그대로 반환(아래에서 방어)

def load_label_map_json(json_path: Path) -> Dict[str, str]:
    if not json_path.exists():
        raise FileNotFoundError(f"TEST_REPORT_JSON 미존재: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map: Dict[str, str] = {}

    # 스키마 A) { "BC_01_0001": "metastasis", ... }
    if isinstance(data, dict) and all(isinstance(v, (str, int)) for v in data.values()):
        label_map = {k: _norm_label(v) for k, v in data.items()}

    # 스키마 B) [ {"id":"...","label":"..."}, {"id":"...","report":"..."} ... ]
    elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
        for row in data:
            sid = (row.get("id") or row.get("slide_id") or "").strip()
            lbl = row.get("label")
            if lbl is None:
                lbl = row.get("report")  # ← 사용자가 준 구조 지원
            if not sid or lbl is None:
                continue
            label_map[sid] = _norm_label(lbl)

    # 스키마 C) {"metastasis":{"train":[paths],...}, "non-metastasis":{...}}
    elif isinstance(data, dict) and any(k in data for k in ["metastasis", "non-metastasis", "nonmetastasis"]):
        def collect_ids(split_dict: dict) -> list[str]:
            ids = []
            for split_name in ("train", "valid", "test"):
                paths = split_dict.get(split_name, [])
                for p in paths:
                    sid = Path(p).name.strip()
                    if sid:
                        ids.append(sid)
            return ids

        if "metastasis" in data and isinstance(data["metastasis"], dict):
            for sid in collect_ids(data["metastasis"]):
                label_map[sid] = "metastasis"

        nonmeta_key = None
        for cand in ["non-metastasis", "nonmetastasis"]:
            if cand in data and isinstance(data[cand], dict):
                nonmeta_key = cand
                break
        if nonmeta_key:
            for sid in collect_ids(data[nonmeta_key]):
                label_map[sid] = "nonmetastasis"

    else:
        raise ValueError(f"test_report.json 형식 오류(지원 스키마 A/B/C): {json_path}")

    # 알 수 없는 라벨 방어
    bad = {k: v for k, v in label_map.items() if v not in {"metastasis", "nonmetastasis"}}
    if bad:
        print(f"⚠️ 알 수 없는 라벨 {len(bad)}건: 예시 {list(bad.items())[:5]}")

    print(f"✅ 라벨 로드 완료: {len(label_map)} slides from {json_path}")
    if label_map:
        print("   예시:", list(label_map.items())[:5])
    return label_map

@torch.inference_mode()
def get_embedding(img_path: Path) -> np.ndarray:
    # PIL Image는 컨텍스트 매니저로 안전하게 열기
    with Image.open(img_path) as im:
        image = im.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    use_amp = (DEVICE.type == "cuda")
    if use_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            feat = model(tensor)  # [1, 1536]
    else:
        feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, dim=-1)  # L2 정규화
    return feat.squeeze(0).detach().cpu().numpy()

def embed_and_store(slide_root: Path, slide_id: str, label: str):
    files = list_tiles_recursive(slide_root)   # ← 재귀적으로 전부 찾기
    if not files:
        print(f"⚠️ 타일 없음: {slide_root}")
        return

    # 기존 동일 slide_id 데이터 삭제 후 재삽입
    collection.delete(where={"slide_id": slide_id})

    embeddings, ids, metadatas = [], [], []
    for img_path in tqdm(files, desc=f"Embedding {slide_id}", leave=False):
        emb = get_embedding(img_path)
        embeddings.append(emb.tolist())

        # 고유 ID는 파일명만 써도 되지만, 중복 방지를 위해 상대경로 사용 권장
        rel = img_path.relative_to(slide_root)
        ids.append(f"{slide_id}_{rel.as_posix()}")

        metadatas.append({
            "slide_id": slide_id,
            "tile_name": rel.name,
            "tile_path": str(img_path),
            "label": label,
        })

        if len(ids) >= ADD_BATCH:
            collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
            embeddings, ids, metadatas = [], [], []

    if ids:
        collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)

    print(f"📊 {slide_id} 타일 수: {len(files)} @ {slide_root}")
    print(json.dumps(metadatas[:2], ensure_ascii=False, indent=2) if metadatas else "  (empty)")

# ======================
# ✅ Main
# ======================
if __name__ == "__main__":
    # 1) 임베딩 대상 슬라이드
    ids = load_id_list(IDS_TXT)

    # 2) GT 라벨 로드 (test_report.json)
    label_map = load_label_map_json(TEST_REPORT_JSON)

    # 3) 실제 존재하는 폴더만 필터
    exist_ids = [sid for sid in ids if (ROOT_DIR / sid).is_dir()]
    miss = set(ids) - set(exist_ids)
    if miss:
        print(f"⚠️ 존재하지 않는 폴더 {len(miss)}개 스킵: {sorted(list(miss))[:5]} ...")

    # 4) 라벨 매핑 누락 확인
    no_label = [sid for sid in exist_ids if sid not in label_map]
    if no_label:
        print(f"⚠️ GT 라벨 누락 {len(no_label)}개: 예시 {no_label[:10]}")
        print("   → 누락 슬라이드는 기본값 'nonmetastasis'로 처리합니다.")

    # 5) 진행 정보
    print(f"임베딩 대상 슬라이드: {len(exist_ids)} (from {IDS_TXT.name})")
    print(f"라벨 보유 슬라이드:   {len(label_map)} (from {TEST_REPORT_JSON.name})")

    # 6) 슬라이드별 임베딩
    for slide_id in exist_ids:
        label = label_map.get(slide_id, "nonmetastasis")
        if label not in {"metastasis", "nonmetastasis"}:
            print(f"⚠️ 알 수 없는 라벨 값 '{label}' → 'nonmetastasis'로 대체 ({slide_id})")
            label = "nonmetastasis"

        slide_dir = ROOT_DIR / slide_id
        try:
            embed_and_store(slide_dir, slide_id, label)
            print(f"✅ 저장 완료: {slide_id} ({label})")
        except Exception as e:
            print(f"❌ 오류 발생: {slide_id} → {e}")
