#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageFile
import torch
import numpy as np
from chromadb import PersistentClient

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 설정
# =========================
BASE_ROOT = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data")
VERSION_DIR = BASE_ROOT / "V1.0.0"
ROOT_DIR = VERSION_DIR

TEST_IDS_TXT = BASE_ROOT / "test_ids.txt"

DB_PATH  = BASE_ROOT / "vectordb/lnmp_RAG_embedding_db_v0.1.0"
COLLECTION_NAME = "lnmp_tile_embeddings_UNI2"   # 실제 컬렉션명으로 확인 완료했으면 유지

TOP_K = 3
VOTE_MODE = "weighted"       # "majority" | "weighted"
OUTPUT_PATH = Path("/home/mts/ssd_16tb/member/jks/lnmp_RAG/inference/result/predictions_v0.1.1.json")

USE_SOFTMAX = True
SOFTMAX_T   = 0.3
IMG_EXTS = (".jpg", ".jpeg", ".png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = (device.type == "cuda")

# =========================
# 모델 / 변환
# =========================
timm_kwargs = {
    "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
    "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667*2,
    "num_classes": 0, "no_embed_class": True,
    "mlp_layer": SwiGLUPacked, "act_layer": torch.nn.SiLU,
    "reg_tokens": 8, "dynamic_img_size": True,
}
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).to(device)
model.eval()
data_cfg = resolve_data_config({}, model=model)
transform = create_transform(**data_cfg)

# =========================
# DB
# =========================
client = PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# 컬렉션 비어있음 방지 안내
try:
    _cnt = collection.count()
    if _cnt == 0:
        print(f"❗ 경고: 컬렉션 '{COLLECTION_NAME}' 이 비어 있습니다. 컬렉션명을 다시 확인하세요.")
except Exception as e:
    print(f"❗ 컬렉션 카운트 실패: {e}")

# =========================
# 유틸
# =========================
SLIDE_RE = re.compile(r"(BC_\d+_\d+)")
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def softmax(x, T=1.0, eps=1e-12):
    x = np.array(x, dtype=np.float64) / max(T, eps)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + eps)

def parse_test_list(path: Path):
    """
    test_ids.txt 파싱:
      - 슬라이드 ID (BC_xx_xxxx) 또는 *.tiff → slide 리스트에 추가
      - *.png 경로/이름 → 해당 슬라이드의 tile whitelist에 추가
    한 줄에 여러 ID가 공백으로 나열되어 있어도 모두 처리.
    """
    slide_set = set()
    tile_whitelist = defaultdict(set)  # slide_id -> set(tile_basename_lower)
    if not path.exists():
        print(f"⚠️ test_ids.txt가 없습니다. 전체 슬라이드 사용: {path}")
        return slide_set, tile_whitelist, False

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            # 공백 분리된 토큰들을 모두 처리
            for s in raw.split():
                sl = s.lower()
                if sl.endswith(".png"):
                    p = Path(s)
                    tile_name = p.name.lower()  # 대소문자 무시
                    m = SLIDE_RE.search(s)
                    slide_id = m.group(1) if m else p.parent.name
                    slide_set.add(slide_id)
                    tile_whitelist[slide_id].add(tile_name)
                elif sl.endswith(".tiff") or sl.endswith(".svs"):
                    slide_id = Path(s).stem
                    slide_set.add(slide_id)
                else:
                    # 슬라이드 ID로 간주
                    slide_set.add(s)
    has_tile_filter = any(tile_whitelist.values())
    return slide_set, tile_whitelist, has_tile_filter

def resolve_search_dir(slide_root: Path) -> Path:
    """
    슬라이드 디렉토리에서 실제 타일이 있는 검색 루트 반환.
    /<sid>/<sid> 구조면 안쪽 폴더를, 아니면 현재 폴더를 반환.
    """
    nested = slide_root / slide_root.name
    return nested if nested.is_dir() else slide_root

# =========================
# 메인
# =========================
def main():
    # test_ids 로드
    sel_slides, tile_filter, has_tile_filter = parse_test_list(TEST_IDS_TXT)
    # 화이트리스트는 소문자로 통일되어 있음(위에서 lower 적용)

    if sel_slides:
        print(f"🎯 선택된 슬라이드 수: {len(sel_slides)} (tile 필터: {has_tile_filter})")
    else:
        print("🎯 선택 슬라이드 미지정 → 전체 사용")

    # 처리 대상 슬라이드 폴더(1-depth에서 ID 매칭)
    if sel_slides:
        targets = []
        missing = []
        for d in sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()]):
            if d.name in sel_slides:
                targets.append(d)
        # 누락 확인
        want = set(sel_slides)
        got = {d.name for d in targets}
        miss = sorted(list(want - got))
        if miss:
            print(f"❗ test_ids에 있으나 1-depth에 없는 슬라이드(샘플): {miss[:10]}")
        slide_dirs = targets
    else:
        slide_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])

    results = []

    # 컬렉션 크기 확인해서 top_k 보정
    try:
        col_count = collection.count()
        effective_topk = max(1, min(TOP_K, col_count))
    except Exception:
        effective_topk = TOP_K

    for slide_dir in slide_dirs:
        slide_id = slide_dir.name

        # 🔎 중첩 폴더 지원 + 재귀 수집 + 대소문자 무시
        search_dir = resolve_search_dir(slide_dir)
        tiles_all = [p for p in search_dir.rglob("*")
                     if p.is_file() and p.suffix.lower() in IMG_EXTS]

        if has_tile_filter and slide_id in tile_filter:
            want = tile_filter[slide_id]  # 이미 lower() set
            tile_paths = [p for p in tiles_all if p.name.lower() in want]
        else:
            tile_paths = tiles_all

        if not tile_paths:
            print(f"⚠️ 타일 없음: slide={slide_id}  search_dir={search_dir}  예시={[p.name for p in tiles_all[:5]]}")
            continue

        vote_scores = {"metastasis": 0.0, "nonmetastasis": 0.0}
        total_votes = 0.0

        for path in tile_paths:
            try:
                with Image.open(path) as im:
                    img = im.convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    if use_fp16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            feats = model(img_t)          # (1, 1536)
                    else:
                        feats = model(img_t)

                feats = l2_normalize(feats, dim=-1)
                embedding = feats.squeeze(0).detach().cpu().tolist()

                q = collection.query(
                    query_embeddings=[embedding],
                    n_results=effective_topk,
                    include=["metadatas", "distances"]
                )
                if not q.get("metadatas") or not q["metadatas"][0]:
                    continue

                metas = q["metadatas"][0]
                dists = q.get("distances", [[]])[0]
                sims  = [1.0 - float(d) for d in dists]

                if VOTE_MODE == "majority":
                    weights = [1.0] * len(metas)
                elif VOTE_MODE == "weighted":
                    if USE_SOFTMAX:
                        weights = softmax(sims, T=SOFTMAX_T)
                    else:
                        s = np.clip(np.asarray(sims, dtype=np.float64), 0.0, None)
                        weights = s / (s.sum() if s.sum() > 0 else 1.0)
                else:
                    raise ValueError(f"지원하지 않는 vote_mode: {VOTE_MODE}")

                for m, w in zip(metas, weights):
                    lbl = m.get("label")
                    if lbl in vote_scores:
                        vote_scores[lbl] += float(w)
                    total_votes += float(w)

            except Exception as e:
                print(f"❌ 오류: {path} → {e}")

        pred = max(vote_scores.items(), key=lambda x: x[1])[0]
        print(f"\n✅ {slide_id} | pred={pred}, scores={vote_scores}, mode={VOTE_MODE}, top_k={effective_topk}, tiles={len(tile_paths)}")
        print(f"🧮 total weighted votes: {total_votes:.4f}")
        results.append({"id": f"{slide_id}.tiff", "prediction": pred})

    # 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 결과 저장: {OUTPUT_PATH}  (n={len(results)})")

if __name__ == "__main__":
    main()
