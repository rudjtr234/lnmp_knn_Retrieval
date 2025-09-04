#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from collections import Counter

GT_PATH   = "test_report.json"          # 예: [{ "id": "...", "report": "non-metastasis" }, ...]
PRED_PATH = "predictions_v0.1.0.json"   # 예: [{ "id": "...", "prediction": "nonmetastasis" }, ...]

def norm_id(s: str) -> str:
    s = Path(s).name
    return s.replace(".tiff", "").replace(".tif", "")

def norm_label(s) -> str:
    s = str(s).strip().lower().replace("-", "").replace(" ", "")
    if s in {"met", "meta", "positive", "pos", "1", "true"}:
        return "metastasis"
    if s in {"nonmeta", "negative", "neg", "0", "false"}:
        return "nonmetastasis"
    # 기대 라벨: metastasis / nonmetastasis
    return s

# --- GT 로드(list) → dict[id]=label ---
with open(GT_PATH, "r", encoding="utf-8") as f:
    gt_list = json.load(f)

gt = {}
for item in gt_list:
    if not isinstance(item, dict): 
        continue
    if "id" in item and "report" in item:
        sid = norm_id(item["id"])
        lab = norm_label(item["report"])
        gt[sid] = lab

# --- Pred 로드(list) → dict[id]=label ---
with open(PRED_PATH, "r", encoding="utf-8") as f:
    pred_list = json.load(f)

pred = {}
for item in pred_list:
    if not isinstance(item, dict):
        continue
    if "id" in item and "prediction" in item:
        sid = norm_id(item["id"])
        lab = norm_label(item["prediction"])
        pred[sid] = lab

# --- 교집합 기준 비교 ---
ids_inter = sorted(set(gt.keys()) & set(pred.keys()))
n = len(ids_inter)
if n == 0:
    raise RuntimeError("교집합 ID가 없습니다. 파일과 ID 포맷을 확인하세요.")

hits = 0
pairs = []
for sid in ids_inter:
    g = gt[sid]
    p = pred[sid]
    ok = (g == p) and (g in {"metastasis", "nonmetastasis"}) and (p in {"metastasis", "nonmetastasis"})
    hits += int(ok)
    pairs.append((sid, g, p, ok))

acc = hits / n

# --- 리포트 ---
print(f"✅ 교집합 ID 수: {n}")
print(f"✅ 일치 개수: {hits}")
print(f"✅ 일치도(accuracy): {acc:.4f}")

# 간단한 혼동 카운트(교집합 한정)
cm = Counter()
for _, g, p, _ in pairs:
    if g in {"metastasis", "nonmetastasis"} and p in {"metastasis", "nonmetastasis"}:
        cm[(g, p)] += 1

print("\n🧮 Confusion (교집합 기준)")
print(f" true=metastasis,  pred=metastasis:   {cm[('metastasis','metastasis')]}")
print(f" true=metastasis,  pred=nonmetastasis:{cm[('metastasis','nonmetastasis')]}")
print(f" true=nonmetastasis,pred=metastasis:   {cm[('nonmetastasis','metastasis')]}")
print(f" true=nonmetastasis,pred=nonmetastasis:{cm[('nonmetastasis','nonmetastasis')]}")

# 누락된 ID 확인(참고)
miss_in_pred = sorted(set(gt.keys()) - set(pred.keys()))
miss_in_gt   = sorted(set(pred.keys()) - set(gt.keys()))
print(f"\nℹ️ GT에만 있고 Pred에 없는 ID: {len(miss_in_pred)}개")
print(f"   예시: {miss_in_pred[:10]}")
print(f"ℹ️ Pred에만 있고 GT에 없는 ID: {len(miss_in_gt)}개")
print(f"   예시: {miss_in_gt[:10]}")
