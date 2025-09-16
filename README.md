# LNMP_KNN_Retrieval

## 📌 개요
본 프로젝트는 **유방암 림프절 전이(LNMP, Lymph Node Metastasis Prediction)** 판별을 위해
타일 단위 임베딩을 활용한 **KNN 기반 Retrieval 알고리즘**을 구현한 것입니다.

- 입력: 512 x 512 병리 타일 (WSI 기반)
- 임베딩 모델: **UNI2-h (ViT-H/14, 1536차원)**
- 벡터 검색: **FAISS k-NN (τ = 99.99% 분위수 임계치 적용)**
- 최종 판정: 다수결 기반 투표(KNN_VOTE)로 슬라이드 단위 **Non-metastasis 여부** 판정

---

## 🖼 알고리즘 개요 다이어그램
![LNMP KNN Retrieval](./image/lnmp_knn_retrieval.png)

---

## 🧩 알고리즘 절차
1. **Tile Extraction**: 512×512 크기의 WSI 타일 추출
2. **Embedding**: UNI2-h 모델로 1536차원 벡터화
3. **Indexing**: ChromaDB 및 FAISS GPU index 구축
4. **Thresholding (Fence 설정)**:
   - 상위 분위수(τ_high, 예: 99.97% ~ 99.93%) → **Fence 밖 → META 확정**
   - 하위 분위수(τ_low=98%) → **Fence 안 → NON-META 확정**
   - τ_low ≤ dist ≤ τ_high → **애매 구간 → KNN 투표 진행**
5. **Retrieval & Voting**:
   - Query 타일에 대해 FAISS L2 거리 기반 최근접 k=5 이웃 탐색
   - **5-NN 만장일치 규칙** 적용 → 이웃 모두 fence 밖일 경우 META 판정
6. **Slide-level Aggregation**:
   - META 타일 비율 ≥ **0.005**  
   - META 타일 개수 ≥ **10**  
   → 조건 만족 시 슬라이드 전체를 **Metastasis(전이)**로 최종 판정
---

## 📊 Prediction Report 예시
```json
[
  {
    "id": "BC_01_0001.svs",
    "metastasis": 1
  },
  {
    "id": "BC_01_0003.svs",
    "metastasis": 1
  },
  {
    "id": "BC_01_0004.svs",
    "metastasis": 1
  }
]

```

---

## 📊 결과(2025_09_16) 

| 버전     | VectorDB 규모 | τ 설정(Quantile) | Accuracy  | Precision | Recall (Sensitivity) | Specificity | F1-score  |
| ------ | ----------- | -------------- | --------- | --------- | -------------------- | ----------- | --------- |
| v0.2.7 | ~138만 타일   | τ=99.97% / 98% | **0.640** | **0.769** | **0.400**            | **0.880**   | **0.526** |
| v0.2.8 | ~138만 타일   | τ=99.94% / 98% | **0.560** | **0.552** | **0.640**            | **0.480**   | **0.593** |
| v0.2.9 | ~138만 타일   | τ=99.93% / 98% | **0.560** | **0.545** | **0.720**            | **0.400**   | **0.621** |


🔹 Confusion Matrix

📌 v0.2.7

|                     | Predicted Positive (Meta) | Predicted Negative (Non-Meta) |
| ------------------- | ------------------------- | ----------------------------- |
| **Actual Positive** | 22 (TP)                   | 3 (FN)                        |
| **Actual Negative** | 25 (FP)                   | 0 (TN)                        |

📌 v0.2.8

|                     | Predicted Positive (Meta) | Predicted Negative (Non-Meta) |
| ------------------- | ------------------------- | ----------------------------- |
| **Actual Positive** | 16 (TP)                   | 9 (FN)                        |
| **Actual Negative** | 11 (FP)                   | 14 (TN)                       |

📌 v0.2.9

|                     | Predicted Positive (Meta) | Predicted Negative (Non-Meta) |
| ------------------- | ------------------------- | ----------------------------- |
| **Actual Positive** | 18 (TP)                   | 7 (FN)                        |
| **Actual Negative** | 15 (FP)                   | 10 (TN)                       |
