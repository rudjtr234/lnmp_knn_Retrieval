
# cal_tau_only.py
import json
import time
import faiss
import numpy as np
from pathlib import Path

# =========================
# 설정
# =========================
BASE_ROOT_2 = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/vectordb/faiss")

BASE_ROOT_3 = Path("/home/mts/ssd_16tb/member/jks/lnmp_breast_data/tau/")

# 기존 저장된 index / tau 파일
INDEX_FILE = BASE_ROOT_2 / "hnsw_index_v0.1.0.faiss"
TAU_FILE   = BASE_ROOT_3 / "tau_params_v0.1.4.json"

# quantile 값 (원하는 대로 바꿔서 실험 가능)
Q_LOW   = 98
Q_HIGH  = 99.99

# =========================
# 실행
# =========================
def main():
    start_t = time.time()

    # --- FAISS Index 불러오기 ---
    print(f"[Load] FAISS index 불러오는 중: {INDEX_FILE}")
    hnsw_index = faiss.read_index(str(INDEX_FILE))
    d = hnsw_index.d

    # --- index 안의 벡터 수 확인 ---
    ntotal = hnsw_index.ntotal
    print(f"[Index Info] vectors={ntotal}, dim={d}")

    # --- index에서 거리 분포 다시 계산 ---
    print(f"[τ 계산] 시작: q_low={Q_LOW}, q_high={Q_HIGH}")
    dists, _ = hnsw_index.search(hnsw_index.reconstruct_n(0, ntotal), 2)
    top1_dist = dists[:, 1]

    tau_high = float(np.percentile(top1_dist, Q_HIGH))
    tau_low  = float(np.percentile(top1_dist, Q_LOW))

    # --- 저장 ---
    with open(TAU_FILE, "w") as f:
        json.dump({"tau_low": tau_low, "tau_high": tau_high}, f, indent=2)

    print(f"[τ 결과] tau_low={tau_low:.6f}, tau_high={tau_high:.6f}")
    print(f"[Save] τ 값 저장 완료: {TAU_FILE}")
    print(f"✅ Done. Elapsed={ (time.time()-start_t)/60:.2f} min")

if __name__ == "__main__":
    main()
