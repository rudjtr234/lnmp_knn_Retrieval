import json
from pathlib import Path

# ===== 설정 =====
CV_DIR = Path("/home/mts/ssd_16tb/member/jks/lnmp_RAG/json/lnmp_cv")  # cv_*.json 위치
OUT_ID_REPORT = Path("all_cv_id_report.json")  # [{"id": "BC_..", "report": "metastasis|non-metastasis"}]

# test 전용만 모을지 여부 (원하면 True 로)
USE_TEST_ONLY = False

def flatten_split(dct, test_only=False):
    """cv_k.json의 클래스 딕셔너리({train:[], valid:[], test:[]})를 평탄화
       test_only=True면 'test'만 사용"""
    if test_only:
        return list(dct.get("test", []))
    out = []
    for k, v in dct.items():
        if isinstance(v, list):
            out.extend(v)
    return out

def load_all_cv(cv_dir: Path):
    json_paths = sorted(cv_dir.glob("cv_*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No cv_*.json found under: {cv_dir}")
    return json_paths

def build_wsi_label(json_paths, test_only=False):
    """모든 cv 파일에서 WSI별 report를 구축 (충돌 시 에러)"""
    wsi_seen_label = {}  # {wsi_id: 1/0}
    wsi_seen_report = {} # {wsi_id: "metastasis"/"non-metastasis"}

    for jp in json_paths:
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for cls_name, label in [("metastasis", 1), ("non-metastasis", 0)]:
            if cls_name not in data:
                continue
            flat = flatten_split(data[cls_name], test_only=test_only)
            for p in flat:
                wsi_id = Path(p).name
                # 라벨 충돌 체크
                if wsi_id in wsi_seen_label and wsi_seen_label[wsi_id] != label:
                    raise ValueError(
                        f"[라벨 충돌] {wsi_id}: 이전={wsi_seen_label[wsi_id]}, 새로운={label}, 파일={jp}"
                    )
                wsi_seen_label[wsi_id] = label
                wsi_seen_report[wsi_id] = cls_name  # "metastasis" 또는 "non-metastasis"

    return wsi_seen_report

if __name__ == "__main__":
    # 1) 모든 CV 파일 로드
    json_paths = load_all_cv(CV_DIR)

    # 2) WSI → report("metastasis"/"non-metastasis") 매핑 생성
    wsi2report = build_wsi_label(json_paths, test_only=USE_TEST_ONLY)

    # 3) 최종 형식: [{"id": <wsi_id>, "report": <"metastasis"|"non-metastasis">}, ...]
    id_report_list = [{"id": wsi_id, "report": rep}
                      for wsi_id, rep in sorted(wsi2report.items())]

    # 4) 저장
    OUT_ID_REPORT.write_text(json.dumps(id_report_list, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Done")
    print(f" - unique WSI: {len(id_report_list)}")
    print(f" - output: {OUT_ID_REPORT}")
