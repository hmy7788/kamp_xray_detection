"""확정 데이터 요약을 data/README.txt 로 쓴다 (zip 에 같이 들어간다).

사용: python common/data_summary.py
내용: 폴더 구조, 분할별·장비별·출처별 사진 수, 라벨 상자 수, 빈 라벨 수, 해상도, 라벨 형식.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def table(df: pd.DataFrame, index: str, columns: str) -> str:
    t = pd.crosstab(df[index], df[columns], margins=True, margins_name="합계")
    return t.to_string()


def main() -> None:
    df = kx.load_manifest()
    df["resolution"] = df["width"].astype(str) + "x" + df["height"].astype(str)
    df["n_boxes"] = [len(kx.read_yolo_labels(kx.DATA / p)) for p in df.label_path]
    df["machine"] = df["machine"].astype(str) + "호기"
    df["split"] = pd.Categorical(df["split"], kx.PARTS)
    info = kx.read_json(kx.SPLIT_INFO) if kx.SPLIT_INFO.exists() else {}

    lines = []
    lines.append("KAMP X-ray 이물질 탐지 — 확정 데이터 (2026-10-05)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("폴더 구조")
    lines.append("  data/train/images/*.png, data/train/labels/*.txt   학습")
    lines.append("  data/val/images,   data/val/labels                검증 (평소 실험 채점)")
    lines.append("  data/test/images,  data/test/labels               최종 테스트 (최종 후보만 한 번 채점)")
    lines.append("  data/manifest.csv                                 사진 목록 (split, source, labeler, machine, 크기, 촬영 시각, burst_id)")
    lines.append("")
    lines.append(f"사진 수: {len(df)}장 전부 라벨 있음. 라벨 형식: YOLO txt, 한 줄에 상자 하나 '0 cx cy w h' (0~1 비율). 클래스 0 = defect(금속구)")
    lines.append(f"분할 규칙: {info.get('rule', '')} (묶음 burst 단위, seed {info.get('seed', '')}). 같은 촬영 묶음(60초 이내 연속)은 한쪽에만 들어간다")
    lines.append(f"manifest sha256: {kx.manifest_sha()}")
    lines.append("")
    lines.append("[분할 × 장비] 사진 수")
    lines.append(table(df, "split", "machine"))
    lines.append("")
    lines.append("[분할 × 출처] 사진 수  (official = KAMP 공식 라벨 500, team = 팀이 찍은 라벨 2,032)")
    lines.append(table(df, "split", "source"))
    lines.append("")
    lines.append("[장비 × 출처] 사진 수")
    lines.append(table(df, "machine", "source"))
    lines.append("")
    lines.append("[분할 × 월] 사진 수")
    lines.append(table(df, "split", "month"))
    lines.append("")
    lines.append("[장비 × 해상도] 사진 수")
    lines.append(table(df, "machine", "resolution"))
    lines.append("")
    box = df.groupby("split", observed=True)["n_boxes"].agg(["sum", lambda s: int((s == 0).sum())])
    box.columns = ["상자 수", "빈 라벨(상자 0개) 사진"]
    lines.append("[분할별 라벨 상자]")
    lines.append(box.to_string())
    lines.append(f"  전체 상자 {int(df.n_boxes.sum())}개, 사진당 상자 수 분포 {df.n_boxes.value_counts().sort_index().to_dict()}")
    lines.append("  빈 라벨은 제품이 반만 찍혀 테스트피스가 없는 사진(7/27 1호기 104장 등). 이물질 없음 사진으로 쓸 수 있다")
    lines.append("")
    lines.append("[팀 라벨 작성자]")
    lines.append(df[df.source == "team"].groupby("labeler").size().to_string())
    lines.append("")
    lines.append("상자 크기: 공식 라벨 한 변 중앙값 10px(1·2호기)·13px(3호기). 팀 라벨은 클릭 중심에 같은 크기 네모를 자동 생성")
    lines.append("데이터 계보: KAMP 원본 BMP → 중복 제거 → 장비 색상 박스 제거(주변 회색 메움) → 회색조 PNG. 저장소 archive/legacy_v1_fill/README.md")
    out = kx.DATA / "README.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
