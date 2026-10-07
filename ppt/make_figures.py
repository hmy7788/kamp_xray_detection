"""보고서·발표용 그림을 결과 파일에서 직접 만든다 (숫자를 손으로 옮기지 않음).

  python ppt/make_figures.py            # ppt/figures/ 에 PNG 저장

입력: data/manifest.csv·라벨, runs/minyeop/**/ (group_stats, 05~07 결과, 점 제거 결과), docs/experiments.md 의 평가 v2 값(아래 V2에 옮겨 둠)
모델 색은 모든 그림에서 같다: R50 파랑, YOLOv3-tiny 주황, MobileNetV3 초록.
복사만 하는 그림(오류 확대, 합성 점 예시, 그룹 그림)은 각 분석 스크립트가 만든 그림을 그대로 쓴다.
"""
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ppt" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RUN = ROOT / "runs" / "minyeop"

f = "C:/Windows/Fonts/malgun.ttf"
font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

MODELS = ["Faster R-CNN R50-FPN", "YOLOv3-tiny", "Faster R-CNN MobileNetV3-FPN"]
SHORT = {MODELS[0]: "R50", MODELS[1]: "YOLOv3-tiny", MODELS[2]: "MobileNetV3"}
COL = {MODELS[0]: "#3b6fd4", MODELS[1]: "#e8912d", MODELS[2]: "#4aa564"}
DPI = 150


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("저장", name)


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


# ------------------------------------------------------------------ 0. 이야기 흐름
def fig0_story():
    fig, ax = plt.subplots(figsize=(14, 3.7))
    ax.axis("off")
    steps = [("① 학습·검증·평가", "6개 모델\n임계값은 val에서 고정\ntest는 한 번", "#dde6f7"),
             ("② 몇 개 못 찾음", "정답 663개 중 6~9개\nF1 0.99 안팎", "#fbe7cc"),
             ("③ 라벨 문제", "공식 라벨 박스 크기\n중심 거리로 재채점\n→ 거의 다 찾음", "#d6ecdc"),
             ("④ 일반화·robust", "점 제거 · 가짜 정상\n합성 점(진하기·크기·\n위치) · 호기별", "#f5d9d9"),
             ("⑤ 종합 → 베스트", "val 동률 선두\n미탐지 강건성 최상\n→ D-FINE-N", "#e6dcf0"),
             ("⑥ 현장 활용", "3단 판정·재검사\n모델 선택 가이드", "#f1ecd2")]
    n = len(steps)
    step_w, box_w = 2.3, 2.0
    for i, (t, b, c) in enumerate(steps):
        x = i * step_w
        ax.add_patch(FancyBboxPatch((x, 0.2), box_w, 2.6, boxstyle="round,pad=0.05,rounding_size=0.15", fc=c, ec="#555", lw=1.2))
        ax.text(x + box_w / 2, 2.35, t, ha="center", va="center", fontsize=12.5, fontweight="bold")
        ax.text(x + box_w / 2, 1.2, b, ha="center", va="center", fontsize=10.5, linespacing=1.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + box_w + 0.08, 1.5), (x + step_w - 0.05, 1.5), arrowstyle="-|>", mutation_scale=16, color="#444", lw=1.6))
    ax.set_xlim(-0.2, n * step_w - 0.1)
    ax.set_ylim(0, 3.1)
    ax.set_title("실험 흐름: 학습·평가 → 못 찾은 것의 원인 → 일반화·robust 평가 → 베스트 모델", fontsize=14, pad=8)
    save(fig, "fig0_story_flow.png")


# ------------------------------------------------------------------ 1. 데이터 이해
def fig1_data():
    man = {r["image_id"]: r for r in read_csv(ROOT / "data/manifest.csv")}
    sides = {"official": [], "team": []}
    for sp in ("train", "val", "test"):
        for lab in (ROOT / "data" / sp / "labels").glob("*.txt"):
            m = man[lab.stem]
            W, H = int(m["width"]), int(m["height"])
            for ln in lab.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    _, cx, cy, bw, bh = (float(v) for v in ln.split()[:5])
                    sides[m["source"]].append(max(bw * W, bh * H))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    bins = np.arange(4, 23, 1)
    ax.hist(sides["team"], bins=bins, color="#9aa7b8", label=f"팀 라벨 ({len(sides['team']):,}개): 항상 10px 또는 약 13px", alpha=0.95)
    ax.hist(sides["official"], bins=bins, color="#d9534f", label=f"공식 라벨 ({len(sides['official']):,}개): 5~21px로 제각각", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("정답 박스 한 변 길이 (px)")
    ax.set_ylabel("박스 수 (로그 눈금)")
    ax.set_title("① 라벨 두 종류: 크기가 고정된 팀 라벨이 점수를 부풀린다", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax = axes[1]
    groups = {}
    for m in man.values():
        k = f"{m['machine']}호기\n{m['width']}x{m['height']}"
        d = groups.setdefault(k, {"official": 0, "team": 0})
        d[m["source"]] += 1
    keys = sorted(groups)
    off = [groups[k]["official"] for k in keys]
    team = [groups[k]["team"] for k in keys]
    ax.bar(keys, team, color="#9aa7b8", label="팀 라벨 이미지")
    ax.bar(keys, off, bottom=team, color="#d9534f", label="공식 라벨 이미지")
    for i, k in enumerate(keys):
        ax.text(i, team[i] + off[i] + 12, f"{team[i] + off[i]}", ha="center", fontsize=9)
    ax.set_ylabel("이미지 수 (총 2,532장)")
    ax.set_title("② 호기·해상도 구성 (해상도는 호기에 고정되지 않음)", fontsize=12)
    ax.legend(fontsize=9)
    save(fig, "fig1a_data_labels.png")


# ------------------------------------------------------------------ 2. 모델·평가
def fig2_saturation():
    g = read_csv(RUN / "07_group_stats_v1/group_stats_test.csv")
    go = read_csv(RUN / "07_group_stats_v1/group_stats_test_official.csv")
    val = lambda rows, m, k: float(next(r for r in rows if r["모델"] == m and r["그룹"] == "전체")[k])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(2)
    for ax, key, title, ylim in ((axes[0], "F1", "F1 (IoU 0.5): 세 모델이 사실상 같다 (세로축 0.94~1.0)", (0.94, 1.0)),
                                  (axes[1], "mAP50-95", "mAP50-95: R50이 앞선다", (0, 0.7))):
        for i, m in enumerate(MODELS):
            vals = [val(g, m, key), val(go, m, key)]
            b = ax.bar(x + (i - 1) * 0.26, vals, width=0.25, color=COL[m], label=SHORT[m])
            for xx, v in zip(x + (i - 1) * 0.26, vals):
                ax.text(xx, v + (0.0012 if key == "F1" else 0.008), f"{v:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["전체(test)", "공식 라벨만"])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.04), frameon=False)
    save(fig, "fig2a_saturation.png")


V2 = {  # docs/experiments.md "평가 v2" 표 (R = 1, 1.5, 2, 3, 5). F1, 공식 라벨 F1
    "R": [1, 1.5, 2, 3, 5],
    MODELS[0]: ([0.771, 0.953, 0.988, 1.000, 1.000], [0.565, 0.870, 0.959, 1.000, 1.000]),
    MODELS[1]: ([0.719, 0.934, 0.982, 0.998, 1.000], [0.492, 0.829, 0.943, 0.995, 1.000]),
    MODELS[2]: ([0.682, 0.900, 0.965, 0.997, 0.998], [0.477, 0.777, 0.912, 0.995, 1.000]),
}


def fig2_center():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, idx, title in ((axes[0], 0, "전체 (test)"), (axes[1], 1, "공식 라벨만 (test)")):
        for m in MODELS:
            ax.plot(V2["R"], V2[m][idx], "-o", color=COL[m], label=SHORT[m], lw=2)
        ax.axvspan(2.8, 5.3, color="#e8f3e8", alpha=0.7)
        ax.text(4.0, 0.76, "R ≥ 3px: 세 모델 모두\n거의 만점(포화)", ha="center", fontsize=10.5, color="#2b6a2b")
        ax.axvline(2, color="#999", ls=":", lw=1)
        ax.text(2.05, 0.60, "IoU 0.5는 R 2px와\n비슷한 엄격도", fontsize=9, color="#666")
        ax.set_xlabel("중심 거리 허용 R (px)")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xticks(V2["R"])
    axes[0].set_ylabel("F1 (평가 v2: 중심 거리 기준)")
    axes[0].legend(fontsize=10, loc="lower right")
    axes[0].set_ylim(0.4, 1.02)
    fig.suptitle("찾았는가는 R ≥ 3px에서 포화, 모델 차이는 R 1~2px(위치 정밀도)에서만 보인다", fontsize=13, y=1.02)
    save(fig, "fig2b_center_distance.png")


def fig2_tradeoff():
    d = {MODELS[1]: dict(params=8.7, gpu=4.8, cpu4=43, cpu1=142, map=0.558),
         MODELS[2]: dict(params=19.0, gpu=11.3, cpu4=73, cpu1=149, map=0.559),
         MODELS[0]: dict(params=41.4, gpu=48.4, cpu4=756, cpu1=2073, map=0.611)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, xl in ((axes[0], "gpu", "GPU 추론 시간 (ms/장, 배치 1)"), (axes[1], "cpu4", "CPU 4스레드 추론 시간 (ms/장, 배치 1)")):
        for m, v in d.items():
            ax.scatter(v[key], v["map"], s=v["params"] * 38, color=COL[m], alpha=0.8, edgecolor="#333")
            dx, dy, ha = {MODELS[1]: (-12, -42, "left"), MODELS[2]: (0, 28, "center"), MODELS[0]: (0, 36, "center")}[m]
            ax.annotate(f"{SHORT[m]}  {v['params']}M", (v[key], v["map"]), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=10.5)
        ax.set_xscale("log")
        ticks = [3, 5, 10, 20, 50, 100] if key == "gpu" else [30, 50, 100, 300, 1000, 4000]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) for x in ticks])
        ax.set_xlim((3, 130) if key == "gpu" else (28, 6500))
        ax.set_xlabel(xl + "  → 느림")
        ax.set_ylabel("test mAP50-95  ↑ 정확")
        ax.set_ylim(0.52, 0.65)
        ax.grid(alpha=0.3)
    axes[1].axvline(4000, color="#c00", ls="--", lw=1.2)
    axes[1].text(3800, 0.535, "촬영 간격 약 4초", color="#c00", ha="right", fontsize=10)
    fig.suptitle("정확도(mAP50-95)와 속도: R50이 가장 정확하고 가장 느리다 (원 크기 = 파라미터 수)", fontsize=13, y=1.02)
    save(fig, "fig2c_accuracy_speed.png")


# ------------------------------------------------------------------ 3. 오류·강건성 분석
def fig3_dot_removal():
    def n_left(tag, model):
        s = json.loads((RUN / "03_dot_removal_v1" / tag / "summary.json").read_text(encoding="utf-8"))
        return 100 * s["모델"][model]["점 지운 뒤"]["정답 자리 검출 수"] / 663
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (model, title) in zip(axes, (("Faster R-CNN", "Faster R-CNN R50"), ("YOLOv3-tiny", "YOLOv3-tiny"))):
        xs = [9, 13, 17]
        ax.plot(xs, [n_left(f"mean_half{h}", model) for h in (4, 6, 8)], "-o", color="#2b8a3e", lw=2.2, label="평균 보간 (노이즈 없음)")
        ax.plot(xs, [n_left(f"ns_half{h}", model) for h in (4, 6, 8)], "-s", color="#d9534f", lw=2.2, label="Navier-Stokes + 노이즈")
        ax.plot([13], [n_left("meannoise_half6", model)], "D", color="#8e44ad", ms=9, label="평균 보간 + 노이즈 (13x13)")
        ax.axhline(100, color="#999", ls=":")
        ax.text(9.1, 96, "원본: 100%", fontsize=9, color="#666")
        ax.set_xticks(xs)
        ax.set_xticklabels(["9x9", "13x13", "17x17"])
        ax.set_xlabel("지운 정사각형 크기 (px, 실제 점은 약 3px)")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("점을 지운 뒤 정답 자리에서 남은 검출 (%)")
    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("점을 충분히 지우면 검출이 사라진다 → 모델은 점 자체를 본다 (위치만 보는 지름길이 아님)", fontsize=13, y=1.02)
    save(fig, "fig3b_dot_removal.png")


def fig3_false_alarm():
    s = json.loads((RUN / "05_fake_normal_v1/summary.json").read_text(encoding="utf-8"))["결과"]
    tags = [("mean_half6", "평균 보간\n13x13"), ("mean_half8", "평균 보간\n17x17"), ("ns_half6", "N-S+노이즈\n13x13"), ("ns_half8", "N-S+노이즈\n17x17")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(tags))
    for i, m in enumerate(MODELS):
        v = [100 * s[t][m]["오경보율"] for t, _ in tags]
        ax.bar(x + (i - 1) * 0.26, v, width=0.25, color=COL[m], label=SHORT[m])
        for xx, vv in zip(x + (i - 1) * 0.26, v):
            ax.text(xx, vv + 1, f"{vv:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in tags])
    ax.set_ylabel("오경보 이미지 비율 (%) — 점을 지운 369장 중")
    ax.set_title("가짜 정상 이미지의 오경보: 지우는 방식에 따라 0~64%. 노이즈 무늬에 R50만 강건하다", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, 0.62, "진짜 정상 제품이 아니라\n범위로 해석할 것", transform=ax.transAxes, ha="right", fontsize=10, color="#555", bbox=dict(fc="#fff7e0", ec="#d9b44a"))
    save(fig, "fig3c_false_alarm.png")


def synth_rows():
    rows = read_csv(RUN / "06_synth_insert_v1/per_dot.csv")
    for r in rows:
        r["contrast"] = float(r["contrast"])
        for m in MODELS:
            r[m + "|det"] = int(r[m + "|det"])
            r[m] = float(r[m])
    return rows


def fig3_contrast_curve(rows):
    edges = [-99, 3, 6, 9, 13, 18, 25, 99]
    mids = [1.5, 4.5, 7.5, 11, 15.5, 21.5, 29]
    labs = ["<3", "3~6", "6~9", "9~13", "13~18", "18~25", "25+"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, site, title in ((axes[0], "orig", "원래 점이 있던 자리에 넣은 점"), (axes[1], "rand", "제품 안 무작위 자리에 넣은 점")):
        for m in MODELS:
            ys = []
            for i in range(7):
                sel = [r for r in rows if r["site"] == site and edges[i] <= r["contrast"] < edges[i + 1]]
                ys.append(100 * np.mean([r[m + "|det"] for r in sel]))
            ax.plot(range(7), ys, "-o", color=COL[m], lw=2.2, label=SHORT[m])
        ax.axvspan(4.6, 6.4, color="#e8f3e8", alpha=0.6)
        ax.text(5.5, 5, "실제 점의 대비\n(중앙값 22.8, 하위 10% 약 17)", ha="center", fontsize=9, color="#2b6a2b")
        ax.set_xticks(range(7))
        ax.set_xticklabels(labs)
        ax.set_xlabel("점의 대비 (둘레 밝기에서 점 밝기를 뺀 값, 0~255)")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("합성 점 검출률 (%)")
    axes[0].legend(fontsize=10, loc="upper left")
    fig.suptitle("옅은 점(대비 약 13 미만)에서 놓치기 시작한다. MobileNet은 무작위 자리에서 거의 못 찾는다", fontsize=13, y=1.02)
    save(fig, "fig3d_synth_contrast_curve.png")


def fig3_heatmap():
    ss = json.loads((RUN / "06_synth_insert_v1/summary.json").read_text(encoding="utf-8"))["검출률"]
    S = [0.15, 0.3, 0.5, 1.0]
    F = [0.67, 1.0, 1.5]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, m in enumerate(MODELS):
        for i, (site, st) in enumerate((("orig", "원래 자리"), ("rand", "무작위 자리"))):
            ax = axes[i][j]
            Z = np.array([[100 * ss[m][f"{site}|s{s}|f{f}"]["검출률"] for f in F] for s in S])
            ax.imshow(Z, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
            for a in range(len(S)):
                for b in range(len(F)):
                    ax.text(b, a, f"{Z[a][b]:.0f}", ha="center", va="center", fontsize=11, color="#111")
            ax.set_xticks(range(len(F)))
            ax.set_xticklabels([f"{f}배" for f in F])
            ax.set_yticks(range(len(S)))
            ax.set_yticklabels([f"{s}" for s in S])
            ax.set_title(f"{SHORT[m]} · {st}", fontsize=11)
            if j == 0:
                ax.set_ylabel("깊이 배율 s (1.0 = 실제 세기)")
            if i == 1:
                ax.set_xlabel("크기 배율 f")
    fig.suptitle("합성 점 검출률(%) 격자 — 옅고 작을수록 놓친다", fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, "fig3e_synth_heatmap.png")


def fig4_threshold(rows):
    fa = read_csv(RUN / "05_fake_normal_v1/per_image.csv")
    thr_val = {MODELS[0]: 0.95, MODELS[1]: 0.06, MODELS[2]: 0.93}
    ts = np.linspace(0.02, 0.99, 98)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, m in zip(axes, MODELS):
        faint = [r[m] for r in rows if 9 <= r["contrast"] < 18 and r["site"] == "orig"]
        det = [100 * np.mean([x >= t for x in faint]) for t in ts]
        ax.plot(ts, det, color=COL[m], lw=2.4, label="옅은 점(대비 9~18) 검출률")
        for tag, ls, lab in (("mean_half6", "--", "가짜 정상 오경보 (평균 보간)"), ("ns_half6", ":", "가짜 정상 오경보 (노이즈 보간)")):
            sc = [float(r["가짜정상 최고신뢰도"]) for r in fa if r["설정"] == tag and r["모델"] == m]
            ax.plot(ts, [100 * np.mean([x >= t for x in sc]) for t in ts], ls, color="#c0392b", lw=2, label=lab)
        ax.axvline(thr_val[m], color="#333", lw=1.2)
        ax.set_title(SHORT[m], fontsize=12)
        ax.set_xlabel(f"신뢰도 임계값 (세로선 = val 고정 {thr_val[m]})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8.5, loc="center left")
    fig.suptitle("임계값을 낮추면 옅은 점 검출은 늘지만 오경보 증가폭은 모델마다 다르다 (합성·가짜 정상 기준, 참고용)", fontsize=12.5, y=1.04)
    save(fig, "fig4a_threshold_tradeoff.png")


def copy_figs():
    pairs = {RUN / "02_yolov3tiny_img640_v1/figures/test/miss_cases.png": "fig3a_fn_cases.png",
             RUN / "07_group_stats_v1/figures/group_chart_test.png": "fig3f_groups_chart.png",
             RUN / "07_group_stats_v1/figures/group_samples_test.png": "fig3h_group_samples.png",
             RUN / "06_synth_insert_v1/figures/synth_check.png": "fig3g_synth_examples.png",
             RUN / "06_synth_insert_v1/figures/synth_full.png": "fig3i_synth_full.png",
             RUN / "08_synth_eval_v1/figures/synth_eval.png": "fig3j_synth_eval_bars.png",
             RUN / "09_extra_models_v1/figures/compare6.png": "fig3k_compare6.png",
             RUN / "09_extra_models_v1/figures/robust6.png": "fig3l_robust6.png",
             RUN / "09_extra_models_v1/figures/final_dfine.png": "fig4b_final_dfine.png",
             RUN / "09_extra_models_v1/figures/v2_sweep_6models.png": "fig7a_v2_sweep.png",
             RUN / "09_extra_models_v1/figures/dot_removal_6models.png": "fig7b_dot_removal_6models.png",
             RUN / "12_group_stats_6models_v1/figures/group_chart_test_6models.png": "fig7c_groups_6models.png"}
    for src, dst in pairs.items():
        shutil.copy(src, OUT / dst)
        print("복사", dst)


def fig3_bar_preview():
    """막대 안 합성 점 데이터(scripts/synth_eval.py)의 미리보기: 호기마다 이미지 1장, 원본(초록=실제 점)과 합성(하늘색=막대 윤곽). 임시 폴더에 데이터를 다시 만든다(시드 42, 약 1분)."""
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("synth_eval", ROOT / "scripts" / "synth_eval.py")
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test"
        dots, imgs, skipped, extra, man = se.generate("test", out, 4, se.SEED)
        se.preview(out, "test", extra, man, n_per_machine=1, scale=1.3)
        shutil.copy(out / "preview_overview.png", OUT / "fig3m_synth_bars.png")
    print("생성 fig3m_synth_bars.png")


if __name__ == "__main__":
    fig0_story()
    fig1_data()
    fig2_saturation()
    fig2_center()
    fig2_tradeoff()
    fig3_dot_removal()
    fig3_false_alarm()
    rows = synth_rows()
    fig3_contrast_curve(rows)
    fig3_heatmap()
    fig4_threshold(rows)
    copy_figs()
    fig3_bar_preview()
    print("완료:", OUT)
