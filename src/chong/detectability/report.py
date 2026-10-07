"""Render measured outputs as a Korean report with exportable scientific plots."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pct(x):
    return "—" if x is None else f"{100*x:.2f}%"


def maps(reports, out, filename="position_maps.png"):
    chosen = [r for r in reports if r.get("probe_maps")]
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4*len(chosen)), squeeze=False, layout="constrained")
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dddddd")
    for row, report in enumerate(chosen):
        for col, machine in enumerate(["1", "2", "3"]):
            ax = axes[row, col]
            grid = 1+max(c["row"] for c in report["probe_maps"][0]["cells"])
            data = np.full((grid, grid), np.nan)
            cells = [c for c in report["probe_maps"][0]["cells"] if c["machine"] == machine]
            for c in cells:
                if c["n"]:
                    data[c["row"], c["col"]] = c["new_hit_fraction_all"]
            view = ax.imshow(data, vmin=0, vmax=1, cmap=cmap)
            for c in cells:
                ax.text(c["col"], c["row"], f'{c["new_hit"]}/{c["n"]}' if c["n"] else 'N/A',
                        ha="center", va="center", fontsize=10, color="#181818")
            ax.set_title(f'{report["setting"]["id"]} / machine {machine}')
            ax.set_xlabel("Normalized product column")
            ax.set_ylabel("Normalized product row")
            ax.set_xticks(range(grid), range(1, grid+1))
            ax.set_yticks(range(grid), range(1, grid+1))
    fig.suptitle("New detections / tested insertions (controls must be negative)\nFixed empirical probes; not physical sensitivity or equipment miss rate", fontsize=13)
    fig.colorbar(view, ax=axes.ravel().tolist(), shrink=.65, label="Fraction (threshold 0.42)")
    fig.savefig(out / filename, dpi=170)
    plt.close(fig)


def tradeoff(reports, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout="constrained")
    yy = np.arange(len(reports))
    names = [r["setting"]["id"] for r in reports]
    delays = [r["full_api_latency"]["p95_ms"] for r in reports]
    recalls = [r["fixed_thresholds"][0]["center_recall"]*100 for r in reports]
    axes[0].barh(yy, delays, color="#257e9c")
    axes[0].set_yticks(yy, names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(delays)*1.25)
    axes[0].set_xlabel("Batch-one full API p95 (ms)")
    for y, d in zip(yy, delays):
        axes[0].text(d+max(delays)*.015, y, f"{d:.1f}", va="center", fontsize=9)
    axes[1].barh(yy, recalls, color="#419774")
    axes[1].set_xlim(max(0, min(recalls)-2), 101.5)
    axes[1].set_xlabel("Original validation center recall (%)")
    for y, d in zip(yy, recalls):
        axes[1].text(d+.05, y, f"{d:.2f}%", va="center", fontsize=9)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Same checkpoint: measured speed and retained detection\n369 validation images, threshold 0.42; current PC only", fontsize=13)
    fig.savefig(out / "speed_accuracy.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    a = ap.parse_args()
    record = json.loads((a.input / "record.json").read_text(encoding="utf-8"))
    reports = []
    for setting in record["config"]["settings"]:
        p = a.input / f'{setting["id"]}.json'
        if not p.exists():
            raise FileNotFoundError(f"Incomplete comparison: {p}")
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["weights_sha256"] != record["weights_sha256"]:
            raise ValueError("Checkpoint mismatch")
        reports.append(r)
    maps(reports, a.input)
    tradeoff(reports, a.input)
    rows, probe_rows = [], []
    for r in reports:
        s, t, timing = r["setting"], r["fixed_thresholds"][0], r["full_api_latency"]
        rows.append([s["id"], pct(r["ap50"]), pct(t["f1"]), f'{t["center_tp"]}/{r["n_gt"]}',
                     str(t["fn"]), f'{timing["p50_ms"]:.1f}', f'{timing["p95_ms"]:.1f}',
                     f'{timing["p99_ms"]:.1f}', f'{timing["serial_images_per_second"]:.1f}',
                     f'{r["model_bytes"]/2**20:.2f}',
                     f'{r["peak_sampled_rss_mib"]:.0f}',
                     "—" if r["cuda_peak_allocated_mib"] is None else f'{r["cuda_peak_allocated_mib"]:.0f}'])
        for p in r.get("probe_maps", []):
            o = p["overall"]
            probe_rows.append([s["id"], str(p["threshold"]), f'{o["new_hit"]}/{o["n"]}',
                               pct(o["new_hit_fraction_all"]), str(o["before"]), str(o["sham"]),
                               f'{o["new_hit"]}/{o["eligible"]}', pct(p["burst_macro_new_hit_fraction"])])
    header = ["설정", "AP50", "F1@.42", "중심 적중", "IoU FN", "p50 ms", "p95 ms", "p99 ms", "직렬 장/초", "파일 MiB", "RSS MiB", "GPU 할당 MiB"]
    pheader = ["설정", "임계값", "신규 검출/전체", "비율", "원본 검출", "대조 검출", "신규/대조음성", "묶음 평균"]

    def table(headers, values):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(v)}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in row)+'</tr>' for row in values)+'</tbody></table></div>'

    standalone_rows = []
    for key, label in [("fp32_t1", "FP32 / 1 thread"), ("fp32_t4", "FP32 / 4 threads"), ("int8_t1", "INT8 Conv / 1 thread")]:
        r = json.loads((a.input / f"standalone_{key}.json").read_text(encoding="utf-8"))
        standalone_rows.append([label, f'{r["p95_ms"]:.2f}', f'{r["serial_images_per_second"]:.2f}', f'{r["peak_sampled_rss_mib"]:.1f}'])
    standalone_header = ["독립 CPU 실행", "p95 ms", "직렬 장/초", "RSS MiB"]
    diagnostics = json.loads((a.input / "position_diagnostics.json").read_text(encoding="utf-8"))
    paired_rows = [[r["setting"], str(r["lost_vs_1024"]), str(r["gained_vs_1024"])] for r in diagnostics["results"]]
    paired_header = ["설정", "1024 성공 → 누락", "1024 누락 → 성공"]
    verification = json.loads((a.input / "verification.json").read_text(encoding="utf-8"))
    roi_root = a.input.parent / "07_roi_followup"
    roi_cpu = json.loads((roi_root / "roi_cpu_t1.json").read_text(encoding="utf-8"))
    roi_timing = json.loads((roi_root / "standalone_roi_t1.json").read_text(encoding="utf-8"))
    roi_gpu = json.loads((roi_root / "roi.json").read_text(encoding="utf-8"))
    if roi_cpu["weights_sha256"] != record["weights_sha256"] or roi_cpu["probes_sha256"] != record["probes_sha256"]:
        raise ValueError("ROI follow-up used different checkpoint or probe set")
    onnx_cpu = next(r for r in reports if r["setting"]["id"] == "onnx640_t1")
    maps([reports[0], onnx_cpu, roi_cpu], a.input, "roi_position_maps.png")
    roi_rows = []
    for r, speed, name in [(onnx_cpu, json.loads((a.input / "standalone_fp32_t1.json").read_text()), "CPU 640 전체 영상"),
                           (roi_cpu, roi_timing, "CPU 640 제품 주변 자르기")]:
        o, fixed = r["probe_maps"][0]["overall"], r["fixed_thresholds"][0]
        roi_rows.append([name, pct(fixed["f1"]), f'{fixed["center_tp"]}/{r["n_gt"]}',
                         f'{o["new_hit"]}/{o["n"]}', pct(o["new_hit_fraction_all"]), f'{speed["p95_ms"]:.2f}', f'{speed["peak_sampled_rss_mib"]:.1f}'])
    roi_header = ["설정", "F1@.42", "원본 중심 적중", "합성 신규 검출", "비율", "독립 실행 p95 ms", "RSS MiB"]
    rp = verification["roi_paired_vs_full640_cpu"]
    roi_claim = (f'같은 ONNX 640 모델에 제품 주변 자르기를 적용하면 기존 합성 성공 중 {rp["lost"]}곳을 놓치고, '
                 f'기존 누락 중 {rp["gained"]}곳을 새로 찾았습니다. 순증은 {rp["gained"]-rp["lost"]}곳입니다. '
                 f'원본 val 정답 상자 {roi_cpu["n_gt"]}개 중 ROI 밖으로 잘린 상자는 {roi_cpu["roi_excluded_gt"]}개였습니다.')

    baseline, small = reports[:2]
    first = baseline["probe_maps"][0]["overall"]
    size_delta = small["fixed_thresholds"][0]["center_tp"]-baseline["fixed_thresholds"][0]["center_tp"]
    synthetic_delta = small["probe_maps"][0]["overall"]["new_hit"]-first["new_hit"]
    claim = (f'1024에서 640으로 바꾸었을 때 원본 중심 적중은 {size_delta:+d}개, '
             f'합성 위치 신규 검출은 {synthetic_delta:+d}개 변했습니다. '
             '두 결과를 함께 보고 입력 크기를 선택해야 합니다.')
    next_step = ('대조 처리가 검출에 영향을 준 위치는 추가 원인 검토가 필요합니다. '
                 if first["before"] or first["sham"] else '이번 고정 위치 집합에서는 원본·대조 처리의 해당 위치 검출이 없었습니다. ')
    if first["new_hit"] < first["n"]:
        next_step += ('합성 이동 신호에서 검출 누락이 있어 후속 학습 후보를 만들 수 있습니다. '
                      '이는 실제 장비 미탐이나 위치 암기의 확정 증거가 아닙니다. '
                      '후속 증강을 한다면 train 출처만 사용하고, 동일 조건의 비증강 대조 학습과 원본 val 성능을 함께 비교해야 합니다.')
    else:
        next_step += '기본 입력에서 이번 합성 위치는 모두 검출됐습니다. 실패를 만들기 위한 임의 난이도 증가는 하지 않습니다.'
    env = record["env"]
    machine_text = f'{env["cpu"]} · {env["gpu"]} · PyTorch {env["torch"]} · Ultralytics {env["ultralytics"]}'
    payload = json.dumps([{**r["setting"], "p95": r["full_api_latency"]["p95_ms"],
                           "center": r["fixed_thresholds"][0]["center_recall"],
                           "map": r["probe_maps"][0]["overall"]["new_hit_fraction_all"] if r.get("probe_maps") else None}
                          for r in reports], ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X-ray 위치별 검출과 실행 조건</title>
<style>
:root{font-family:system-ui,'Malgun Gothic',sans-serif;color:#172637;background:#f3f5f8}body{max-width:1250px;margin:36px auto;padding:0 24px;line-height:1.7}h1{font-size:34px;line-height:1.3;margin:14px 0}h2{font-size:24px;margin-top:0}.eyebrow{letter-spacing:2px;color:#236a70;font-size:13px;font-weight:700}.lead{max-width:1000px;font-size:18px;color:#415569}.card{padding:28px;background:white;border:1px solid #dbe2e9;border-radius:14px;margin:24px 0}.note{border-left:4px solid #d69c3b;background:#fff9ed;padding:16px 20px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:11px;border-bottom:1px solid #dbe2e9;white-space:nowrap}th{background:#edf3f7}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}code{background:#edf1f6;padding:3px 7px;border-radius:4px}input[type=range]{width:min(100%,650px)}.muted{color:#63758b;font-size:14px}.ok{color:#13725e}.bad{color:#a13a30}li{margin:6px 0}a{color:#176a96}footer{margin:30px 0;color:#63758b;font-size:13px}
</style></head><body><div class="eyebrow">KAMP · 이총 · 제공 데이터 검증</div><h1>같은 이물을 옮기고,<br>연산량을 줄였을 때도 찾는가</h1>'''
    document += f'<p class="lead">{html.escape(claim)}</p><p class="muted">{html.escape(machine_text)}</p>'
    document += '<div class="note"><strong>이 결과의 범위</strong><br>실제 장비 미탐률·정상 제품 오경보율·물리적 최소 검출 크기를 측정한 결과가 아닙니다. 한 번 재학습한 모델의 원본 val 성능과, 표시 영상에 같은 이물 잔차를 이동시킨 조건을 비교했습니다. test는 열지 않았습니다.</div>'
    document += '<section class="card"><h2>원본 성능과 현재 PC 처리시간</h2><p>같은 best.pt · 배치 1 · 워밍업 제외 · PNG 읽기부터 결과 반환까지. 장/초는 평균 처리시간의 역수이며 생산라인 보장 FPS가 아닙니다.</p>'+table(header, rows)+'<img src="speed_accuracy.png" alt="원본 검증 중심 적중과 처리시간 비교"><p class="muted">F1/FN은 IoU 0.5, 중심 적중은 예측 중심이 정답 상자 안에 있는 경우입니다. RSS는 프로세스 전체의 표본 피크이고 GPU 값은 PyTorch 할당 피크입니다. PT 체크포인트와 FP32 ONNX는 저장 정밀도가 달라 파일 크기를 같은 의미로 비교할 수 없습니다.</p></section>'
    document += '<section class="card"><h2>PyTorch 없는 CPU 실행 환경</h2><p>위 통합 실험의 RSS에는 PyTorch가 포함됩니다. 실제 CPU 배포에 가까운 별도 환경에서 같은 369장을 다시 측정했습니다. 통합 실행과 독립 실행의 예측값 일치도 확인했습니다.</p>'+table(standalone_header, standalone_rows)+'<p>이번 Conv INT8 후보는 파일은 작아졌지만 CPU 처리시간이 늘고 F1도 낮아져 채택하지 않습니다. 모든 INT8 모델이 느리다는 뜻은 아닙니다.</p><p class="muted">CPU 4 threads는 여유 CPU가 충분할 때의 후보입니다. 1 thread 결과와 함께 필요한 자원을 제시하며, 장비에 4 threads를 쓸 수 있다고 가정하지 않습니다.</p></section>'
    document += '<section class="card"><h2>가상 시험편 위치 지도</h2><p>호기별 신호 2개와 배경 촬영 묶음 6개, 총 18개 배경 묶음에 762개 위치를 고정했습니다. 신호의 크기·명암·각도를 바꾸지 않았습니다. 원본 이물은 유지하고 새 위치만 채점했습니다.</p>'+table(pheader, probe_rows)+'<p>신규 검출은 삽입 후에만 그 위치를 찾은 경우입니다. 원본·대조 처리에서 이미 검출됐던 위치는 신규 성공으로 세지 않습니다. 분모가 0인 지도 칸은 미측정입니다.</p><img src="position_maps.png" alt="호기별 위치별 신규 검출 지도"><p class="muted">같은 촬영 묶음과 템플릿을 반복 사용했으므로 762개의 독립 현장 시험으로 해석하지 않습니다. 칸의 숫자는 신규 검출 수/유효 삽입 수입니다.</p><p>'+html.escape(next_step)+'</p><a href="../05_probe_map_val/qa.html">합성 방식과 원본·대조·삽입 확대 보기</a></section>'
    document += '<section class="card"><h2>평균 성능 뒤에 가려진 위치별 변화</h2><p>640 FP32는 원본 F1이 높아졌지만, 합성 지도에서는 1024에서 찾던 119곳을 놓치고 70곳을 새로 찾았습니다. 전체 성공 수만 비교하면 서로 다른 실패 위치가 가려집니다.</p>'+table(paired_header, paired_rows)+'<p>분말유크림 검사에 연결할 시나리오는 <strong>제공 영상에서 드물었던 이물 위치와 주변 영상 무늬에도 검출이 유지되는가</strong>입니다. 이물을 임의로 더 작게 만들거나 흐리게 만들지 않고, 같은 신호를 옮겨 연산량 감소의 영향을 점검했습니다.</p><p class="muted">영상 밝기·질감과 결과의 동반 변화는 관찰할 수 있지만, 이를 분말 두께나 충전 상태가 원인이라고 해석할 수는 없습니다. 호기·템플릿 차이도 함께 섞여 있습니다.</p></section>'
    document += '<section class="card"><h2>후속 개선: 빈 배경을 줄이고 한 번만 추론</h2><p>첫 지도와 제품 배치를 보고 추가한 탐색 실험입니다. 원본 영상만으로 큰 어두운 영역을 찾고, 긴 변의 12% 여백을 둡니다. 영역 추정이 불안정하면 전체 영상을 사용합니다. 같은 모델을 한 번만 호출하고 좌표를 원본 위치로 복원합니다.</p>'+table(roi_header, roi_rows)+'<p>'+html.escape(roi_claim)+'</p><img src="roi_position_maps.png" alt="1024 전체 영상, CPU 640 전체 영상, CPU 640 제품 영역의 위치별 비교"><div class="note">검증 결과를 본 뒤 제안한 개선 후보이므로 독립 시험 성능으로 제시하지 않습니다. 원본·대조·삽입 비교에서는 원본에서 구한 ROI를 고정했습니다. 모든 위치에서 개선된 것도 아닙니다.</div><p>GPU에서는 ROI 640의 p95가 '+f'{roi_gpu["full_api_latency"]["p95_ms"]:.2f}'+' ms로, 전체 영상 1024보다 느렸습니다. CPU와 GPU에서 같은 전처리의 효용이 다를 수 있습니다.</p></section>'
    document += '<section class="card"><h2>현재의 선택과 대회에서 설명할 차별점</h2><ul><li><strong>CPU 배포 기준:</strong> ONNX FP32/640 전체 영상, 단일 모델·1회 추론. 스레드는 장비에서 확보할 수 있는 자원에 맞춥니다.</li><li><strong>추가 개선 후보:</strong> CPU에서 제품 주변 자르기는 시간 증가가 작고 합성 위치 검출이 늘었습니다. 별도 후속 검증 전까지 탐색 후보로 구분합니다.</li><li><strong>GPU를 쓸 수 있다면:</strong> 이 PC에서는 1024 FP32를 유지할 이유가 있습니다. 640/FP16은 원본 성능만 보고 자동 채택하지 않습니다.</li><li><strong>차별점:</strong> 높은 원본 점수에서 멈추지 않고, 같은 이물을 이동한 고정 지도와 시간·메모리 측정을 연결해 경량화 시 놓치는 조건을 드러냈습니다. 지도에서 나온 관찰을 단일 추론 전처리 개선까지 이어서 확인했습니다.</li></ul></section>'
    document += '<section class="card"><h2>장비 사양을 모를 때 필요한 조건 제시</h2><p>아래는 통합 평가에서 측정한 현재 PC의 p95와 가정한 소프트웨어 시간 예산의 비교입니다. 실제 장비로 이식하면 같은 측정 절차를 다시 실행해야 합니다. 독립 CPU 실행 수치는 위 별도 표를 사용합니다.</p><label for="budget">영상 한 장에 쓸 수 있는 시간: <strong id="ms"></strong> ms</label><br><input id="budget" type="range" min="5" max="250" step="5" value="50"><div id="budgetTable"></div><p class="muted">50 ms는 설명용 초기값입니다. 실제 장비의 요구값을 뜻하지 않습니다. p95 통과는 하드 실시간 보장도 아니며, 카메라/PLC/배출 구동 시간을 별도로 확보해야 합니다.</p></section>'
    document += '<section class="card"><h2>해석 및 재현</h2><ul><li>이물 재질은 확인되지 않아 금속구라고 단정하지 않습니다.</li><li>과거 76% 잔존 검출은 제거 품질 문제로 위치 암기 해석을 철회했습니다.</li><li>합성은 가공 영상의 잔차를 더하는 방법입니다. X선 감쇠·두께·산란·장비 영상 보정을 재현하지 않습니다.</li><li>배포 후보는 단일 모델·한 번의 추론입니다. 지도 생성과 대조 실험은 오프라인 개발 단계에서만 합니다.</li><li>CPU 스레드 제한은 이 PC의 자원 예산 비교이며 다른 엣지 CPU의 성능 보장이 아닙니다.</li></ul><p><a href="record.json">실험 설정·환경</a> · <a href="../../../src/chong/detectability/README.md">재현 안내</a></p><p class="muted">가중치 SHA256: <code>'+record["weights_sha256"]+'</code></p></section>'
    document += '<footer>자동 생성된 측정 보고서 · 모델 선택용 검증 결과 · 실제 현장 검증은 별도</footer><script>const records='+payload+''';const input=document.getElementById('budget');function update(){const b=Number(input.value);document.getElementById('ms').textContent=b;document.getElementById('budgetTable').innerHTML='<div class="scroll"><table><tr><th>설정</th><th>측정 p95</th><th>시간 예산</th><th>원본 중심 적중</th><th>합성 신규 검출</th></tr>'+records.map(r=>`<tr><td>${r.id}</td><td>${r.p95.toFixed(1)} ms</td><td class="${r.p95<=b?'ok':'bad'}">${r.p95<=b?'범위 이내':'초과'}</td><td>${(r.center*100).toFixed(2)}%</td><td>${r.map===null?'미측정':(r.map*100).toFixed(2)+'%'}</td></tr>`).join('')+'</table></div>'}input.addEventListener('input',update);update();</script></body></html>'''
    (a.input / "report.html").write_text(document, encoding="utf-8")
    markdown = '# 위치별 검출과 배포 조건 비교\n\n'+claim+'\n\n'+machine_text+'\n\n'
    for h, rr in [(header, rows), (standalone_header, standalone_rows), (pheader, probe_rows), (paired_header, paired_rows), (roi_header, roi_rows)]:
        markdown += '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'
        markdown += ''.join('| '+' | '.join(row)+' |\n' for row in rr)+'\n'
    markdown += roi_claim+'\n\nCPU ONNX FP32/640 전체 영상을 배포 기준으로 두고, ROI는 검증 자료에서 발견한 탐색 개선 후보로 구분한다. 이번 INT8 후보는 작아졌으나 느려져 채택하지 않는다. GPU가 가능하면 1024 기준 설정의 위치별 검출을 함께 고려한다.\n\n'+next_step+'\n\n실제 장비 미탐률, 정상 제품 오경보율, mm 검출 한계를 주장하지 않는다. 임계값 .42/.10 고정, test 미사용. 자세한 조건과 그림은 report.html을 참고한다.\n'
    (a.input / "summary.md").write_text(markdown, encoding="utf-8")
    print(a.input / "report.html")


if __name__ == "__main__":
    main()
