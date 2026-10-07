"""Record visual QA of a specific, frozen probe set after inspecting qa.html."""
import argparse
import json
from pathlib import Path

from core import digest, json_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--accept", action="store_true")
    a = ap.parse_args()
    record = json.loads((a.probes / "record.json").read_text(encoding="utf-8"))
    if digest(a.probes / "samples.jsonl") != record["samples_sha256"]:
        raise ValueError("Placement file changed")
    for t in record["templates"]:
        if digest(a.probes / t["file"]) != t["sha256"]:
            raise ValueError("Template file changed")
    json_write(a.probes / "qa_review.json", {"accepted": a.accept, "reviewer": a.reviewer, "notes": a.notes,
              "record_sha256": digest(a.probes / "record.json"), "samples_sha256": record["samples_sha256"],
              "scope": "Visual synthesis QA, not physical validation or detector performance approval."})
    print(a.probes / "qa_review.json")


if __name__ == "__main__":
    main()
