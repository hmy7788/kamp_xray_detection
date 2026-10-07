"""Verify a portable snapshot and build PC-local paths without reading test data.

Standard library only. Run from any directory after extracting the ZIP.
"""
import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Verify every packaged file against PORTABLE_MANIFEST.json")
    args = ap.parse_args()
    if args.verify:
        manifest = json.loads((REPO / "PORTABLE_MANIFEST.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = (REPO / item["path"]).resolve()
            if not path.is_relative_to(REPO) or not path.is_file() or digest(path) != item["sha256"]:
                raise RuntimeError(f"Missing or changed packaged file: {item['path']}")
        print(f"Verified {len(manifest['files'])} packaged files.")
    comparison = REPO / "runs/chong/06_deployment_compare"
    inputs = json.loads((comparison / "work/standalone_inputs.json").read_text(encoding="utf-8"))
    if inputs["split"] != "val" or len(inputs["images"]) != 369:
        raise ValueError("Expected the frozen 369-image validation input list")
    for row in inputs["images"]:
        iid = row["id"]
        if any(c in iid for c in ("/", "\\", ":")) or iid in (".", ".."):
            raise ValueError("Invalid image ID")
        path = REPO / "data/val/images" / f"{iid}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        row["path"] = str(path)
    output = comparison / "work/standalone_inputs_portable.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created current-PC input paths: {output}")
    print("Historical experiment records were preserved. No inference or training was run.")


if __name__ == "__main__":
    main()
