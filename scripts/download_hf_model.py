"""Download a Hugging Face model locally and prepare it for WinSCP transfer."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_REPO = "Qwen/Qwen3-4B"
DEFAULT_OUTPUT = Path("D:/models/Qwen3-4B")
MANIFEST_NAME = "TRANSFER_MANIFEST.json"
VERIFY_SCRIPT_NAME = "verify_transfer.py"


VERIFY_SCRIPT = r'''"""Verify files copied from the download machine. Uses only Python stdlib."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    manifest = json.loads((args.model_dir / "TRANSFER_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for expected in manifest["files"]:
        path = args.model_dir / expected["path"]
        if not path.is_file():
            errors.append(f"missing: {expected['path']}")
            continue
        if path.stat().st_size != expected["size"]:
            errors.append(f"size mismatch: {expected['path']}")
            continue
        if sha256(path) != expected["sha256"]:
            errors.append(f"sha256 mismatch: {expected['path']}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"OK: {len(manifest['files'])} files, {manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(model_dir: Path) -> list[Path]:
    return sorted(
        path for path in model_dir.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(model_dir).parts
        and path.name not in {MANIFEST_NAME, VERIFY_SCRIPT_NAME}
    )


def validate_snapshot(model_dir: Path) -> None:
    required = [model_dir / "config.json", model_dir / "tokenizer_config.json"]
    missing = [str(path.name) for path in required if not path.is_file()]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_files = sorted(set(index.get("weight_map", {}).values()))
        missing.extend(name for name in weight_files if not (model_dir / name).is_file())
    elif not any(model_dir.glob("*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        raise RuntimeError("Downloaded snapshot is incomplete: " + ", ".join(missing))


def write_transfer_artifacts(model_dir: Path, repo_id: str, revision: str) -> dict:
    rows = []
    total_bytes = 0
    files = model_files(model_dir)
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(model_dir).as_posix()
        size = path.stat().st_size
        print(f"hashing [{index}/{len(files)}] {relative}", flush=True)
        rows.append({"path": relative, "size": size, "sha256": sha256(path)})
        total_bytes += size
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_bytes": total_bytes,
        "files": rows,
    }
    (model_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / VERIFY_SCRIPT_NAME).write_text(VERIFY_SCRIPT, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    resolved_revision = args.revision
    if not args.verify_only:
        info = HfApi(token=args.token).model_info(args.repo_id, revision=args.revision)
        resolved_revision = info.sha
        print(f"downloading {args.repo_id}@{resolved_revision} to {args.output}", flush=True)
        snapshot_download(
            repo_id=args.repo_id,
            revision=resolved_revision,
            local_dir=args.output,
            token=args.token,
            max_workers=args.max_workers,
        )

    validate_snapshot(args.output)
    manifest = write_transfer_artifacts(args.output, args.repo_id, resolved_revision)
    print(json.dumps({
        "status": "ready",
        "path": str(args.output.resolve()),
        "repo_id": args.repo_id,
        "revision": resolved_revision,
        "files": len(manifest["files"]),
        "size_gb": round(manifest["total_bytes"] / 1024**3, 2),
        "server_verify": "python verify_transfer.py .",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
