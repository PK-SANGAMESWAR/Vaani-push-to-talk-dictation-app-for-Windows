"""Svarah: Indian-accented English (ai4bharat/Svarah, CC-BY-4.0, gated on the HF Hub).

Access: log in with `hf auth login` and accept the terms at
https://huggingface.co/datasets/ai4bharat/Svarah (approval is automatic).

The audio lives inside parquet shards. Only the clips in the subset are written out to
evals/data/svarah/clips/ so the loader can treat them like any other file.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .common import DATA_DIR, Clip, audio_info, select_subset, sha256_file

REPO = "ai4bharat/Svarah"
REVISION = "ebbf7777fe771490696a3f7b007097606fa8c924"  # pinned 2026-10-24
SHARDS = [f"data/test-0000{i}-of-00003.parquet" for i in range(3)]
META_COLS = ["duration", "text", "gender", "age-group", "primary_language", "native_place_state"]


def _clip_id(audio_path: str) -> str:
    return PurePosixPath(audio_path.replace("\\", "/")).stem


def _download_shards(root: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return [
            Path(hf_hub_download(REPO, s, repo_type="dataset", revision=REVISION, local_dir=root / "hub"))
            for s in SHARDS
        ]
    except GatedRepoError as e:
        raise RuntimeError(
            "Svarah is gated. Run `hf auth login`, accept the terms at "
            "https://huggingface.co/datasets/ai4bharat/Svarah, then re-run prepare."
        ) from e


def prepare(name: str, n: int, seed: int, full: bool) -> tuple[dict, list[Clip]]:
    import pyarrow.parquet as pq

    root = DATA_DIR / "svarah"
    shards = _download_shards(root)

    # Pass 1: metadata only (the audio struct's `path`, not its bytes).
    rows: dict[str, dict] = {}
    for shard in shards:
        t = pq.read_table(shard, columns=["audio_filepath.path", *META_COLS]).to_pylist()
        for r in t:
            path = (
                r["audio_filepath"]["path"]
                if isinstance(r.get("audio_filepath"), dict)
                else r["audio_filepath.path"]
            )
            cid = _clip_id(path)
            if cid in rows:
                raise RuntimeError(f"duplicate Svarah clip id {cid}")
            rows[cid] = {"shard": shard, "ext": PurePosixPath(path).suffix or ".wav", **r}

    ids = list(rows) if full else select_subset(list(rows), n, seed)
    wanted = set(ids)

    # Pass 2: write the selected clips' bytes to disk (once).
    clip_dir = root / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        missing = {
            c
            for c in wanted
            if rows[c]["shard"] == shard and not (clip_dir / f"{c}{rows[c]['ext']}").exists()
        }
        if not missing:
            continue
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256, columns=["audio_filepath"]):
            for a in batch.column("audio_filepath").to_pylist():
                cid = _clip_id(a["path"])
                if cid in missing:
                    (clip_dir / f"{cid}{rows[cid]['ext']}").write_bytes(a["bytes"])

    clips = []
    for cid in ids:
        r = rows[cid]
        rel = f"svarah/clips/{cid}{r['ext']}"
        sr, dur = audio_info(DATA_DIR / rel)
        if sr != 16_000:
            raise RuntimeError(f"{rel} is {sr} Hz; add resampling to the Svarah loader")
        tags = {
            "l1": r["primary_language"] or "unknown",
            "gender": (r["gender"] or "unknown").lower(),
            "age_group": r["age-group"] or "unknown",
            "state": r["native_place_state"] or "unknown",
        }
        clips.append(Clip(cid, name, rel, round(dur, 3), r["text"], name, tags=tags))

    meta = {
        "dataset": name,
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "CC-BY-4.0",
        "shard_sha256": {s: sha256_file(p) for s, p in zip(SHARDS, shards, strict=True)},
        "population": len(rows),
    }
    return meta, clips
