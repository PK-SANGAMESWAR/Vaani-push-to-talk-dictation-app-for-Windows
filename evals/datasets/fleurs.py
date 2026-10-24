"""FLEURS en_us test split (google/fleurs, CC-BY-4.0). 16 kHz WAV inside a tar.gz on the HF Hub.

Reference = `raw_transcription` (cased, punctuated); the normalizer handles case and punctuation.
FLEURS reads each sentence with several speakers, so the clip id is the WAV file name, not the
sentence id. FLEURS has no speaker ids; gender is kept as a tag.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .common import DATA_DIR, Clip, audio_info, extract_tar, select_subset, sha256_file

REPO = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"  # pinned 2026-10-24
NAME = "fleurs-en"


def prepare(name: str, n: int, seed: int, full: bool) -> tuple[dict, list[Clip]]:
    from huggingface_hub import hf_hub_download

    root = DATA_DIR / "fleurs"
    kw = {"repo_id": REPO, "repo_type": "dataset", "revision": REVISION, "local_dir": root / "hub"}
    tsv = Path(hf_hub_download(filename="data/en_us/test.tsv", **kw))
    archive = Path(hf_hub_download(filename="data/en_us/audio/test.tar.gz", **kw))
    audio_dir = root / "en_us"
    if not (audio_dir / "test").exists():
        extract_tar(archive, audio_dir)

    rows: dict[str, dict] = {}
    with tsv.open(encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sent_id, file_name, raw, _norm, _phon, _num_samples, gender = r[:7]
            rows[file_name.removesuffix(".wav")] = {"sentence": sent_id, "ref": raw, "gender": gender}

    ids = list(rows) if full else select_subset(list(rows), n, seed)
    clips = []
    for cid in ids:
        rel = f"fleurs/en_us/test/{cid}.wav"
        sr, dur = audio_info(DATA_DIR / rel)
        assert sr == 16_000, rel
        r = rows[cid]
        tags = {"gender": r["gender"].lower(), "sentence": r["sentence"]}
        clips.append(Clip(cid, name, rel, round(dur, 3), r["ref"], name, tags=tags))

    meta = {
        "dataset": name,
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "CC-BY-4.0",
        "archive_sha256": sha256_file(archive),
        "tsv_sha256": sha256_file(tsv),
        "population": len(rows),
    }
    return meta, clips
