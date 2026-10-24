"""LibriSpeech test-clean / test-other (CC-BY-4.0). 16 kHz FLAC.

Two sources hold the same data:
- "hf" (default): the openslr org's copy on the HF Hub (openslr/librispeech_asr, parquet with the
  original FLAC bytes). Much faster to download than openslr.org from most places.
- "openslr": the original tarballs from openslr.org, checked against the published md5s.

Either way, clips land at librispeech/LibriSpeech/<split>/<speaker>/<chapter>/<utt>.flac (the tarball
layout), so manifests are identical whichever source was used; only `_meta` differs.
"""

from __future__ import annotations

import os
from pathlib import Path

from .common import DATA_DIR, Clip, audio_info, download, extract_tar, md5_file, select_subset, sha256_file

SPLITS = {"librispeech-clean": "test-clean", "librispeech-other": "test-other"}
SOURCE = os.environ.get("VAANI_LIBRISPEECH_SOURCE", "hf")  # "hf" | "openslr"

OPENSLR_URL = "https://www.openslr.org/resources/12"
# Published at https://www.openslr.org/resources/12/md5sum.txt
OPENSLR_MD5 = {
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
}

HF_REPO = "openslr/librispeech_asr"
HF_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"  # pinned 2026-10-24
HF_FILES = {"test-clean": "clean/test/0000.parquet", "test-other": "other/test/0000.parquet"}

ROOT = DATA_DIR / "librispeech"


def _rel(split: str, utt: str) -> str:
    spk, chapter, _ = utt.split("-")
    return f"librispeech/LibriSpeech/{split}/{spk}/{chapter}/{utt}.flac"


def _from_openslr(split: str) -> tuple[dict, dict[str, tuple[str, str]], None]:
    url = f"{OPENSLR_URL}/{split}.tar.gz"
    archive = download(url, ROOT / f"{split}.tar.gz")
    md5 = md5_file(archive)
    if md5 != OPENSLR_MD5[split]:
        raise RuntimeError(f"{archive} md5 {md5} != published {OPENSLR_MD5[split]}; delete it and retry")
    split_dir = ROOT / "LibriSpeech" / split
    if not split_dir.exists():
        extract_tar(archive, ROOT)
    refs: dict[str, tuple[str, str]] = {}
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, _, text = line.partition(" ")
            refs[utt] = (text, utt.split("-")[0])
    meta = {"source": url, "archive_md5": md5, "archive_sha256": sha256_file(archive)}
    return meta, refs, None


def _from_hf(split: str) -> tuple[dict, dict[str, tuple[str, str]], Path]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            HF_REPO, HF_FILES[split], repo_type="dataset", revision=HF_REVISION, local_dir=ROOT / "hub"
        )
    )
    table = pq.read_table(path, columns=["id", "text", "speaker_id"]).to_pylist()
    refs = {r["id"]: (r["text"], str(r["speaker_id"])) for r in table}
    meta = {
        "source": f"https://huggingface.co/datasets/{HF_REPO}",
        "revision": HF_REVISION,
        "file": HF_FILES[split],
        "file_sha256": sha256_file(path),
    }
    return meta, refs, path


def _extract_from_parquet(parquet: Path, split: str, wanted: set[str]) -> None:
    import pyarrow.parquet as pq

    todo = {u for u in wanted if not (DATA_DIR / _rel(split, u)).exists()}
    if not todo:
        return
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=256, columns=["id", "audio"]):
        for utt, audio in zip(batch.column("id").to_pylist(), batch.column("audio").to_pylist(), strict=True):
            if utt in todo:
                out = DATA_DIR / _rel(split, utt)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(audio["bytes"])


def prepare(name: str, n: int, seed: int, full: bool) -> tuple[dict, list[Clip]]:
    split = SPLITS[name]
    if SOURCE == "openslr":
        meta, refs, parquet = _from_openslr(split)
    else:
        meta, refs, parquet = _from_hf(split)

    ids = list(refs) if full else select_subset(list(refs), n, seed)
    if parquet is not None:
        _extract_from_parquet(parquet, split, set(ids))

    clips = []
    for utt in ids:
        rel = _rel(split, utt)
        sr, dur = audio_info(DATA_DIR / rel)
        if sr != 16_000:
            raise RuntimeError(f"{rel} is {sr} Hz, expected 16000")
        text, speaker = refs[utt]
        clips.append(Clip(utt, name, rel, round(dur, 3), text, name, speaker=speaker))

    meta |= {"dataset": name, "split": split, "license": "CC-BY-4.0", "population": len(refs)}
    return meta, clips
