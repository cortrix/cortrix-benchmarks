from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, Set, Tuple


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    split: str = "test"
    local_path: str | None = None


DATASETS: Dict[str, DatasetSpec] = {
    "scifact": DatasetSpec(
        "scifact",
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    ),
    "nfcorpus": DatasetSpec(
        "nfcorpus",
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    ),
    "fiqa": DatasetSpec(
        "fiqa",
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    ),
    "webis-touche2020": DatasetSpec(
        "webis-touche2020",
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
    ),
    "hotpotqa": DatasetSpec(
        "hotpotqa",
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
    ),
    "quora": DatasetSpec(
        "quora",
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    ),
}
# The registry lists datasets this repository ships or can download. Two
# local-fixture entries, fiqa-mini-120 and fiqa-mini-600, were removed: their
# fixture directories are not part of this repository, so selecting them raised
# FileNotFoundError, and once --dataset became a constrained choice they were
# advertised in --help as though they were available. A local-fixture extension
# mechanism can be designed separately; keeping unobtainable names here would
# make the advertised surface depend on whichever private checkout is in use.
#
# DatasetSpec.local_path and ensure_dataset's local branch are deliberately
# retained for that future mechanism.
#
# _strip_known_dataset_prefix in run_benchmark and cortrix_client still lists the
# removed names. That is not an oversight: it recovers a document id from a
# prefixed filename, so dropping the names there would silently break doc-id
# extraction for any namespace already ingested from those fixtures.


def download_dataset(spec: DatasetSpec, cache_dir: Path, log: LogFn = print) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{spec.name}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        log(f"[download] reuse {zip_path}")
        return zip_path

    tmp_path = zip_path.with_suffix(".zip.part")
    log(f"[download] {spec.url} -> {zip_path}")
    with urllib.request.urlopen(spec.url, timeout=120) as response:
        with tmp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    tmp_path.replace(zip_path)
    return zip_path


def extract_dataset(zip_path: Path, cache_dir: Path, log: LogFn = print) -> Path:
    extract_root = cache_dir / zip_path.stem
    if _has_dataset_files(extract_root):
        log(f"[extract] reuse {extract_root}")
        return find_dataset_root(extract_root)

    log(f"[extract] {zip_path} -> {extract_root}")
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)
    return find_dataset_root(extract_root)


def ensure_dataset(spec: DatasetSpec, cache_dir: Path, log: LogFn = print) -> Path:
    if spec.local_path:
        root = _resolve_local_path(spec.local_path)
        log(f"[dataset] use local fixture {spec.name}: {root}")
        return find_dataset_root(root)
    zip_path = download_dataset(spec, cache_dir, log=log)
    return extract_dataset(zip_path, cache_dir, log=log)


def _resolve_local_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def find_dataset_root(root: Path) -> Path:
    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    for candidate in candidates:
        if _has_dataset_files(candidate):
            return candidate
    raise FileNotFoundError(f"BEIR dataset root not found under {root}")


def _has_dataset_files(path: Path) -> bool:
    return (path / "corpus.jsonl").exists() and (path / "queries.jsonl").exists()


def qrels_path(root: Path, split: str = "test") -> Path:
    path = root / "qrels" / f"{split}.tsv"
    if path.exists():
        return path
    root_qrels = root / "qrels.tsv"
    if root_qrels.exists():
        return root_qrels
    qrels_dir = root / "qrels"
    candidates = sorted(qrels_dir.glob("*.tsv")) if qrels_dir.exists() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"qrels TSV not found under {root}")


def iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield item


def iter_corpus(root: Path) -> Iterator[Tuple[str, str, str]]:
    for item in iter_jsonl(root / "corpus.jsonl"):
        doc_id = str(item.get("_id", ""))
        title = str(item.get("title", "") or "")
        text = str(item.get("text", "") or "")
        if doc_id:
            yield doc_id, title, text


def iter_queries(root: Path) -> Iterator[Tuple[str, str]]:
    for item in iter_jsonl(root / "queries.jsonl"):
        query_id = str(item.get("_id", ""))
        text = str(item.get("text", "") or "")
        if query_id:
            yield query_id, text


def count_corpus_rows(root: Path) -> int:
    return sum(1 for _ in iter_corpus(root))


def load_qrels(root: Path, split: str = "test") -> Dict[str, Dict[str, float]]:
    path = qrels_path(root, split=split)
    qrels: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        first = True
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if first:
                first = False
                if parts[:3] == ["query-id", "corpus-id", "score"]:
                    continue
            if len(parts) < 3:
                continue
            query_id, doc_id, score_text = parts[0], parts[1], parts[2]
            try:
                score = float(score_text)
            except ValueError:
                continue
            if score <= 0:
                continue
            qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def load_queries(root: Path, query_ids: Set[str]) -> Dict[str, str]:
    wanted = set(query_ids)
    queries: Dict[str, str] = {}
    for query_id, text in iter_queries(root):
        if query_id in wanted:
            queries[query_id] = text
            if len(queries) == len(wanted):
                break
    return queries


def file_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total
