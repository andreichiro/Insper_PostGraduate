from __future__ import annotations

import csv
import heapq
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Generic,
    Optional,
)

# Type variables / Protocols

T = TypeVar("T")  # Input record type (e.g., CSV row)
K = TypeVar("K")  # Map/Reduce key type
V = TypeVar("V")  # Map output value type
R = TypeVar("R")  # Reduce output value type

Mapper = Callable[[T], Iterable[Tuple[K, V]]]
Reducer = Callable[[K, Iterable[V]], Iterable[Tuple[K, R]]]


# CSV utils w/ lazy
def iter_csv_dicts(
    path: str,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
    quotechar: str = '"'
) -> Iterator[Dict[str, str]]:
    """Yield CSV rows as dictionaries (header required)."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            # normalize header keys (strip spaces / quotes)
            yield {k.strip(): (v.strip() if v is not None else v) for k, v in row.items()}


# MapReduce 'engine'
@dataclass(frozen=True)
class MapReduce(Generic[T, K, V, R]):
    """
    A sequential MapReduce pipeline:
    - map
    - shuffle
    - sort (by key)
    - reduce
    """
    mapper: Mapper[T, K, V]
    reducer: Reducer[K, V, R]
    sort_keys: bool = True
    key_sort_key: Optional[Callable[[K], Any]] = None  # how keys are ordered

    def execute(self, data: Iterable[T]) -> List[Tuple[K, R]]:
        # Map 
        intermediate: List[Tuple[K, V]] = []
        for record in data:
            for kv in self.mapper(record):
                intermediate.append(kv)

        # Shuffle (group by key) 
        grouped: DefaultDict[K, List[V]] = defaultdict(list)
        for k, v in intermediate:
            grouped[k].append(v)

        # Sort (by key) 
        items: Iterable[Tuple[K, List[V]]]
        if self.sort_keys:
            key_fn = self.key_sort_key or (lambda x: x)
            items = sorted(grouped.items(), key=lambda kv: key_fn(kv[0]))
        else:
            items = grouped.items()

        #  Reduce 
        out: List[Tuple[K, R]] = []
        for k, values in items:
            for out_k, out_v in self.reducer(k, values):
                out.append((out_k, out_v))
        return out


# ---------------------------
# Helpers: takeOrdered (top-N)
# ---------------------------

def take_ordered(
    items: Iterable[Tuple[K, R]],
    n: int,
    *,
    key: Callable[[Tuple[K, R]], Any],
    reverse: bool = False
) -> List[Tuple[K, R]]:
    """
    Return N items ordered by the provided key.
    reverse=False -> smallest N; reverse=True -> largest N
    """
    if reverse:
        return heapq.nlargest(n, items, key=key)
    return heapq.nsmallest(n, items, key=key)


# ============================================================
# JOB 1: Citation counts from CSV (top-10 most cited patents)
# ============================================================

def citation_mapper(row: Dict[str, str]) -> Iterable[Tuple[int, int]]:
    """
    Expecting CSV headers: CITING, CITED.
    Emits (CITED_id, 1)
    """
    cited_raw = row.get("CITED")
    if not cited_raw:
        return []
    try:
        cited = int(cited_raw)
    except ValueError:
        return []
    return [(cited, 1)]


def sum_reducer(key: int, values: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """Emit a single aggregated (key, sum(values))."""
    total = 0
    for v in values:
        total += v
    yield (key, total)


# ============================================================
# JOB 2: Inverted index (word -> sorted unique doc IDs)
# ============================================================

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> Iterator[str]:
    for m in _WORD_RE.finditer(text.lower()):
        yield m.group(0)

def inverted_index_mapper(doc: Tuple[str, str]) -> Iterable[Tuple[str, str]]:
    """
    Input: (doc_id, text)
    Emits (word, doc_id) for each token.
    """
    doc_id, text = doc
    for w in tokenize(text):
        yield (w, doc_id)

def inverted_index_reducer(word: str, doc_ids: Iterable[str]) -> Iterable[Tuple[str, List[str]]]:
    """
    Deduplicate and sort doc IDs per word.
    (Augment here to keep positions if needed.)
    """
    uniq_sorted = sorted(set(doc_ids))
    yield (word, uniq_sorted)


# ============================================================
# JOB 3: Distributed sort (map emits key,record; reduce passes through)
# ============================================================

def distributed_sort_mapper(
    row: Dict[str, str],
    key_extractor: Callable[[Dict[str, str]], Any]
) -> Iterable[Tuple[Any, Dict[str, str]]]:
    """
    Extract (key, row) for sorting by key.
    """
    return [(key_extractor(row), row)]

def distributed_sort_reducer(
    key: Any,
    rows: Iterable[Dict[str, str]]
) -> Iterable[Tuple[Any, Dict[str, str]]]:
    """
    Emit all pairs unchanged (grouped by sorted key).
    """
    for r in rows:
        yield (key, r)


# ============================================================
# Example usage (as functions, not __main__)
# ============================================================

def run_citation_counts(csv_path: str, top_n: int = 10) -> List[Tuple[int, int]]:
    """
    Load citations CSV and return top-N most cited patents.
    Assumes header with 'CITING','CITED'.
    """
    data = iter_csv_dicts(csv_path)
    job = MapReduce(mapper=citation_mapper, reducer=sum_reducer, sort_keys=True)
    counts = job.execute(data)

    # top-N by count desc (takeOrdered with reverse=True)
    return take_ordered(counts, top_n, key=lambda kv: kv[1], reverse=True)


def run_inverted_index(docs: Iterable[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
    """
    Build inverted index: word -> sorted list of doc IDs.
    """
    job = MapReduce(mapper=inverted_index_mapper, reducer=inverted_index_reducer, sort_keys=True)
    return job.execute(docs)


def run_distributed_sort(
    csv_path: str,
    *,
    key_extractor: Callable[[Dict[str, str]], Any]
) -> List[Tuple[Any, Dict[str, str]]]:
    """
    Simulate distributed sort:
    - Map: emit (key, row)
    - Shuffle+Sort: global sort by key
    - Reduce: emit unchanged
    """
    # Partial application of the mapper to freeze the key extractor:
    def _mapper(row: Dict[str, str]) -> Iterable[Tuple[Any, Dict[str, str]]]:
        return distributed_sort_mapper(row, key_extractor)

    job = MapReduce(mapper=_mapper, reducer=distributed_sort_reducer, sort_keys=True)
    return job.execute(iter_csv_dicts(csv_path))
