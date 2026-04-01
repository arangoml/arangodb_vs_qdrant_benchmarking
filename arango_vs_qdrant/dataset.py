"""
BEIR / HuggingFace dataset loading.

Downloads parquet shards locally via huggingface_hub (cached after first run),
then queries them with DuckDB for fast filtering.
"""

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from config import HF_DATASET, HF_SUBSET

# BEIR subsets use different split names for qrels/queries
_SPLIT_MAP = {"msmarco": "dev", "nq": "test", "hotpotqa": "test", "fever": "test",
              "quora": "test", "robust04": "test", "fiqa": "test", "scifact": "test"}


def _download(subset: str, filename: str) -> str:
    """Download a parquet file from HF and return the local cached path."""
    return hf_hub_download(
        repo_id=HF_DATASET, filename=f"{subset}/{filename}",
        repo_type="dataset",
    )


def _init_duckdb():
    import duckdb
    con = duckdb.connect()
    return con


def load_beir_dataset(subset: str = HF_SUBSET, target_doc_count: int = 1_000_000) -> tuple[dict, dict, dict]:
    """
    Load corpus, queries, and qrels from Cohere/beir-embed-english-v3.

    Returns:
        corpus:  {doc_id: {"text": str, "title": str, "embedding": list[float]}}
        queries: {query_id: {"text": str, "embedding": list[float]}}
        qrels:   {query_id: {doc_id: relevance_score}}
    """
    split = _SPLIT_MAP.get(subset, "test")
    con = _init_duckdb()

    print(f"  Loading {HF_DATASET} / {subset} (local cache + DuckDB) …")

    # ── 1. Read qrels ─────────────────────────────────────────────────────
    print("    Loading qrels …")
    qrels_path = _download(subset, f"qrels/{split}.parquet")
    qrels_rows = con.execute(f"""
        SELECT query_id, corpus_id, score
        FROM '{qrels_path}'
        WHERE score > 0
    """).fetchall()

    qrels: dict[str, dict[str, int]] = {}
    required_doc_ids: set[str] = set()
    for query_id, corpus_id, score in qrels_rows:
        qid, cid = str(query_id), str(corpus_id)
        qrels.setdefault(qid, {})[cid] = score
        required_doc_ids.add(cid)

    print(f"    QRels: {len(qrels):,} queries, {len(required_doc_ids):,} required docs")

    # ── 2. Fetch required docs + padding from shards ───────────────────────
    print("    Fetching corpus docs …")
    corpus: dict[str, dict] = {}
    remaining_ids = list(required_doc_ids)
    shard_idx = 0

    while remaining_ids or len(corpus) < target_doc_count:
        try:
            shard_path = _download(subset, f"corpus/{shard_idx:04d}.parquet")
        except EntryNotFoundError:
            break

        print(f"      Shard {shard_idx} …")

        # Fetch required docs from this shard
        if remaining_ids:
            rows = con.execute(f"""
                SELECT _id, title, text, emb
                FROM '{shard_path}'
                WHERE _id IN (SELECT unnest($1::VARCHAR[]))
            """, [remaining_ids]).fetchall()
            for _id, title, text, emb in rows:
                corpus[_id] = {"text": text, "title": title or "", "embedding": emb}
            found = len(rows)
            remaining_ids = [d for d in remaining_ids if d not in corpus]
        else:
            found = 0

        # Fetch padding docs from this shard if needed
        still_needed = target_doc_count - len(corpus)
        if still_needed > 0:
            already = list(corpus.keys()) or ["__none__"]
            padding_rows = con.execute(f"""
                SELECT _id, title, text, emb
                FROM '{shard_path}'
                WHERE _id NOT IN (SELECT unnest($1::VARCHAR[]))
                LIMIT {still_needed}
            """, [already]).fetchall()
            for _id, title, text, emb in padding_rows:
                corpus[_id] = {"text": text, "title": title or "", "embedding": emb}
            pad = len(padding_rows)
        else:
            pad = 0

        print(f"        {found} required, {pad} padding")
        shard_idx += 1

    print(f"    Corpus total: {len(corpus):,} documents")

    # ── 3. Filter qrels to usable queries ─────────────────────────────────
    corpus_ids = set(corpus.keys())
    usable_qrels: dict[str, dict[str, int]] = {}
    for qid, rels in qrels.items():
        if all(cid in corpus_ids for cid in rels):
            usable_qrels[qid] = rels

    print(f"    Usable queries: {len(usable_qrels):,}/{len(qrels):,}")

    # ── 4. Read queries ───────────────────────────────────────────────────
    usable_qids = list(usable_qrels.keys())
    print(f"    Loading {len(usable_qids):,} queries …")
    queries_path = _download(subset, f"queries/{split}.parquet")
    query_rows = con.execute(f"""
        SELECT _id, text, emb
        FROM '{queries_path}'
        WHERE _id IN (SELECT unnest($1::VARCHAR[]))
    """, [usable_qids]).fetchall()

    queries: dict[str, dict] = {}
    for _id, text, emb in query_rows:
        queries[_id] = {"text": text, "embedding": emb}

    print(f"    Queries: {len(queries):,}")

    con.close()
    return corpus, queries, usable_qrels
