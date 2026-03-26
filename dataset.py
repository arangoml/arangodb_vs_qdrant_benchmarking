"""
BEIR / HuggingFace dataset loading.
"""

from tqdm import tqdm

from config import HF_DATASET, HF_SUBSET


def load_beir_dataset(subset: str = HF_SUBSET) -> tuple[dict, dict, dict]:
    """
    Load corpus, queries, and qrels from Cohere/beir-embed-english-v3.

    Returns:
        corpus:  {doc_id: {"text": str, "title": str, "embedding": list[float]}}
        queries: {query_id: {"text": str, "embedding": list[float]}}
        qrels:   {query_id: {doc_id: relevance_score}}
    """
    from datasets import load_dataset

    print(f"  Loading {HF_DATASET} / {subset} …")

    # --- Corpus ---
    print("    Loading corpus …")
    corpus_ds = load_dataset(HF_DATASET, f"{subset}-corpus", split="train")
    corpus = {}
    for row in tqdm(corpus_ds, desc="    Parsing corpus", leave=False):
        corpus[row["_id"]] = {
            "text": row["text"],
            "title": row.get("title", ""),
            "embedding": row["emb"],
        }
    print(f"    Corpus: {len(corpus):,} documents")

    # --- Queries ---
    print("    Loading queries …")
    queries_ds = load_dataset(
        HF_DATASET, f"{subset}-queries", split="test",
        data_files={"test": f"{subset}/queries/test.parquet"},
    )
    queries = {}
    for row in queries_ds:
        queries[row["_id"]] = {
            "text": row["text"],
            "embedding": row["emb"],
        }
    print(f"    Queries: {len(queries):,}")

    # --- QRels ---
    print("    Loading qrels …")
    qrels_ds = load_dataset(
        HF_DATASET, f"{subset}-qrels", split="test",
        data_files={"test": f"{subset}/qrels/test.parquet"},
    )
    qrels = {}
    for row in qrels_ds:
        qid = str(row["query_id"])
        cid = str(row["corpus_id"])
        score = row["score"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][cid] = score
    print(f"    QRels: {len(qrels):,} queries with judgments")

    return corpus, queries, qrels
