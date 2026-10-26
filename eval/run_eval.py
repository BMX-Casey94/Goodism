"""Evaluation harness: measures retrieval recall@k and phrase-hit rate.

Run after every ingest or model change:

    python eval/run_eval.py [--k 5] [--out eval/results/<stamp>.json]

Metrics per question and aggregated:
  - recall@k: did any expected doc appear in the top k hits?
  - mrr: reciprocal rank of the first expected doc.
  - phrase hit rate: fraction of key phrases present in the retrieved text.

The run writes a JSON report into eval/results/ so accuracy can be tracked
over time and compared across embedding models and rerank settings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from tools.kb import config  # noqa: E402
from tools.kb.indexer import index_status, search  # noqa: E402


def load_questions() -> list[dict]:
    path = config.EVAL_DIR / "questions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


def evaluate(k: int = 5) -> dict:
    questions = load_questions()
    results = []
    recall_hits = 0
    mrr_sum = 0.0
    phrase_sum = 0.0
    phrase_total = 0

    for case in questions:
        query = case["question"]
        expected = set(case.get("expected_docs", []))
        phrases = case.get("key_phrases", [])
        hits = search(query, top_k=k)

        ranks = {hit.doc_id: i + 1 for i, hit in enumerate(hits)}
        hit_ranks = [ranks[d] for d in expected if d in ranks]
        recalled = bool(hit_ranks)
        recall_hits += int(recalled)
        mrr_sum += (1.0 / min(hit_ranks)) if hit_ranks else 0.0

        retrieved_text = " ".join(h.text.lower() for h in hits)
        phrase_hits = sum(1 for p in phrases if p.lower() in retrieved_text)
        if phrases:
            phrase_sum += phrase_hits / len(phrases)
            phrase_total += 1

        results.append(
            {
                "id": case["id"],
                "question": query,
                "recalled": recalled,
                "first_rank": min(hit_ranks) if hit_ranks else None,
                "phrase_hits": phrase_hits,
                "phrase_total": len(phrases),
            }
        )

    n = len(questions)
    report = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "k": k,
        "index": index_status(),
        "questions": n,
        "recall_at_k": round(recall_hits / n, 4) if n else None,
        "mrr": round(mrr_sum / n, 4) if n else None,
        "phrase_hit_rate": round(phrase_sum / phrase_total, 4) if phrase_total else None,
        "cases": results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config.load_dotenv_if_present()
    report = evaluate(k=args.k)

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out or (config.EVAL_RESULTS_DIR / f"eval-{stamp}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print(f"\nreport written to {out}")

    # Non-zero exit if recall falls below 1.0 so CI can gate on it.
    return 0 if report["recall_at_k"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
