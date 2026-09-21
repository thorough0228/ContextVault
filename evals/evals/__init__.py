"""ContextVault RAG quality evaluation harness.

Four layered suites (see ``evals/README.md`` for the full handbook):

* **Retrieval** — deterministic Recall@k / HitRate@k / MRR@k / NDCG@k
  against golden relevance labels. Offline (local bge embeddings),
  reproducible, regression-gated against ``baseline.json``.
* **Generation** — DeepEval LLM-as-judge metrics (faithfulness, answer
  relevancy, citation accuracy, refusal) over real chat turns. Online
  (MiniMax via the app's own ``LLMProvider`` abstraction).
* **Streaming contract** — programmatic NDJSON event-order / think-filter
  / error-path assertions. Offline, no LLM.
* **Robustness** — prompt-injection corpus + out-of-KB refusal.
  Mechanics offline; semantic judgement online.

All AI calls (judge included) go through the API app's existing
``LLMProvider`` / ``EmbeddingProvider`` abstractions — the harness adds
no new provider pathway.
"""

from __future__ import annotations

import sys
from pathlib import Path

# pytest picks apps/api up via pyproject pythonpath; CLI entry points
# (python -m evals.datasets.synthesize, ...) do not — shim it here so
# every consumer of this package can import the API app unchanged.
_API_DIR = Path(__file__).resolve().parents[2] / "apps" / "api"
if _API_DIR.is_dir() and str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))
