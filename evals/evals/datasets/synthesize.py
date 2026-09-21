"""LLM-powered golden dataset synthesis (online — manual trigger only).

Builds ``retrieval.jsonl`` / ``qa.jsonl`` per corpus and the shared
``out_of_kb.jsonl`` by asking MiniMax-M3 (through the app's own
``LLMProvider`` abstraction — never a bespoke HTTP path) to invent
realistic questions whose answers live inside known source records.

Golden positives are exact by construction: the record that seeded the
question IS the positive, and the synthesizer only samples records the
eval suite will ingest (same ``sample_records`` seed/size), so the
labels always reconcile with the corpus.

Usage (from ``evals/``, conda env ``agents``):

    python -m evals.datasets.synthesize --corpus table_tennis --count 60
    python -m evals.datasets.synthesize --corpus indian_law --count 30
    python -m evals.datasets.synthesize --corpus marvel --count 25
    python -m evals.datasets.synthesize --out-of-kb --count 15

The run is resumable (existing ids are skipped) and every raw model
response is archived under ``datasets/_raw/`` for human audit.
Synthetic data is never trusted blindly — spot-check before promoting
a dataset to baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

from evals.config import DATASETS_DIR, get_eval_settings
from evals.harness.corpus import CORPUS_SOURCES, sample_records

_RECORD_PROMPT = """你是一个 RAG 评测数据集构造器。下面给出来自知识库的一段原文，请产出真实用户会提出的检索问题和参考答案。

要求：
1. question_native：用与原文相同的语言提问，答案必须能从这段原文中直接找到；问题中不得出现"这段文本""上文""原文"等指代词。
2. question_zh：同一问题的中文版本（若原文本身就是中文，请写一个意思相同的中文改写）。
3. answer：基于原文的简短参考答案（1~3 句），只使用原文中的信息，不要添加原文之外的内容。

原文：
{text}

只输出一个 JSON 对象，格式：{{"question_native": "...", "question_zh": "...", "answer": "..."}}，不要输出其他任何内容。"""

_OOB_PROMPT = """你要为一个 RAG 系统构造"域外问题"（知识库中不存在答案的问题），用于测试系统是否会诚实拒答而不是编造。

该系统的知识库只覆盖三个领域：
1. 乒乓球器材与装备知识（胶皮、底板、海绵等器材选购知识；不含运动员 biography、赛事历史、规则历史）
2. 印度劳动法与公司法（仅印度法律；不含中国/美国/欧盟法律）
3. 漫威影视作品信息（仅漫威；不含 DC、其他公司影视）

请生成第 {index} 个域外问题：看起来与上述领域相关、让人以为知识库里可能有答案，但答案明确不在覆盖范围内。
只输出一个 JSON 对象，格式：{{"question": "...", "why_out_of_kb": "用一句话解释为什么知识库覆盖不了"}}，不要输出其他任何内容。问题语言随机在中文/英文之间选择。"""


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort strict-JSON extraction (models love code fences)."""

    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


async def _chat_json(llm, prompt: str, *, debug_tag: str = "") -> Optional[dict]:
    """One call + one repair retry; None means skip this record.

    Unparseable raw responses are archived under ``datasets/_raw/`` so
    a systematic failure (rate limiting, think-strip eat-all, model
    drift) is diagnosable after the fact.
    """

    for attempt in range(2):
        try:
            raw = await llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001 — network/auth issues skip record
            print(f"    llm error: {exc}", file=sys.stderr)
            return None
        data = _extract_json(raw)
        if data is not None:
            return data
        _dump_failed_raw(debug_tag or "unknown", attempt, raw)
        prompt = (
            "你上一次的输出不是合法 JSON。请严格只输出一个 JSON 对象，"
            "不要任何解释、不要 markdown 代码块。\n\n" + prompt
        )
    return None


def _dump_failed_raw(tag: str, attempt: int, raw: str) -> None:
    path = DATASETS_DIR / "_raw" / "failed_responses.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(
                {"tag": tag, "attempt": attempt, "len": len(raw),
                 "head": raw[:200]}, ensure_ascii=False,
            ) + "\n")
    except OSError:
        pass  # diagnostics must never break synthesis


def _load_done_ids(path: Path) -> set[str]:
    """Case ids already in the file, plus their base record ids.

    Retrieval rows are stored as ``{base}-native`` / ``{base}-zh``, but
    the resume check tests the bare base id — collect both so a
    partially-written record never re-synthesizes.
    """

    if not path.is_file():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            case_id = json.loads(line)["id"]
            ids.add(case_id)
            for suffix in ("-native", "-zh"):
                if case_id.endswith(suffix):
                    ids.add(case_id[: -len(suffix)])
    return ids


def _append_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


async def synthesize_corpus(corpus: str, count: int) -> None:
    from app.llm import get_llm_provider

    settings = get_eval_settings()
    llm = get_llm_provider()
    print(f"llm provider={llm.name} model={llm.model}")

    # Same sampling the suite performs, so positives always reconcile.
    eval_records = sample_records(
        CORPUS_SOURCES[corpus](), settings.corpus_size, settings.seed
    )
    rng = random.Random(settings.seed + 1)
    picked_idx = sorted(rng.sample(range(len(eval_records)), min(count, len(eval_records))))

    retrieval_path = DATASETS_DIR / corpus / "retrieval.jsonl"
    qa_path = DATASETS_DIR / corpus / "qa.jsonl"
    raw_path = DATASETS_DIR / "_raw" / f"{corpus}-synth.jsonl"
    done = _load_done_ids(retrieval_path)
    todo = [
        (n, eval_records[idx])
        for n, idx in enumerate(picked_idx, start=1)
        if f"{corpus}-{eval_records[idx]['id']}" not in done
    ]
    print(f"{corpus}: {len(picked_idx)} targets, {len(todo)} to do (resume skips the rest)")

    # MiniMax-M3 spends ~25s per call (reasoning) — parallelize or a
    # 60-record corpus takes an hour. The lock keeps JSONL appends
    # line-atomic on Windows.
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(3)
    made = skipped = 0

    async def one(record: dict) -> None:
        nonlocal made, skipped
        base_id = f"{corpus}-{record['id']}"
        async with semaphore:
            data = await _chat_json(
                llm, _RECORD_PROMPT.format(text=record["text"][:4000]),
                debug_tag=f"{corpus}/{record['id']}",
            )
        if data is None:
            skipped += 1
            print(f"  SKIP (bad json) {record['id']}")
            return
        q_native = str(data.get("question_native", "")).strip()
        q_zh = str(data.get("question_zh", "")).strip()
        answer = str(data.get("answer", "")).strip()
        if not (q_native and q_zh and answer):
            skipped += 1
            return
        async with lock:
            _append_jsonl(
                retrieval_path,
                [
                    {"id": f"{base_id}-native", "query": q_native,
                     "positive_ids": [record["id"]], "notes": "native"},
                    {"id": f"{base_id}-zh", "query": q_zh,
                     "positive_ids": [record["id"]], "notes": "cross-lingual zh"},
                ],
            )
            _append_jsonl(
                qa_path,
                [{"id": base_id, "question": q_native, "ground_truth": answer,
                  "source_ids": [record["id"]], "category": "factual"}],
            )
            _append_jsonl(
                raw_path,
                {"record_id": record["id"], "raw": data,
                 "text_head": record["text"][:120]},
            )
            made += 1
            print(f"  ok {made}/{len(todo)} {record['id']}: {q_zh[:40]}")

    await asyncio.gather(*(one(rec) for _, rec in todo))
    print(f"{corpus}: synthesized={made} skipped={skipped} (resumable)")


async def synthesize_out_of_kb(count: int) -> None:
    from app.llm import get_llm_provider

    llm = get_llm_provider()
    path = DATASETS_DIR / "out_of_kb.jsonl"
    done = _load_done_ids(path)
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(3)
    made = 0

    async def one(i: int) -> None:
        nonlocal made
        oob_id = f"oob-{i:03d}"
        if oob_id in done:
            return
        async with semaphore:
            data = await _chat_json(llm, _OOB_PROMPT.format(index=i), debug_tag=f"oob-{i}")
        if data is None:
            print(f"  [{i}] SKIP (bad json)")
            return
        question = str(data.get("question", "")).strip()
        why = str(data.get("why_out_of_kb", "")).strip()
        if not question:
            return
        async with lock:
            _append_jsonl(path, [{"id": oob_id, "question": question, "why_out_of_kb": why}])
            made += 1
            print(f"  ok {made}/{count} {question[:50]}")

    await asyncio.gather(*(one(i) for i in range(1, count + 1)))
    print(f"out_of_kb: synthesized={made}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize golden eval datasets (online).")
    parser.add_argument("--corpus", choices=sorted(CORPUS_SOURCES))
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--out-of-kb", action="store_true")
    args = parser.parse_args()

    if args.out_of_kb:
        asyncio.run(synthesize_out_of_kb(args.count))
    elif args.corpus:
        asyncio.run(synthesize_corpus(args.corpus, args.count))
    else:
        parser.error("choose --corpus or --out-of-kb")


if __name__ == "__main__":
    main()
