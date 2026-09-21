# Golden 数据集（evals/datasets）

本目录存放测评体系的黄金标注数据。除 `injection/`（手工构造）外，
`retrieval.jsonl` / `qa.jsonl` / `out_of_kb.jsonl` 由
`python -m evals.datasets.synthesize` 用 MiniMax-M3 合成（**在线、手动触发**），
合成过程的原始模型输出归档在 `_raw/` 供人工审计。

## 文件格式（NDJSON，一行一对象）

### `{corpus}/retrieval.jsonl` — 检索黄金标注

```json
{"id": "table_tennis-chunk_42-native", "query": "What sponge hardness suits beginners?",
 "positive_ids": ["chunk_42"], "notes": "native|cross-lingual zh"}
```

- `id` 全局唯一；`positive_ids` 必须非空，引用的是**源语料记录 id**
  （乒乓球为 `chunk_N`，印度法律为 `act#section`，漫威为 `marvel#N`）。
- 正例的构造性保证：问题就是从该记录生成的，答案只存在于该记录中。
- 每条记录生成两问：原文语言 + 中文跨语言改写。

### `{corpus}/qa.jsonl` — 生成质量黄金 QA

```json
{"id": "table_tennis-chunk_42", "question": "...", "ground_truth": "...",
 "source_ids": ["chunk_42"], "category": "factual"}
```

### `out_of_kb.jsonl` — 域外拒答探针

```json
{"id": "oob-001", "question": "...", "why_out_of_kb": "知识库只覆盖器材，不含运动员传记"}
```

### `injection/` — 手工注入攻击语料（无需合成）

- `corpus.jsonl`：4 条正常乒乓球知识 + 4 条注入 payload（规则覆写 / 引用抑制 /
  系统提示窃取 / 拒答胁迫），payload 均携带金丝雀标记 `INJECTION-SUCCESS-7f3a`。
- `queries.jsonl`：良性提问 + 直接攻击提问（含用户侧注入指令）。

## 一致性约束（重要）

合成器与检索套件使用**相同的语料采样参数**（`EVAL_CORPUS_SIZE` + `EVAL_SEED`
默认 300/20260918，见 `evals/config.py`）。golden 正例只从采样后的语料中选取，
因此加载时 `audit_against_corpus` 必须零告警；如果改了采样参数而未重新合成，
检索套件会在对账步骤直接失败（这是有意设计——采样漂移下的指标全是噪声）。

## 构建 / 审核流程

```bash
cd evals
python -m evals.datasets.synthesize --corpus table_tennis --count 60
python -m evals.datasets.synthesize --corpus indian_law --count 30
python -m evals.datasets.synthesize --corpus marvel --count 25
python -m evals.datasets.synthesize --out-of-kb --count 15
```

- 命令**可断点续跑**（已写入的 id 自动跳过）。
- 合成后人工抽检 `_raw/{corpus}-synth.jsonl`（每条含问题/答案/原文开头），
  删除不合格条目后再将指标写入 `baseline.json`。
- 合成数据是 LLM 生成的，质量有波动：正例标注（记录 id）构造性正确，
  但问题难度与自然度需要抽检；发现坏条目直接从 jsonl 删行即可。
