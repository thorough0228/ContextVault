# ContextVault 测评体系（evals/）

面向 RAG 平台的四层质量测评体系。**检索指标自建**（数学精确计算，
无需 LLM 裁判）；**生成质量用 DeepEval**（业界流行的 pytest 原生 LLM 评测框架），
judge 与被测系统全部通过项目现有的 `LLMProvider` / `EmbeddingProvider`
抽象调用（AGENTS.md 规则 8）。本目录零侵入：不改业务代码、不动数据库、不加 API。

## 四层套件

| # | 套件 | 文件 | 指标 | 依赖 |
|---|------|------|------|------|
| ① | 检索质量 | `suites/test_retrieval.py` | Recall@k / HitRate@k / MRR@k / NDCG@k（k=1,3,5,10） | 离线（本地 bge）+ golden 数据集 |
| ② | 生成质量 | `suites/test_generation.py` | faithfulness / answer_relevancy / citation_accuracy / refusal_correctness | **在线**（MiniMax 聊天 + DeepEval judge） |
| ③ | 流式契约 | `suites/test_streaming_contract.py` | NDJSON 事件序 / `<think>` 过滤边界 / 错误路径不落库 / 重试次数 | 离线 |
| ④ | 健壮性 | `suites/test_robustness.py` | 注入抵抗（金丝雀标记 + G-Eval）/ 攻击下检索不脱落 | 离线机制 + **在线**语义 |
| ⑤ | **ACL 泄露** | `suites/test_acl.py` | 跨租户检索/更新/回滚/Chat/记忆/BM25 腿/Bluegreen 链隔离矩阵 | 离线（双租户确定性） |

另含 harness 自身的回归测试：指标数学手算单测（`test_metrics_unit.py`）、
对齐器单测（`test_alignment_unit.py`）、语料/加载器单测
（`test_corpus_loader_unit.py`）、环境冒烟（`test_env_smoke.py`）、
检索管线机制回归（`test_retrieval.py::test_retrieval_pipeline_mechanics`）。

## 快速开始

```bash
# 必须用 conda agents 环境（PATH 上的 python 是 base Anaconda，依赖不对）
cd evals

# 离线全量（CI 安全，零网络）
python -m pytest evals/suites -m offline

# 在线生成质量（花 MiniMax token；默认每语料采样 10 题）
python -m pytest evals/suites -m online

# live profile：打真实 docker 栈（pgvector 生产路径；需先起栈 + API + worker）
EVAL_PROFILE=live python -m pytest evals/suites/test_retrieval.py -m offline
```

首次运行 ② 之前需要合成 golden 数据集并安装 deepeval（见下）。

## 环境与 profile

- **sqlite（默认）**：进程内 ASGI 应用 + 临时 SQLite + 内存存储 + eager worker，
      embedding 用仓库内 `models/bge-m3`（离线真实语义，1024 维多语模型）。
      完全确定性、可进 CI。已知边界：SQLite 检索路径有 `top_k*50` 候选截断，
      因此语料规模有守卫（`EVAL_SQLITE_CHUNK_GUARD=950`，配套 `EVAL_RETRIEVAL_TOP_K=20`
      使候选帽 1000 覆盖最大语料）。
- **live**：纯 HTTP 打运行中的 docker 栈，走 pgvector HNSW 生产路径，
      独立测评账号（默认 `contextvault-eval@local.test`）+ 独立 RAG，
      结束自动删除（`EVAL_LIVE_KEEP_RAGS=1` 保留供排查），不碰用户数据。

## 关键环境变量（evals/config.py）

| 变量 | 默认 | 说明 |
|------|------|------|
| `EVAL_PROFILE` | `sqlite` | `sqlite` / `live` |
| `EVAL_K_GRID` | `1,3,5,10` | 检索指标 k 网格 |
| `EVAL_RETRIEVAL_TOP_K` | `20` | 检索请求 top_k（一次请求服务全网格；需 ≥ 最大语料 chunk 数 / 50，见下） |
| `EVAL_CHUNK_SIZE` | `500` | 测评入库切块窗口（与 .env 的 CHUNK_SIZE_CHARS 保持一致） |
| `EVAL_CHUNK_OVERLAP` | `100` | 切块重叠（与 .env 的 CHUNK_OVERLAP_CHARS 保持一致） |
| `EVAL_EMBEDDING_MODEL` | `models/bge-m3` | 本地 embedding 模型路径 |
| `EVAL_EMBEDDING_DIMENSION` | `1024` | 模型维度（与 EMBEDDING_DIMENSION 保持一致） |
| `EVAL_QUERY_INSTRUCTION` | 空 | 查询前缀（bge-m3 不需要；bge-small-zh 用中文检索指令） |
| `EVAL_CORPUS_SIZE` | `300` | 每语料采样记录数（sqlite） |
| `EVAL_LIVE_CORPUS_SIZE` | `800` | live profile 语料记录数 |
| `EVAL_QA_SAMPLE` | `10` | 生成套件每语料采样题数（控 token） |
| `EVAL_TOLERANCE` | `0.03` | 回归门禁容差（低于基线-容差即 fail） |
| `EVAL_SEED` | `20260918` | 语料采样种子（改它必须重新合成数据集！） |
| `EVAL_SQLITE_CHUNK_GUARD` | `950` | 单语料 chunk 数守卫（SQLite 候选帽保护） |
| `EVAL_API_URL` | `http://localhost:8000` | live profile API 地址 |
| `EVAL_LIVE_EMAIL/PASSWORD` | 见 config.py | live 测评账号 |

## 已知：免费 embedding API 的间歇性限流

硅基流动免费档在**密集连续调用**（检索套件一次运行 ≈ 900 次入库向量 + 230 次查询）
时偶发整体劣化（表现为某语料指标瞬间崩到 0，数十分钟后自愈；服务端不返回标准 429）。
对策：套件偶发失败时**间隔几分钟重跑**即可；生产在线检索是单查询低频调用，不受影响。
需要完全稳定可切换付费档或充值智谱（改 .env 的 EMBEDDING_OPENAI_* 即可）。

## 混合检索 A/B（Phase 12，已完成 2026-09-19）

生产 `SEARCH_MODE=hybrid`（向量 + jieba 关键词 RRF 融合 + bge-reranker 精排），
完整 A/B 结果已定稿为检索基线（vector 模式基线 → hybrid）：

| 语料 | vector hit@5 | hybrid hit@5 |
|---|---|---|
| table_tennis | 0.858 | **0.942** |
| indian_law | 0.933 | **0.983** |
| marvel | 0.920 | **1.000** |

复跑命令：`EVAL_SEARCH_MODE=hybrid python -m pytest evals/suites/test_retrieval.py::test_retrieval_metrics -m offline -s`

## 指标解读注意事项

- `cite_eligible_turns` / `context_hit_rate` 是**样本量敏感**指标：`EVAL_QA_SAMPLE=5`
  的小采样运行时单语料波动大（±2 题即 ±0.4），回归判断以 `overall` 与
  faithfulness/relevancy 为准；正式基线对比建议用默认 10 题/语料。
- 健壮性套件门禁是程序化的（canary=0、系统提示零泄露、引用保留率），
  `reference_injection_resistance` 仅为 judge 参考分。

judge 模型 = `.env` 中的 LLM 配置（当前 MiniMax-M3）。**换更强 judge 只改 .env，
零代码改动**。已知局限（有实证）：M3 自评存在自偏好且**打分会塌缩**（完美回答也得
0.1），因此健壮性套件以**程序化指标为主门禁**（金丝雀泄漏数=0、系统提示零泄露、
引用保留率），judge 分数仅作参考（`reference_injection_resistance`）。

## Golden 数据集

见 `datasets/README.md`。构建（在线、手动、可断点续跑）：

```bash
python -m evals.datasets.synthesize --corpus table_tennis --count 60
python -m evals.datasets.synthesize --corpus indian_law --count 30
python -m evals.datasets.synthesize --corpus marvel --count 25
python -m evals.datasets.synthesize --out-of-kb --count 15
```

## 报告与基线

- 每次运行写 `results/eval-{时间戳}-{套件}-{profile}.json` + 同名 `.md`。
- `baseline.json` 存人工确认的基线指标；套件结束自动对比，
  低于 `基线 - EVAL_TOLERANCE` 直接 fail（可挂 CI）。
- 无基线时套件用宽松的绝对下限（`report.py: FLOORS`）防灾难性回归。
- 把一次运行提升为基线：

  ```python
  from evals.harness.report import update_baseline
  update_baseline("retrieval", scores)   # scores 取自 results/ 最新 json
  ```

## 对齐原理（为什么检索指标是精确的）

eval 语料由 harness 序列化为 `.jsonl`（一行一记录，`text` 字段），
经**真实**解析→切块→嵌入管线入库。切块器在 metadata 里保存
`char_start/char_end`（相对虚拟页文本的偏移），而页文本 = 200 条 strip 后
记录以 `\n\n` 连接——harness 在 `alignment.py` 逐字符复现该序列化，
因此检索命中能确定性地映射回源记录 id，没有任何模糊匹配。
>10% 命中无法对齐时套件直接失败（说明解析/切块行为漂移）。

## 修改守则

- 本目录只读 `apps/api` 的公开模块（provider 工厂、ingest 常量、模型），
  不修改任何业务代码；发现被测系统问题 → 记录到报告，走正常开发流程修。
- 新增语料：`harness/corpus.py` 加 loader → 合成数据集 → 套件的
  `CORPORA` 列表加名字。
- 新增指标：`metrics/retrieval.py` 纯函数 + 手算单测，报告自动带上。
