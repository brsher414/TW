# TRENDING_WORDS 结构调整方案（仅方案，不修改业务代码）

## 1. 文档目的

本文只描述下一轮准备进行的结构性修改，不代表这些修改已经实施。

本轮目标是先解决以下问题：

1. Manifest 在频繁测试时容易产生混乱状态；
2. 模型已经接收全部 Taxonomy 属性和标签，Taxonomy embedding 排序价值有限；
3. Dashboard 信息过多，职责和 AI Insights 页面有重复；
4. 多个入口脚本重复实现上下文、状态、API 调用和文件处理；
5. 缺少统一的 pipeline 入口；
6. Taxonomy 抽取后必须经过人工核验，不能被自动流程跳过；
7. 仓库根目录 `src/`、`TRENDING_WORDS/src/` 和 `TRENDING_WORDS/core/` 并存，导入方式不统一。

本轮明确不处理趋势算法、业务阈值、聚类参数、Prompt 业务规则及新增业务功能。

---

## 2. 计划中的核心调整

### 2.1 删除 Pipeline Manifest

#### 当前问题

- 每个脚本都会调用 `create_manifest`、`update_stage`；
- 测试时反复运行单个阶段，Manifest 可能显示某阶段失败或运行中，但实际产物来自另一次测试；
- Manifest 同时承担运行状态、产物索引和 active run 指针，职责混合；
- Streamlit 多 Session 或多个进程写同一个 JSON 时还可能互相覆盖。

#### 计划修改

- 删除 `core/run_manifest.py`；
- 删除各入口脚本中的 `create_manifest`、`update_stage` 调用；
- Pipeline 是否成功直接以子进程退出码和目标文件是否存在为准；
- 不再持久化 `running`、`failed`、`completed` 等中间阶段状态；
- 保留一个非常轻量的 active-run pointer，仅供 Streamlit Workspace 选择最近可用 Run。

建议 active-run 文件只包含：

```json
{
  "category_code": "YD",
  "active_run_id": "2025Q1_vs_2026Q1_...",
  "updated_at_utc": "2026-08-26T10:00:00+00:00"
}
```

它不记录阶段状态，也不承担审计功能。

#### 验收标准

- 仓库中不再有 `create_manifest`、`update_stage` 调用；
- 重复执行任意阶段不会生成或修改 `manifest.json`；
- Dashboard 仍能定位用户当前选择的品类和 Run；
- 某阶段失败时返回非零退出码，并保留清晰日志。

---

### 2.2 删除 Taxonomy Embedding 检索链路

#### 当前问题

现有流程会：

1. 将 Taxonomy 属性和标签转换为 embedding；
2. 将 Cluster/Noise Term 转换为 embedding；
3. 按余弦相似度排序 Taxonomy 属性；
4. 最终仍把全部属性和全部标签提供给模型。

如果模型最终看到完整 Taxonomy，embedding 只改变排列顺序，不减少 token，也不限制候选集合，因此需要额外的模型加载、缓存、向量文件和维护成本，但对最终证据范围没有实质影响。

#### 计划修改

- `build_taxonomy_reference.py` 只负责：
  - 读取人工核验后的 Excel；
  - 统一字段名和文本格式；
  - 过滤明确无效标签；
  - 输出标准化 Taxonomy Parquet 和清洗审计；
- 不再生成：
  - `taxonomy_attribute_embeddings.npy`；
  - `taxonomy_embedding_index.parquet`；
- `retrieve_taxonomy_candidates.py` 不再加载 embedding 模型；
- 所有有效属性按稳定顺序提供，例如按 `attribute_code`、`attribute_name` 排序；
- 每个属性仍携带完整有效标签；
- Evidence 中移除或废弃 `similarity` 字段，避免 UI 或 Prompt 将无意义的空分数解释为业务置信度。

#### 保留的 embedding

热词聚类仍然需要 BGE-M3 embedding。此次只删除 **Taxonomy 属性排序 embedding**，不修改：

- 热词 embedding；
- UMAP；
- HDBSCAN；
- Cluster embedding cache。

#### 验收标准

- Taxonomy 阶段不加载 `FlagEmbedding` 或 `SentenceTransformer`；
- 不再生成 Taxonomy `.npy` 向量文件；
- 每条 LLM Evidence 仍包含全部有效属性及其全部有效标签；
- 相同 Taxonomy 输入产生稳定、确定的属性顺序；
- Prompt/schema/dashboard 不再依赖 Taxonomy similarity。

---

### 2.3 增加统一 Pipeline，但保留强制人工核验点

#### 是否需要 `pipeline.py`

建议增加，但它只负责编排，不承载业务计算。

`pipeline.py` 不应该重新实现 ETL、趋势计算、聚类或 Evidence 构建逻辑，而应调用已有的 stage service/entrypoint。这样既提供统一入口，也避免形成另一个超大文件。

#### 建议拆成两个不能自动串联的阶段

##### 阶段 A：`prepare`

```text
ETL
  ↓
趋势热词计算
  ↓
Embedding + Cluster
  ↓
Taxonomy 原始数据抽取与自动清洗
  ↓
停止：等待人工核验
```

建议命令：

```bash
python pipeline.py prepare --category YD
```

输出应明确告诉用户需要核验哪个 Excel，以及后续命令是什么。

##### 阶段 B：`after-review`

```text
显式传入人工核验后的 Excel
  ↓
标准化完整 Taxonomy
  ↓
为 Cluster / Noise Term 构建完整 Taxonomy Evidence
  ↓
AI Insights / Dashboard Cache
```

建议命令：

```bash
python pipeline.py after-review \
  --category YD \
  --run-id <prepare 阶段的 run-id> \
  --reviewed-taxonomy /path/to/reviewed.xlsx
```

#### 人工核验保护规则

- `after-review` 必须显式提供 `--reviewed-taxonomy`；
- 不允许自动把刚抽取的原始或 cleaned 文件当作已审核文件；
- `prepare` 不接受 `--reviewed-taxonomy`，防止两个阶段通过一次调用串联；
- 审核文件不存在、格式错误或缺少必要列时立即退出；
- 后续可考虑增加审核人、审核时间和文件 hash，但本轮不实现审批系统。

#### Pipeline 的职责边界

Pipeline 只负责：

- 解析参数；
- 创建 `ProjectContext`；
- 按顺序调用阶段；
- 传递 category/run/reviewed taxonomy 参数；
- 检查退出码和必要产物；
- 在人工核验点停止。

Pipeline 不负责：

- 直接写 SQL；
- 实现 n-gram 算法；
- 加载 embedding 模型；
- 构建 Prompt；
- 修改 Streamlit session state；
- 自动批准 Taxonomy。

---

### 2.4 合并外层 `src/` 和重复包目录

#### 当前问题

当前仓库同时存在：

```text
src/
TRENDING_WORDS/src/
TRENDING_WORDS/core/
```

入口脚本会修改 `sys.path`，并混用以下导入方式：

```python
from src.connection import ...
from core.project_context import ...
from logger import ...
from connection import ...
```

这使调用方式依赖当前工作目录，也容易出现 `src` 指向错误包的问题。

#### 建议目标结构

在不改业务逻辑的前提下，先统一到一个应用目录：

```text
TRENDING_WORDS/
  pipeline.py
  etl.py
  main.py
  trend_embedding_cluster.py
  extract_taxonomy_source.py
  build_taxonomy_reference.py
  retrieve_taxonomy_candidates.py
  build_cluster_llm_evidence.py
  build_dashboard_period_cache.py
  Dashboard.py
  pages/
  core/
    auth.py
    oracle.py
    logger.py
    data_contract.py
    project_context.py
    taxonomy_common.py
    ...
  templates/
    *.sql.j2
  configs/
    *.toml
tests/
```

#### 文件迁移建议

| 当前文件 | 目标文件 | 说明 |
|---|---|---|
| `src/connection.py` | `TRENDING_WORDS/core/oracle.py` | 避免与 AI `connection_pool.py` 混淆 |
| `src/logger.py` | `TRENDING_WORDS/core/logger.py` | 所有模块共用 |
| `src/auth.py` | `TRENDING_WORDS/core/auth.py` | Streamlit 基础设施 |
| `src/omni_config_tools.py` | `TRENDING_WORDS/core/omni_config_tools.py` | Oracle 配置工具 |
| `src/export_wide_table_only.py` | `TRENDING_WORDS/export_wide_table_only.py` | CLI 入口而非共享组件 |
| `TRENDING_WORDS/src/data_contract.py` | `TRENDING_WORDS/core/data_contract.py` | 删除第二个 `src` |
| `src/templates/*` | `TRENDING_WORDS/templates/*` | 模板跟随应用 |
| `src/coding_team_prod.egg-info/*` | 删除 | 构建产物不进入 Git |

#### 导入约定

短期统一使用一种导入方式，不再同时支持多套 fallback：

```python
from core.oracle import create_oracle_connection_pool
from core.logger import get_logger
from core.project_context import ProjectContext
```

所有官方入口都从 `TRENDING_WORDS/` 目录运行。后续补充 `pyproject.toml` 后，再整体转换成正式包导入：

```python
from trending_words.core.oracle import create_oracle_connection_pool
```

#### ETL 文件命名

`etl_test.py` 实际是生产 ETL，应改名为 `etl.py`。这样可以：

- 避免 pytest 把它作为测试收集；
- 减少“这是测试脚本还是生产脚本”的歧义；
- 让 Pipeline stage 名和入口文件一致。

#### 验收标准

- Git tracked 文件中不再存在根目录 `src/`；
- 不再存在 `TRENDING_WORDS/src/`；
- 不再使用 `from src...`、`from connection...`、`from logger...`；
- 不再通过扫描父目录寻找 `src` 来修改 `sys.path`；
- `python TRENDING_WORDS/pipeline.py --help` 正常；
- 所有入口在约定的执行方式下均能正确导入共享组件。

---

### 2.5 提取可复用组件，缩短大文件

本轮只做无业务语义变化的提取，不重新设计算法。

#### 建议优先提取的组件

##### A. CLI 上下文组件

重复内容：

- `--category`；
- `--run-id`；
- `ProjectContext.from_category/active`；
- `with_run_id`；
- `ensure_directories`。

建议提供：

```python
def add_context_arguments(parser): ...
def context_from_args(args) -> ProjectContext: ...
```

##### B. Stage Runner

供 `pipeline.py` 复用：

```python
class StageRunner:
    def run(self, stage, args): ...
    def require_artifacts(self, paths): ...
```

它只处理命令、日志、退出码和文件存在性，不记录 Manifest。

##### C. AI Client

AI Insights 和 AI Research 当前有不同调用实现。建议后续统一：

- client pool；
- retry；
- stop event；
- token usage；
- output extraction；
- error normalization。

Prompt、tools 和 schema validation 仍由各业务 workflow 传入。

##### D. Cache / Result Store

统一：

- JSONL append；
- latest active record；
- cache signature；
- schema-valid filtering；
- error record。

##### E. Streamlit 展示组件

抽出以下共用组件：

- Workspace selector；
- AI 配置 sidebar；
- 运行进度；
- token summary；
- call history；
- result status badge；
- Excel download。

页面文件只保留页面布局、筛选条件和 workflow 调用。

#### 不建议的做法

- 不把所有逻辑塞进 `pipeline.py`；
- 不建立一个包含所有 helper 的 `utils.py`；
- 不为了减少文件数量合并 Prompt、Schema、Cache 和 UI；
- 不在本轮顺便修改趋势计算或 mapping 业务规则。

---

### 2.6 简化 Dashboard

#### Dashboard 应保留

- 品类和 Run 选择；
- 热词总览；
- 已归簇/待探索数量；
- 热词搜索和排序；
- 增长率、增长量、当前规模；
- Cluster 内热词对比；
- 月度趋势；
- AI Insights / AI Research 是否完成的简要状态。

#### 建议移除或收起

- Taxonomy embedding similarity 卡片；
- 与 AI Insights 页面重复的完整标签候选；
- 默认展开的 UMAP 诊断图；
- 过多的中间流水线状态；
- 仅用于工程排障的详细字段。

UMAP 可保留在“诊断信息”折叠区；完整 Taxonomy mapping 和理由放到 AI Insights 页面展示。

#### 验收标准

- Dashboard 默认视图聚焦趋势和业务结果；
- 页面不再展示 Taxonomy embedding similarity；
- 诊断信息默认折叠；
- 删除展示逻辑不会删除底层 Evidence 数据；
- AI Insights 和 AI Research 页面仍能查看完整结果。

---

## 3. 建议实施顺序

### Phase 1：目录和导入统一

1. 移动根 `src/` 文件；
2. 移动 `TRENDING_WORDS/src/data_contract.py`；
3. 移动 SQL 模板；
4. 更新全部 import；
5. 将 `etl_test.py` 改名为 `etl.py`；
6. 删除 egg-info；
7. 执行 compile/import smoke tests。

这是最先执行的阶段，因为后续重构都依赖稳定的 import 结构。

### Phase 2：移除 Manifest

1. 独立 active-run pointer；
2. 删除各入口的 Manifest 调用；
3. 删除 `run_manifest.py`；
4. 用退出码和 artifact existence 替代阶段状态；
5. 验证 Workspace selector。

### Phase 3：简化 Taxonomy 流程

1. 删除 Taxonomy embedding 生成；
2. 删除 Taxonomy embedding 读取；
3. 完整 Taxonomy 使用稳定排序；
4. 更新 Evidence contract；
5. 更新 Prompt/schema；
6. 删除 Dashboard similarity 展示。

### Phase 4：加入两阶段 Pipeline

1. 实现 `prepare`；
2. 强制停止在人工核验点；
3. 实现 `after-review`；
4. 强制显式传入审核文件；
5. 增加 CLI tests。

### Phase 5：提取共享组件和瘦身页面

1. CLI context helper；
2. Stage runner；
3. 统一 AI client；
4. 统一 result store；
5. Streamlit 共用展示组件；
6. 缩短两个 AI 页面。

---

## 4. 本轮明确不修改的内容

- n-gram 生成方式；
- 繁简转换和文本清洗规则；
- 噪声词业务规则；
- 趋势增长率公式；
- Cohesion 公式；
- UMAP/HDBSCAN 参数；
- Cluster 生成逻辑；
- Taxonomy 新标签/新属性业务规则；
- LLM Prompt 的业务判断标准；
- 外部研究主题类型；
- 新的审批系统；
- 新的业务指标或图表。

如果结构重构过程中必须调整接口，只做等价迁移，并用测试证明输入输出契约没有变化。

---

## 5. 建议增加的测试和检查

### 必须通过

```bash
python -m compileall -q TRENDING_WORDS tests
pytest -q
```

### 结构检查

```bash
# 不应再存在旧导入
rg 'from src\.|from connection import|from logger import' TRENDING_WORDS

# 不应再存在 Manifest 调用
rg 'run_manifest|create_manifest|update_stage' TRENDING_WORDS

# 不应再存在 Taxonomy embedding artifact
rg 'taxonomy_attribute_embeddings|taxonomy_embedding_index' TRENDING_WORDS
```

### Pipeline 检查

1. `pipeline.py --help` 成功；
2. `prepare` 会停在人工核验点；
3. `after-review` 缺少审核文件时失败；
4. `after-review` 传入不存在文件时失败；
5. 不能通过一个命令自动串联人工核验前后阶段。

### 回归检查

- 同一份人工核验 Taxonomy 输入，输出的属性与标签集合保持完整；
- 同一 Run 的热词、Cluster 和 Evidence query key 不变；
- Dashboard 能加载已有合法 artifact；
- AI Insights/Research cache 仍能识别旧记录，或者提供明确迁移说明。

---

## 6. 需要在编码前确认的事项

1. 人工核验后的 Excel 是否要求另存为新文件，还是允许覆盖导出的 cleaned 文件？建议另存新文件，以便保留自动清洗结果和人工修改结果的差异。
2. `latest.json` 是否仍需要自动更新，还是完全由 Dashboard 的 Workspace selector 选择 Run？
3. `export_wide_table_only.py` 和 `omni_config_tools.py` 是否属于本项目正式能力？如果只是历史工具，可移入 `tools/legacy/`，而不是进入主 `core/`。
4. AI Insights 是否确实需要每个分析对象携带全部标签？如果 Taxonomy 很大，应先测量 Prompt token 数，但本轮不重新引入 embedding 截断。
5. 旧 Run 中已经存在的 Taxonomy embedding 文件是否直接忽略，还是提供一次清理命令？

以上事项确认后，再按 Phase 1–5 分批提交代码；每个 Phase 应独立可运行、可回滚，不在同一个提交中同时修改结构和业务逻辑。

---

## 7. 补充审阅：除已讨论问题外的模块不合理之处

这一节记录 Manifest、Taxonomy embedding、Pipeline、目录合并和 Dashboard 精简之外，当前代码结构中仍值得优先处理的问题。以下建议仍以“不改变业务规则”为前提。

### 7.1 `core/` 已经变成职责过宽的集合

当前 `core/` 同时包含：

- 项目配置和路径；
- Oracle/LLM 连接；
- Prompt 文本；
- LLM Schema 校验；
- Taxonomy 业务规则；
- Cache 和 JSONL 存储；
- Dashboard 数据加载；
- Streamlit UI state 和 UI component；
- 外部研究 Topic 构造。

这些模块虽然都可被复用，但并不属于同一个抽象层。`core` 的含义因此变成“除页面和入口以外的所有代码”，维护者无法仅通过目录判断一个文件是领域逻辑、基础设施还是 UI。

#### 建议分组

在第一轮目录合并稳定后，可以逐步整理为：

```text
TRENDING_WORDS/
  domain/
    analysis_unit.py
    taxonomy_rules.py
    external_topic.py
  contracts/
    cluster_schema.py
    noise_term_schema.py
    external_topic_schema.py
    evidence.py
  infrastructure/
    oracle.py
    llm_client.py
    jsonl_store.py
    active_run.py
  workflows/
    trend.py
    cluster.py
    taxonomy.py
    insights.py
    research.py
  web/
    state.py
    components/
    loaders/
  prompts/
    cluster.py
    noise_term.py
    external_topic.py
```

不必一次性移动全部文件。判断规则可以是：

- 不依赖文件、网络、Streamlit 的业务判断放 `domain/`；
- 输入输出格式校验放 `contracts/`；
- Oracle、OpenAI、文件存储放 `infrastructure/`；
- 多步骤用例编排放 `workflows/`；
- 任何 `import streamlit` 的模块只放 `web/`；
- 大段模型指令和示例放 `prompts/`。

### 7.2 页面承担了 Workflow 和基础设施职责

`01_AI Insights.py` 和 `02_AI Research.py` 不只是页面文件。它们还直接处理：

- API Client 创建；
- 并发任务提交；
- Cache 查询和写入；
- Schema validation；
- Token 汇总；
- Excel 导出；
- 错误记录；
- 业务结果转换；
- Streamlit 组件渲染。

这会带来三个问题：

1. 业务流程只能通过 Streamlit 页面触发，难以从 CLI 或测试调用；
2. 页面 rerun 模型和长任务状态相互耦合；
3. 修改 UI 时容易意外改变 Cache 或 API 行为。

#### 建议边界

页面最终只应做：

```python
filters = render_filters(...)
selection = render_selection(...)
if start_clicked:
    result = insight_workflow.run(selection, ai_settings)
render_result(result)
```

并发、Cache、校验和 API 调用应位于 `InsightWorkflow` / `ResearchWorkflow`。Workflow 不应 import Streamlit，而是通过普通 callback 或事件对象报告进度。

### 7.3 AI 调用存在两条实现链路

当前内部 Insights 页面有自己的 `_call` 逻辑，外部 Research 使用 `ApiCaller` 和 `ConnectionPool`。两条链路都在处理相似问题：

- OpenAI-compatible client；
- thinking 参数；
- max output tokens；
- usage extraction；
- retry；
- stop；
- batch concurrency；
- error result。

它们长期并存会出现行为漂移，例如一条链路累计重试 token，另一条不累计；一条支持 stop，另一条只能等待当前请求结束。

#### 建议

统一的 `LLMClient` 只负责一次可靠调用，`BatchRunner` 负责并发和停止：

```python
class LLMClient:
    def complete(self, request: LLMRequest) -> LLMResponse: ...

class BatchRunner:
    def run(self, tasks, worker, on_progress=None): ...
    def stop(self): ...
```

内部洞察和外部研究分别负责构造 request、选择 tools 和验证 response，不把业务 Prompt 塞入通用 Client。

### 7.4 Cache、History 和 Latest Record 逻辑重复

以下模块都在处理相似的持久化问题：

- `cluster_cache.py`；
- `call_history.py`；
- `latest_result_store.py`；
- `dashboard_loader.py` 内的 latest-record 选择；
- 页面中的 current/latest cache helper。

重复点包括：

- JSONL 读取；
- JSONL append；
- ACTIVE 状态判断；
- timestamp 比较；
- business key/query key；
- schema-valid 过滤；
- 最新版本选择；
- compact。

#### 建议

建立一个小而明确的 JSONL Repository，而不是继续增加 helper：

```python
class JsonlRepository:
    def append(self, record): ...
    def read_all(self): ...
    def latest_by(self, key): ...
    def compact(self, key): ...
```

业务层通过 `ClusterInsightRepository`、`ResearchRepository` 包装它并定义 key/schema。Dashboard 只读取 Repository 的查询结果，不再自行实现 latest 算法。

### 7.5 校验职责重叠且存在旧实现残留

当前映射相关校验分散在：

- `cluster_schema.py`；
- `noise_term_schema.py`；
- `mapping_validation.py`；
- `taxonomy_business_rules.py`；
- `internal_result_validator.py`；
- `external_topic_guard.py`。

其中有些负责 JSON 结构，有些负责 Taxonomy 业务约束，有些同时做两者。调用者很难判断“完整校验一个 mapping”究竟应该调用哪个入口，旧 validator 也容易和新逻辑产生不同结论。

#### 建议

明确三层：

1. **Parse/shape**：字段是否存在、类型是否正确；
2. **Contract**：枚举、ID、analysis unit、字段组合是否合法；
3. **Business rules**：标签是否已存在、新标签是否允许、属性是否在目录中。

每一层只保留一个公共入口，并让 Cluster 与 Noise Term 复用同一个 mapping validator。Cluster/Noise schema 只校验各自特有字段。

### 7.6 配置存在多个真相源

当前配置来自：

- 品类 TOML；
- `core/config.py`；
- `core/project_paths.py`；
- `core/dashboard_config.py`；
- `core/dashboard_config_original.py`；
- 各入口脚本顶部常量；
- Streamlit secrets/environment variables。

其中 `project_paths.py` 和部分 legacy config 仍硬编码 YD、季度和旧输出目录；品类参数则已经由 `ProjectContext` 动态生成。两种体系并存时，很容易出现“命令行处理的是新 Run，Dashboard 读取的是旧路径”。

#### 建议

- `ProjectContext`：只负责品类、期间、run 和 artifact path；
- `AppSettings`：只负责模型、API endpoint、worker、retry；
- Streamlit secret/environment：作为 `AppSettings` 的输入来源，不直接散落在页面；
- 删除 `dashboard_config_original.py`；
- 无调用者后删除 `project_paths.py` 中的 legacy 常量；
- 入口文件不再复制 TOML 中已有的默认值。

### 7.7 Taxonomy 处理存在多次清洗和格式转换

Taxonomy 数据会经过抽取脚本、reference builder、candidate retrieval 和 evidence builder。多个阶段都会重新加载、规范化、过滤或转换字段，导致：

- 同一标签可能在不同阶段用不同规则处理；
- Excel、Parquet 和 JSON 中的字段名来回变化；
- 难以确认人工核验发生在第几次清洗之前或之后；
- 审核后文件可能再次被自动规则删除内容。

#### 建议建立单一契约

```text
Oracle raw
  → automatic cleaned workbook
  → HUMAN REVIEW
  → reviewed workbook
  → normalize once
  → CanonicalTaxonomy
  → Evidence
```

`CanonicalTaxonomy` 至少统一：

- `attribute_code`；
- `attribute_name`；
- `label`；
- `is_valid_label`；
- `rejection_reason`；
- 可选的人工修改来源字段。

人工核验后的内容只做格式和契约验证，不应再次套用可能改变业务结论的自动清洗规则。

### 7.8 Pandas 和 Polars 的边界不明确

趋势大数据阶段使用 Polars 是合理的，Dashboard、Excel 和 LLM Evidence 使用 Pandas 也合理。问题不在于同时使用两者，而在于转换边界没有明确规定，部分脚本可能反复读取 Parquet、转 Pandas、再写 Parquet。

#### 建议边界

- ETL、候选统计、Context 聚合、Cluster 输入：Polars；
- 模型输入记录、Excel、Streamlit 展示：Pandas 或普通 Python records；
- 模块边界使用 Parquet schema 或 dataclass contract；
- 不在同一个计算函数中来回转换 Pandas/Polars；
- 对关键 Parquet 文件建立列名、dtype 和 nullability contract test。

### 7.9 入口脚本在 import 阶段执行过多初始化

部分入口会在模块 import 时解析 bootstrap 参数、创建全局 `CONTEXT`、根据配置计算大量路径和常量，页面文件则在 import 时直接执行 Streamlit UI。

这使得：

- 单元测试难以只 import 某个纯函数；
- 不同参数的测试需要 reload module；
- CLI 参数和模块导入耦合；
- import 可能触发模型、文件或 Streamlit 状态依赖。

#### 建议

将入口统一为：

```python
def build_parser() -> argparse.ArgumentParser: ...
def run(context: ProjectContext, settings: StageSettings) -> StageResult: ...
def main(argv: list[str] | None = None) -> int: ...
```

纯函数只接收参数，不读取全局 `CONTEXT`。`main()` 是唯一读取 CLI、环境变量并执行 I/O 的位置。

### 7.10 Prompt 文件和 Schema 文件过大，但不能简单合并

Prompt 与 Schema 拆开是正确方向，但当前大段常量、JSON 示例和规则说明仍然难以 review。问题不是文件数量，而是业务规则同时出现在 Prompt 文本和 Python validator 中，修改时容易只更新一侧。

#### 建议

- 为每种 contract 建立明确版本对象；
- 枚举和字段规则从同一常量来源生成 Prompt 片段和 validator；
- Prompt 示例移到独立模板文件；
- Python 文件保留 Prompt builder，不保存大量与代码混排的长字符串；
- 增加测试，确认 Prompt 宣称的 allowed values 与 Schema 实际允许值一致。

---

## 8. 补充审阅：难以理解和冗余的代码形态

### 8.1 多语句单行和压缩式代码

`etl_test.py`、`run_manifest.py`、部分 Research 页面和工具脚本存在大量如下形式：

```python
a=parse_args();c=load_context(a);s=settings(c);c.ensure_directories()
if not files:return
```

这会降低可读性，并让断点、异常定位、代码审查和 diff 都更困难。建议使用 formatter 统一展开，每行只表达一个主要动作。

### 8.2 大函数同时包含多个抽象层

典型大函数会同时：

1. 读取文件；
2. 清洗 DataFrame；
3. 计算指标；
4. 组装业务对象；
5. 写多种格式；
6. 更新状态；
7. 打印日志。

拆分标准不应是“每 50 行一个函数”，而应按副作用和输出契约拆分：

```python
raw = repository.load(...)
normalized = normalize(raw)
result = calculate(normalized, settings)
validate(result)
repository.save(result)
```

这样纯计算可以单测，I/O 失败也不会和业务校验混在一起。

### 8.3 重复的数值、文本和 JSON 安全转换

多个模块分别定义 `_number`、`fnum`、`inum`、`finite`、`clean_text`、`to_json_safe`、`json_dump`、`json_load`。它们名字相似，但对 `NaN`、`None`、Decimal、NumPy scalar 和空字符串的行为未必一致。

建议只抽取真正具有统一契约的三类工具：

- `normalize_text(value) -> str`；
- `finite_number(value, default=None)`；
- `to_json_value(value)`。

业务含义不同的转换仍留在各自模块，避免建立无边界的 `utils.py`。

### 8.4 重复的 latest/current/active 判断

页面、Dashboard loader、Cache 和 compact store 都会判断记录是不是：

- 最新；
- ACTIVE；
- schema valid；
- 当前 signature；
- 当前 evidence hash。

这些规则如果不集中，Dashboard 展示的“已完成”可能与页面认为的“已完成”不一致。应由 Repository 返回统一的 `CurrentRecord`，UI 不再自行拼条件。

### 8.5 重复的 Streamlit CSS 和展示格式

Dashboard、AI Insights、AI Research 都含较长 CSS、HTML card 和 status label。重复不仅增加行数，也会让同一状态在不同页面使用不同颜色和文字。

建议建立：

- `inject_base_styles()`；
- `render_metric_strip()`；
- `render_status_badge()`；
- `render_call_history()`；
- `render_download_actions()`。

不要过早建立完整 design system，只提取已经在两个以上页面重复且业务含义一致的展示。

### 8.6 SQL 同时存在模板、独立 SQL 和 Python 内联字符串

当前有 Jinja SQL 模板、`SEGMENT_LABEL.sql`，也有脚本内的大段 SQL。SQL 分布方式不一致，会让参数化、安全检查和数据库 review 困难。

建议：

- 长查询统一放 `templates/sql/`；
- Python 只传绑定参数；
- 短且稳定的单条查询可以保留在模块常量中；
- 禁止使用字符串拼接注入业务值；
- 动态表名必须通过单独 identifier validator。

### 8.7 `try/except Exception` 范围过大

部分 stage 用一个大 `try/except Exception` 包住完整流程，只记录 `failed` 后重新抛出；这类包装不能增加恢复能力，反而可能让真正失败步骤不清楚。

删除 Manifest 后，应只在以下位置捕获异常：

- 可以添加业务上下文时；
- 可以清理临时文件或连接时；
- 可以转换为明确的领域错误时；
- UI 边界需要把异常转为用户提示时。

其余情况让异常自然传播，并使用 `finally` 或 context manager 清理资源。

### 8.8 注释混合历史记录、解释和临时调试信息

部分注释包含日期、开发过程问题或“NEW/改进点”等历史信息。它们对理解当前契约帮助有限。建议：

- 注释解释“为什么”，而不是重复“做什么”；
- 历史修改原因放 Git commit/ADR；
- 删除已经完成的 NEW/TODO 标记；
- 面向维护者保留算法假设、数据口径和不能简化的业务原因。

---

## 9. 补充优先级：哪些应该先改

### P0：结构调整前必须确认

1. 当前补充的运行环境文件是否已经成为依赖的唯一真相源；
2. 人工审核文件的正式命名和保存位置；
3. 旧 Run/Cache 是否必须向后兼容；
4. 外层 `src` 中哪些工具属于主项目，哪些是历史工具。

### P1：与已提出重构一起完成

1. 统一目录和 imports；
2. 重命名生产 `etl_test.py`；
3. 删除 Manifest；
4. 建立人工 Taxonomy checkpoint；
5. 删除 Taxonomy embedding；
6. 统一 Taxonomy canonical contract；
7. 删除明显 legacy 模块和配置；
8. 建立 CLI context helper 和 Stage Runner。

### P2：紧接着处理的可维护性问题

1. 从两个 Streamlit 页面提取 Workflow；
2. 统一 AI Client 和 Batch Runner；
3. 统一 JSONL Repository/latest-record 规则；
4. 合并 mapping validation 入口；
5. 统一 Pandas/Polars 边界；
6. 减少 import-time 初始化；
7. 提取重复 UI component。

### P3：可以随后渐进处理

1. Prompt 模板化；
2. SQL 目录统一；
3. 类型检查覆盖；
4. 更完整的 artifact contract tests；
5. ADR/开发文档。

---

## 10. 对本次补充审阅的结论

项目最主要的问题不是单个算法写错，而是已经从一组脚本逐渐成长为完整应用后，模块边界仍保留“脚本集合”的形态。

目前值得保留的方向包括：

- `ProjectContext` 对品类和 Run 的隔离；
- Prompt 与 Schema 分离；
- Cluster 和 Noise Term 使用稳定 query key；
- AI 结果使用 signature/evidence hash；
- 内部洞察与外部研究分成两个业务阶段。

下一步不建议全面重写。更稳妥的方式是：

1. 先统一目录、导入和运行入口；
2. 再移除 Manifest 和冗余 Taxonomy embedding；
3. 固定人工审核边界与 Canonical Taxonomy；
4. 将页面中的 Workflow、AI 调用和 Repository 逐个抽出；
5. 每次只移动一个职责，并用 contract test 确认业务结果不变。

这样可以在不修改趋势算法和业务判断的情况下，显著降低大文件、重复实现、隐式状态和跨层依赖带来的维护成本。
