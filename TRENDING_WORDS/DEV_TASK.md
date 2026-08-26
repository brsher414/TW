# 开发任务书：商品属性热词自动化发现系统

> **版本**：v2.0（无 POS/NER · Polars + Pydantic · Parquet）
> **日期**：2026-06-02
> **作者**：（你的名字）
> **目标读者**：AI Agent（Cursor / Claude Code / GPT-4o）负责生成代码；人类负责校验

---

## 1. 项目概述

### 1.1 背景
现有 BERT 分类模型是基于历史数据训练的，但电商商品标题的写法、卖点、包装形态一直在漂移（新功能宣称如「低GI」、新包装如「泵头」、新场景词如「旅行装」）。人工维护规则既慢又不可持续。

### 1.2 目标
构建一个 **离线、只读、可重复运行** 的批处理管道：

- 输入：抽样商品标题（Parquet，已由 Oracle 分层导出）
- 处理：纯统计（n-gram × 邻域熵 × 跨期差分 × 分桶隔离）
- 输出：一份 `hot_words_report.json`，列出「哪些品类 / 哪些词 / 涨了多少 / 证据标题」

**全程不使用 POS、NER、LLM 推理。**

### 1.3 核心策略（一句话）
> 利用旧模型给出的 `cat_id`（分桶）和 `probability`（分层：L1/L2/L3），在 **L2 低置信区** 抓「写法范式正在漂移」的信号；用邻域熵把僵化营销模板滤掉；用差分证明它「变热了」而不是「一直都在」。

---

## 2. 硬性约束（AI 不允许违反）

| # | 约束 | 原因 |
|---|---|---|
| C1 | **必须用 Polars**（≒ 不允许 `import pandas` / `import pd`） | 2000万 级吞吐 & 内存 |
| C2 | **必须用 Pydantic** 定义输入输出契约 | 让 Agent 有明确 schema，减少幻觉 |
| C3 | 输入文件格式 **必须是 Parquet** | 自带 schema + 压缩 + 列式快 |
| C4 | **禁止 POS / NER / 任何 NLP 模型加载** | 成本、维护、Recall 杀手 |
| C5 | **禁止 Python-level `for` / `while` 遍历行**（`pl.iter_rows()` 也算） | 性能杀手；必须向量化 / groupby / window |
| C6 | 所有路径可离线跑（无外网 API 调用） | 生产合规 |

---

## 3. 文件结构（Agent 应按此生成）

```
project_root/
├── data/
│   └── v2_products_sampled.parquet   # ← 你（人）提供的输入
├── src/
│   ├── __init__.py
│   ├── config.py                      # 阈值/常量
│   ├── data_contract.py               # Pydantic models（⬇ 第4节）
│   ├── ngram_pipeline.py              # 核心统计（n-gram / 熵 / 依附度）
│   └── reporter.py                    # 差分筛选 → 组装报告
├── main.py                            # 入口（lazy scan → collect → report）
└── outputs/
    └── hot_words_report.json          # 最终产物
```

---

## 4. 数据契约（⚠️ 让 Agent 先写/对齐这个文件）

> Agent 应创建 `src/data_contract.py`，内容如下（**原样照搬**）：

```python
# src/data_contract.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ProductRow(BaseModel):
    """Parquet 里每条记录的期望字段（由 validate_scan 做类型断言）"""

    id: int = Field(..., description="商品ID / 主键")
    clean_title: str = Field(..., description="清洗后的标题（空格分词形态也可）")
    cat_id: int = Field(..., ge=1, le=999, description="旧模型预测的品类ID / 桶号")
    probability: float = Field(..., ge=0.0, le=1.0, description="旧模型预测置信度")
    layer: str = Field(..., description="'L1'(≥0.95) / 'L2'(0.60~0.94) / 'L3'(<0.60)")


class NGramRecord(BaseModel):
    """n-gram 在一个 cat_id 桶里的统计快照"""

    term: str
    cat_id: int
    df: int = Field(..., ge=1, description="出现文档数")
    left_entropy: float = Field(0.0, description="左邻域信息熵")
    right_entropy: float = Field(0.0, description="右邻域信息熵")
    attach_ratio: float = Field(0.0, description="与 cat_id 的共现依附度")


class HotWordResult(BaseModel):
    """报告里的一条发现"""

    cat_id: int
    term: str
    df_old: int
    df_new: int
    growth: float
    neighbor_entropy: float
    attach_ratio: float
    evidence_samples: List[str] = Field(default_factory=list)


class HotWordsReport(BaseModel):
    items: List[HotWordResult]
```

---

## 5. 输入 Parquet 的期望 Schema（Agent 必须做 assert）

| 列名 | 类型 | 说明 |
|---|---|---|
| `id` | Int64 | 商品ID |
| `clean_title` | Utf8 | 清洗后标题（空格分词 / 或仍可空格 split） |
| `cat_id` | Int64 | 分层桶（旧模型输出） |
| `probability` | Float64 | 置信度 |
| `layer` | Utf8 | `'L1'` / `'L2'` / `'L3'` |

Agent 应在 `main.py` 启动后用一次 **scan + schema 断言**（`pl.scan_parquet(...).schema`），若缺列立刻退出并提示。

---

## 6. 核心算法（Agent 要实现的东西）

### 6.1 预处理（Polars lazy）

1. **切分 tokens**（标题已是空格分隔时最简单；不是的话用 `str.split(' ')`）
2. 生成 **2-gram list**（每条记录产生 N-1 个二元组）
3. `explode` → 得到 `(cat_id, term)` 一行一个二元组（并携带一个 `title_sample` 用于证据回填）

### 6.2 统计 per (cat_id, term)

对每个桶算：

- **df** = `count(distinct id)`（文档频）
- **左/右邻词频** → 由 explode 后的行用窗口/lead-lag 统计分布
- **邻域熵**  
  \[
  H=\sum p\log p,\quad p=\frac{\text{邻词频}}{\sum\text{邻词频}}
  \]
- **attach_ratio** = `count(cat_id, term) / count(term)`（term 出现在本 cat 的比例）

### 6.3 差分（本期 vs 上期）

最简可运行假设（交给 Agent 实现二选一）：

- **A）按 layer 伪两期**：`old = L1`, `new = L2`（工程上最稳，因为 layer 已存在于同一文件）
- **B）按时间分区**：如果 Parquet 里有 `month_key`，用 `prev_month` vs `curr_month`

筛选条件（必须同时满足）：

```text
df_new >= MIN_DF
growth = df_new / (df_old + ε) >= GROWTH_THRESHOLD
( left_entropy + right_entropy ) >= ENTROPY_THRESHOLD
```

### 6.4 证据回填

对命中的 `(cat_id, term)` 拉 3~5 条脱敏 `clean_title` 进 `evidence_samples`。

---

## 7. 验收标准（跑不通就别交）

| # | 检查 | 期望 |
|---|---|---|
| A1 | `python main.py --dry` 不报错 | schema assert OK |
| A2 | 输出 `outputs/hot_words_report.json` 合法 JSON | 可被 `json.load` |
| A3 | 报告中不含 `买一送一 / 现货 / 包邮`（动态黑名单可自生成） | 邻域熵 + 依附度把它们压掉 |
| A4 | 报告中能看到「低GI / 泵头 / 氨基酸」之类候选出现在对应品类桶 | 真阳性通路 |
| P1 | 50 万行 parquet lazy collect 在普通机器 < 2 min | Polars lazy 不吃满内存 |

