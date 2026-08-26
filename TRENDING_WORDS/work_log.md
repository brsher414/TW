# Work Log: 商品属性热词发现系统

## 项目状态
- **当前阶段**: 初始化
- **最后更新**: 2026-06-02

## 核心约束（永不改变）
1. **Polars Only**: 严禁 `import pandas`。
2. **No Loops**: 严禁 `for` 循环遍历行。
3. **No POS/NER**: 严禁加载 NLP 模型。
4. **Input**: `data/v2_products_sampled.parquet`。
5. **Output**: `outputs/hot_words_report.json`。

## 文件路径与用途
- `src/data_contract.py`: Pydantic 数据契约（已锁定）。
- `src/config.py`: 阈值配置（待 AI 生成）。
- `src/ngram_pipeline.py`: 核心统计逻辑（待 AI 生成）。
- `src/reporter.py`: 差分与报告生成（待 AI 生成）。
- `main.py`: 入口（待 AI 生成）。
- `etl_test.py`: 数据导出脚本（已存在，AI 只读）。

## 决策记录
- 2026-06-02: 决定使用 L2 层作为本期数据，L1 作为上期数据。

## 待办事项 (TODO)
- [ ] AI 生成 `src/config.py`
- [ ] AI 生成 `src/ngram_pipeline.py`
- [ ] AI 生成 `src/reporter.py`
- [ ] AI 生成 `main.py`
- [ ] 人类测试运行
- [ ] 性能验证

## 已知问题 / Bug 记录
- 无