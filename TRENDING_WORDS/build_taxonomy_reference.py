from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from core.project_context import ProjectContext
from core.run_manifest import create_manifest, update_stage
from core.taxonomy_common import (
    EXCLUDED_DIMENSION_CODES, attribute_directory, category_context, encode_texts,
    load_taxonomy, normalize_range_label, valid_labels_by_code,
)

def context_from_args(args):
    context = ProjectContext.from_category(args.category, project_root=ROOT) if args.category else ProjectContext.active(project_root=ROOT)
    return context.with_run_id(args.run_id) if args.run_id else context

def run(source: Path, output_dir: Path, model_name: str) -> None:
    df = load_taxonomy(source); output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "taxonomy_source_normalized.parquet", index=False)
    rejected = df[~df.is_valid_label][["attribute_code","attribute_name","label","label_rejection_reason"]]
    rejected.to_csv(output_dir / "taxonomy_label_cleaning_audit.csv", index=False, encoding="utf-8-sig")
    (output_dir / "category_context.json").write_text(json.dumps(category_context(df), ensure_ascii=False, indent=2), encoding="utf-8")
    labels = valid_labels_by_code(df); records=[]
    for item in attribute_directory(df):
        code,name=item["attribute_code"],item["attribute_name"]; current=labels.get(code,[])
        text=f"属性代码：{code}；属性名称：{name}"
        if current: text += "；现有有效标签：" + "、".join(normalize_range_label(x) for x in current)
        records.append({"attribute_code":code,"attribute_name":name,"retrieval_text":text,"total_valid_label_count":len(current),"all_labels_json":json.dumps(current,ensure_ascii=False)})
    reference=pd.DataFrame(records).sort_values("attribute_code").reset_index(drop=True)
    forbidden=EXCLUDED_DIMENSION_CODES & set(reference.attribute_code)
    if forbidden: raise AssertionError(f"非属性维度进入 Reference: {sorted(forbidden)}")
    reference.to_parquet(output_dir / "taxonomy_reference.parquet", index=False)
    vectors=encode_texts(reference.retrieval_text.tolist(), model_name)
    np.save(output_dir / "taxonomy_attribute_embeddings.npy", vectors)
    reference[["attribute_code","attribute_name"]].to_parquet(output_dir / "taxonomy_embedding_index.parquet", index=False)
    summary={"created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"source":str(source),"excluded_dimensions":sorted(EXCLUDED_DIMENSION_CODES),"attribute_count":len(reference),"valid_label_count":int(reference.total_valid_label_count.sum()),"removed_placeholder_rows":len(rejected),"removed_by_reason":rejected.label_rejection_reason.value_counts().to_dict()}
    (output_dir / "build_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[OK] attributes={len(reference)} valid_labels={summary['valid_label_count']}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--category"); ap.add_argument("--run-id"); ap.add_argument("--taxonomy",type=Path); ap.add_argument("--output-dir",type=Path); ap.add_argument("--model")
    args=ap.parse_args(); context=context_from_args(args); context.ensure_directories(); create_manifest(context)
    source=args.taxonomy or context.taxonomy_source_file; output=args.output_dir or context.taxonomy_dir; model=args.model or str(context.taxonomy.get("reference_embedding_model",context.cluster.get("embedding_model","BAAI/bge-m3")))
    if not source.exists(): raise FileNotFoundError(f"Taxonomy 文件不存在：{source}")
    update_stage(context,"taxonomy","building_reference")
    try:
        run(source,output,model)
        update_stage(context,"taxonomy","reference_completed",artifacts={"taxonomy_source_normalized":output/"taxonomy_source_normalized.parquet","taxonomy_reference":output/"taxonomy_reference.parquet","taxonomy_embeddings":output/"taxonomy_attribute_embeddings.npy","taxonomy_category_context":output/"category_context.json","taxonomy_cleaning_audit":output/"taxonomy_label_cleaning_audit.csv"})
    except Exception: update_stage(context,"taxonomy","failed"); raise
if __name__=="__main__": main()
