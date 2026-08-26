from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.cluster_loader import load_cluster_evidence
from core.project_context import ProjectContext
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--category");ap.add_argument("--run-id");ap.add_argument("--evidence",type=Path);a=ap.parse_args()
    c=ProjectContext.from_category(a.category,project_root=ROOT) if a.category else ProjectContext.active(project_root=ROOT);c=c.with_run_id(a.run_id) if a.run_id else c;path=a.evidence or c.cluster_evidence_file
    records,report=load_cluster_evidence(path,strict=True);print(f"[OK] category={c.category_code} run_id={c.run_id} valid_records={len(records)} warnings={report.warning_count}")
if __name__=="__main__":main()
