"""Category-isolated Oracle ETL. Shared Oracle utilities stay in ../src/connection.py."""
from __future__ import annotations
import argparse,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import polars as pl
TRENDING_ROOT=Path(__file__).resolve().parent;WORKSPACE_ROOT=TRENDING_ROOT.parent
for required in (WORKSPACE_ROOT/"src"/"connection.py",WORKSPACE_ROOT/"src"/"logger.py"):
    if not required.exists(): raise FileNotFoundError(f"缺少共享模块：{required}")
for root in (WORKSPACE_ROOT,TRENDING_ROOT):
    if str(root) not in sys.path:sys.path.insert(0,str(root))
from src.connection import create_oracle_connection_pool,execute_query,stream_query_to_csv
from src.logger import get_logger
from core.project_context import ProjectContext
from core.run_manifest import create_manifest,set_active_run,update_stage
logger=get_logger(__name__)
def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--category");p.add_argument("--run-id");p.add_argument("--bundle");p.add_argument("--periodcode");p.add_argument("--service",choices=("02","03"));p.add_argument("--limit-rows",type=int);p.add_argument("--partition-limit",type=int);p.add_argument("--keep-csv",action="store_true");return p.parse_args()
def load_context(a):
    c=ProjectContext.from_category(a.category,project_root=TRENDING_ROOT) if a.category else ProjectContext.active(project_root=TRENDING_ROOT)
    return c.with_run_id(a.run_id) if a.run_id else c
def settings(c):
    e=c.config["etl"];return {"service":str(e.get("service","03")),"max_workers":max(1,int(e.get("max_workers",6))),"parallel_hint":max(1,int(e.get("parallel_hint",4))),"sample_ratio":float(e.get("sample_ratio",.8)),"debug_timeout":max(1,int(e.get("query_timeout_ms_debug",60000)))}
def partitions(pool,*,start,end,hint,bundle,limit=None):
    base="""SELECT DISTINCT t2.bundle,t1.periodcode FROM new_item t1 JOIN coded_trans_catcode t2 ON t1.itemid=t2.itemid JOIN new_sales t3 ON t1.itemid=t3.itemid WHERE t1.periodcode BETWEEN :start_period AND :end_period AND t2.bundle IN (SELECT DISTINCT catcode FROM db_cate_segment) AND t2.bundle=:bundle ORDER BY t2.bundle,t1.periodcode"""
    params={"start_period":start,"end_period":end,"bundle":bundle};sql=f"SELECT /*+parallel({hint})*/ * FROM ({base})"
    if limit and limit>0:sql+=" WHERE ROWNUM<=:limit";params["limit"]=limit
    rows,_=execute_query(pool,sql,params=params,return_header=True);return [(str(b),str(p)) for b,p in rows]
def export_partition(pool,*,c,bundle,periodcode,hint,ratio,timeout,limit_rows=None,keep_csv=False):
    if limit_rows and limit_rows>0:
        sql=f"""SELECT /*+parallel({hint})*/ t1.itemid,t1.prod_id,t1.periodcode,t1.brand,t1.prod_desc_raw,t2.flag,t2.catcode,t2.bundle,CAST(0 AS NUMBER) AS svad FROM new_item t1 JOIN coded_trans_catcode t2 ON t1.itemid=t2.itemid WHERE t1.periodcode=:periodcode AND t2.bundle=:bundle AND ROWNUM<=:limit_rows"""
    else:
        sql=f"""SELECT /*+parallel({hint})*/ * FROM (SELECT t1.itemid,t1.prod_id,t1.periodcode,t1.brand,t1.prod_desc_raw,t2.flag,t2.catcode,t2.bundle,NVL(t3.promo_sales_value,t3.sales_value) AS svad,COUNT(*) OVER (PARTITION BY t2.bundle,t1.periodcode) AS total_rows,ROW_NUMBER() OVER (PARTITION BY t2.bundle,t1.periodcode ORDER BY NVL(t3.promo_sales_value,t3.sales_value) DESC,t1.itemid) AS rn FROM new_item t1 JOIN coded_trans_catcode t2 ON t1.itemid=t2.itemid JOIN new_sales t3 ON t1.itemid=t3.itemid WHERE t1.periodcode=:periodcode AND t2.bundle=:bundle) x WHERE rn<=CEIL(total_rows*:sample_ratio) ORDER BY rn"""
    safe="".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in bundle);csvp=c.staging_dir/f"products_{safe}_{periodcode}.csv";parq=c.staging_dir/f"products_{safe}_{periodcode}.parquet";params={"periodcode":periodcode,"bundle":bundle}
    if limit_rows and limit_rows>0:params["limit_rows"]=limit_rows
    else:params["sample_ratio"]=ratio
    try:rows=stream_query_to_csv(pool,sql,csvp,params=params,query_timeout_ms=timeout if limit_rows else None)
    except Exception as exc:
        if limit_rows and "DPY-4024" in str(exc):
            fallback="""SELECT t1.itemid,t1.prod_id,t1.periodcode,t1.brand,t1.prod_desc_raw,CAST(NULL AS VARCHAR2(50)) AS flag,CAST(NULL AS VARCHAR2(50)) AS catcode,:bundle AS bundle,CAST(0 AS NUMBER) AS svad FROM new_item t1 WHERE t1.periodcode=:periodcode AND ROWNUM<=:limit_rows""";rows=stream_query_to_csv(pool,fallback,csvp,params=params,query_timeout_ms=timeout)
        else:raise
    if rows==0:csvp.unlink(missing_ok=True);return ""
    pl.scan_csv(csvp,infer_schema_length=0).sink_parquet(parq,compression="snappy")
    if not keep_csv:csvp.unlink(missing_ok=True)
    logger.info("partition %s/%s rows=%d -> %s",bundle,periodcode,rows,parq);return str(parq)
def main():
    a=parse_args();c=load_context(a);s=settings(c);c.ensure_directories();create_manifest(c)
    bundle=str(a.bundle or c.source["bundle"]);start=str(c.period["start_period"]);end=str(c.period["end_period"]);service=str(a.service or s["service"]);ratio=float(s["sample_ratio"])
    if not 0<ratio<=1:raise ValueError("etl.sample_ratio 必须位于 (0,1]")
    if a.limit_rows is not None and not a.periodcode:raise ValueError("--limit-rows 只能与 --periodcode 同用")
    update_stage(c,"etl","running");pool=None
    try:
        pool=create_oracle_connection_pool(service=service)
        if a.periodcode:
            path=export_partition(pool,c=c,bundle=bundle,periodcode=str(a.periodcode),hint=int(s["parallel_hint"]),ratio=ratio,timeout=int(s["debug_timeout"]),limit_rows=a.limit_rows,keep_csv=a.keep_csv);update_stage(c,"etl","debug_completed" if path else "debug_empty",artifacts={"debug_partition":path} if path else None);return
        parts=partitions(pool,start=start,end=end,hint=int(s["parallel_hint"]),bundle=bundle,limit=a.partition_limit)
        if not parts:update_stage(c,"etl","empty");return
        files=[];failed=0
        with ThreadPoolExecutor(max_workers=min(int(s["max_workers"]),len(parts))) as ex:
            fm={ex.submit(export_partition,pool,c=c,bundle=b,periodcode=p,hint=int(s["parallel_hint"]),ratio=ratio,timeout=int(s["debug_timeout"]),keep_csv=a.keep_csv):(b,p) for b,p in parts}
            for done,f in enumerate(as_completed(fm),1):
                try:
                    path=f.result()
                    if path:files.append(path)
                except Exception:failed+=1;logger.exception("partition failed %s",fm[f])
                logger.info("progress %d/%d failures=%d",done,len(parts),failed)
        if failed:raise RuntimeError(f"{failed}/{len(parts)} 个分区失败，不更新正式文件")
        if not files:update_stage(c,"etl","empty");return
        pl.scan_parquet(sorted(files)).sink_parquet(c.sampled_products_file,compression="snappy");update_stage(c,"etl","completed",artifacts={"sampled_products":c.sampled_products_file});set_active_run(c);logger.info("ETL complete -> %s",c.sampled_products_file)
    except Exception:update_stage(c,"etl","failed");logger.exception("ETL failed category=%s run=%s",c.category_code,c.run_id);raise
    finally:
        if pool is not None:
            try:pool.close()
            except Exception:logger.warning("pool close failed")
if __name__=="__main__":main()
