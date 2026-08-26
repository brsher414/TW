# -*- coding: utf-8 -*-


import os
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from logger import get_logger

logger = get_logger(__name__)
# from auth import require_login

import pandas as pd
from connection import create_oracle_connection_pool, execute_query  
# tqdm 兼容：如果 tqdm 不可用则降级
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


# =====================
# 0) 配置区（请修改）
# =====================
SERVICE_NAME = '03'                 # '02' or '03'
INPUT_PATH = r"C:\Users\feni5001\OneDrive - NIQ\request-Nina\2026年\2603R\01SKIN干扰词替换\BDF干扰词明细SKIN.xlsx" # 支持 xlsx/csv，必须包含 itemid 列（大小写/空格不敏感）
OUTPUT_PATH = r"C:\Users\feni5001\OneDrive - NIQ\request-Nina\2026年\2603R\01SKIN干扰词替换\wide_table_only.xlsx"

CODED_ITEM_TABLE = 'CODED_TRANS_ITEM'  # 或其他包含 CODED_ITEM 的表名（仅字母数字下划线）
MONTH_WINDOW = 37                      # 时间窗口（月）
IN_CHUNK_SIZE = 900                    # IN 子句分块（<= 900 更稳）


# =====================
# 1) 强制使用项目连接（connection）
# =====================

def _ensure_src_importable():
    """尽可能把 repo root 加入 sys.path，以便 import connection。"""
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, Path.cwd(), Path.cwd().parent]
    for base in candidates:
        if (base / 'src').is_dir() and str(base) not in sys.path:
            sys.path.insert(0, str(base))
_ensure_src_importable()
 


def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def normalize_itemid_column(df: pd.DataFrame) -> pd.DataFrame:
    """自动识别 itemid 列：忽略大小写与空格"""
    cols_norm = {c: re.sub(r'\s+', '', str(c)).lower() for c in df.columns}
    inv = {v: k for k, v in cols_norm.items()}
    if 'itemid' not in inv:
        raise ValueError(f'输入文件未找到 itemid 列。当前列：{list(df.columns)}')
    item_col = inv['itemid']
    out = df.copy()
    out['itemid'] = out[item_col].astype(str).str.strip()
    return out


def safe_ident(name: str) -> str:
    """简单 identifier 校验，避免把奇怪字符注入到表名"""
    if not re.fullmatch(r'[A-Za-z0-9_]+', name or ''):
        raise ValueError(f'Illegal identifier: {name}')
    return name


def run_query(pool, sql: str, params: Optional[Dict] = None, return_header: bool = False):
    """统一调用你项目里的 execute_query"""
    return execute_query(pool, sql, params=params or {}, return_header=return_header)


def read_input(path: str) -> pd.DataFrame:
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine='openpyxl')
    return normalize_itemid_column(df)


def postprocess_data_source(df: pd.DataFrame) -> pd.DataFrame:
    """将宽表里的 DATA_SOURCE JSON 字段解析为 DATA_SOURCE 键（保持和你原逻辑一致）"""
    if 'DATA_SOURCE' not in df.columns:
        return df
    out = df.copy()
    out['DATA_SOURCE'] = (
        out['DATA_SOURCE']
        .fillna('{}')
        .apply(lambda x: x if isinstance(x, str) else (x.decode() if hasattr(x, 'decode') else str(x)))
        .apply(json.loads)
        .apply(lambda x: x.get('DATA_SOURCE', ''))
    )
    return out


def get_segments(pool, catcode: str) -> List[Tuple[str, str]]:
    sql = (
        'SELECT segno, segtype '
        'FROM omni_db_dic_segment '
        'WHERE catcode = :catcode AND segno NOT IN (20) '
        'GROUP BY segno, segtype'
    )
    segs = run_query(pool, sql, params={'catcode': catcode})
    segs = [(str(a), str(b)) for a, b in segs] if segs else []
    # ensure PACKSIZE
    if segs and ('1526', 'PACKSIZE') not in segs:
        segs.append(('1526', 'PACKSIZE'))
    return segs


def build_wide_sql(catcode: str, coded_item_table_safe: str, segments: List[Tuple[str, str]], month_window: int) -> str:
    """复刻你原 Streamlit 宽表 SQL（动态 pivot）"""
    if not segments:
        # non-report category
        return (
            'select :catcode catcode , ni.itemid , ni.prod_id , ns.extra_info data_source , '
            'e.website , get_web_link(e.website, ni.prod_id) web_link , '
            'ni.CATEGORY_LEVEL_I , ni.CATEGORY_LEVEL_II , ni.CATEGORY_LEVEL_III , '
            'ni.CATEGORY_LEVEL_IV , ni.CATEGORY_LEVEL_LEAF , ni.STORE_ID , '
            'ni.brand "RAW_BRAND" , ni.PROD_DESC_RAW , ni."ATTRIBUTE" '
            'from new_item ni '
            'join new_sales ns on ni.itemid = ns.itemid '
            'join coded_trans_catcode sc on sc.ITEMID = ni.ITEMID '
            'join DB_UNIVERSE_RETAILER e on e.storecode = sc.storecode '
            'join DB_MARKET_RETAIL d on d.STORE = e.STORENAME '
            "where ni.prod_id = :prod_id and d.mktid = '线上总和/CN' and e.dup_storeid is null"
        )

    pivot_cols = ', '.join([f"{segno} AS \"{segtype[:10]}_{segno}\"" for segno, segtype in segments])
    has_subbrand = any(segno == '300' and segtype.upper() == 'SUBBRAND' for segno, segtype in segments)

    seg_select_cols = ',\n       '.join([
        f"cr.\"{segtype[:10]}_{segno}_SNAME\" AS \"{segtype[:10]}_{segno}\""
        for segno, segtype in segments if segno not in ('2126', '1526', '300')
    ])
    seg_block = (',\n       ' + seg_select_cols) if seg_select_cols else ''
    subbrand_select = ' , cr.subbrand_300_sname subbrand_300 ' if has_subbrand else ''

    sql = f"""
with segname as (
  select CATCODE, SEGNO, SEGCODE, ESEGMENT || '(' || CSEGMENT || ')' as SNAME
  from omni_db_dic_segment
  group by CATCODE, SEGNO, SEGCODE, CSEGMENT, ESEGMENT
),
prod_id_filter as (
  select ITEMID from new_item where prod_id = :prod_id
),
coding_result as (
  select * from (
    select to_char(cti.ITEMID) ITEMID, cti.PERIODCODE, cti.ATTRNO, cti.ATTRVALUE, cti.STORECODE, sc.SNAME
    from {coded_item_table_safe} cti
    left join segname sc
      on sc.CATCODE = :catcode and sc.segno = cti.attrno and sc.segcode = cti.ATTRVALUE
    where exists (select 1 from prod_id_filter where prod_id_filter.ITEMID = cti.ITEMID)
      and cti.periodcode >= TO_CHAR(ADD_MONTHS(SYSDATE, -{month_window}), 'YYYY') || '14' || TO_CHAR(ADD_MONTHS(SYSDATE, -{month_window}), 'mm')
  )
  PIVOT (
    MAX(ATTRVALUE) as ATTRVALUE,
    MAX(SNAME) as SNAME
    for ATTRNO in ( {pivot_cols} )
  )
)
select :catcode catcode,
       ni.itemid, ni.prod_id,
       ns.extra_info data_source,
       e.website,
       get_web_link(e.website, ni.prod_id) web_link,
       ni.STORE_ID,
       ni.brand "RAW_BRAND",
       ni.PROD_DESC_RAW,
       ni."ATTRIBUTE",
       cr.brand_2126_sname brand_2126{subbrand_select},
       nvl(cmb.ECMANU, 'O.MANU') || '(' || nvl(cmb.ECCMANU, '其他厂商') || ')' as "MANU"
       {seg_block}
       , packsize_1526_attrvalue packsize
       , cts.adjunit unit
       , nvl(cts.promo_sales, cts.adjsales) sales_value_after_discount
from CODING_RESULT cr
join new_item ni on ni.itemid = cr.itemid
left join db_cate_manu_brand cmb
  on cmb.catcode = :catcode
 and cmb.ecbrandcode = cr.BRAND_2126_ATTRVALUE
left join coded_trans_sales cts on cts.itemid = cr.itemid
join new_sales ns on cr.itemid = ns.itemid
join DB_UNIVERSE_RETAILER e on e.storecode = cr.storecode
join DB_MARKET_RETAIL d on d.STORE = e.STORENAME
where d.mktid = '线上总和/CN'
  and e.dup_storeid is null
"""
    return sql


def main():
    # 读取输入 itemid
    input_df = read_input(INPUT_PATH)
    itemids_unique = (
        input_df['itemid']
        .dropna().astype(str).str.strip()
        .loc[lambda s: s != '']
        .drop_duplicates()
        .tolist()
    )
    print('Input rows:', len(input_df), 'Unique itemids:', len(itemids_unique))
    if not itemids_unique:
        raise ValueError('输入 itemid 为空')

    # 创建连接池（项目封装）
    pool = create_oracle_connection_pool(service=SERVICE_NAME)
    print('Pool ready via connection')

    # itemid -> prod_id
    sql_new_item = 'SELECT itemid, prod_id FROM new_item WHERE itemid IN ({placeholders})'
    rows = []
    for chunk in tqdm(chunked(itemids_unique, IN_CHUNK_SIZE), desc='Map itemid->prod_id'):
        placeholders = ','.join([f':v{i}' for i in range(len(chunk))])
        params = {f'v{i}': chunk[i] for i in range(len(chunk))}
        rows.extend(run_query(pool, sql_new_item.format(placeholders=placeholders), params=params))
    map_df = pd.DataFrame(rows, columns=['ITEMID', 'PROD_ID'])
    map_df['ITEMID'] = map_df['ITEMID'].astype(str)
    map_df['PROD_ID'] = map_df['PROD_ID'].astype(str)

    prod_ids = map_df['PROD_ID'].dropna().drop_duplicates().tolist()
    print('Unique prod_id:', len(prod_ids))
    if not prod_ids:
        raise ValueError('未在 NEW_ITEM 中匹配到任何 prod_id')

    # prod_id -> catcode(bundle)
    sql_prod_cat = f"""
SELECT DISTINCT ni.PROD_ID, ctc.BUNDLE
FROM CODED_TRANS_CATCODE ctc
JOIN NEW_ITEM ni ON ni.ITEMID = ctc.ITEMID
WHERE ni.PROD_ID IN ({{placeholders}})
  AND ni.PERIODCODE >= TO_CHAR(ADD_MONTHS(SYSDATE, -{MONTH_WINDOW}), 'YYYY') || '14' || TO_CHAR(ADD_MONTHS(SYSDATE, -{MONTH_WINDOW}), 'mm')
"""
    pc_rows = []
    for chunk in tqdm(chunked(prod_ids, IN_CHUNK_SIZE), desc='Map prod_id->catcode(bundle)'):
        placeholders = ','.join([f':p{i}' for i in range(len(chunk))])
        params = {f'p{i}': chunk[i] for i in range(len(chunk))}
        pc_rows.extend(run_query(pool, sql_prod_cat.format(placeholders=placeholders), params=params))
    prod_cat_df = pd.DataFrame(pc_rows, columns=['PROD_ID', 'CATCODE'])
    prod_cat_df['PROD_ID'] = prod_cat_df['PROD_ID'].astype(str)
    prod_cat_df['CATCODE'] = prod_cat_df['CATCODE'].astype(str)

    prod_cat = prod_cat_df.groupby('PROD_ID')['CATCODE'].apply(lambda x: sorted(set(x.dropna()))).to_dict()
    print('prod_id with catcode:', len(prod_cat))

    coded_item_table_safe = safe_ident(CODED_ITEM_TABLE)

    wide_frames = []
    wide_errors = []

    for prod_id, catcodes in tqdm(prod_cat.items(), desc='Query wide_table'):
        for catcode in catcodes:
            try:
                segs = get_segments(pool, catcode)
                sql = build_wide_sql(catcode, coded_item_table_safe, segs, MONTH_WINDOW)
                data, header = run_query(pool, sql, params={'catcode': catcode, 'prod_id': prod_id}, return_header=True)
                df = pd.DataFrame(data, columns=header)
                df = postprocess_data_source(df)
                wide_frames.append(df)
            except Exception as e:
                wide_errors.append({'PROD_ID': prod_id, 'CATCODE': catcode, 'ERROR': str(e)})

    wide_table = pd.concat(wide_frames, ignore_index=True) if wide_frames else pd.DataFrame()
    print('wide_table rows:', len(wide_table), 'errors:', len(wide_errors))

    # 输出仅一个 sheet：wide_table
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        wide_table.to_excel(writer, sheet_name='wide_table', index=False)

    print('Exported:', OUTPUT_PATH)

    # 可选：错误落地
    if wide_errors:
        err_path = os.path.splitext(OUTPUT_PATH)[0] + '_errors.csv'
        pd.DataFrame(wide_errors).to_csv(err_path, index=False, encoding='utf-8-sig')
        print('Wide table errors saved to:', err_path)


if __name__ == '__main__':
    main()
