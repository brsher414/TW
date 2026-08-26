import argparse
import concurrent.futures as cf
import sys
import math
import os
from pickle import TRUE
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from opencc import OpenCC

TRENDING_ROOT = Path(__file__).resolve().parent
if str(TRENDING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRENDING_ROOT))
from core.project_context import ProjectContext
from core.run_manifest import create_manifest, set_active_run, update_stage


# ============================================================
# Config
# ============================================================

cc = OpenCC("t2s")

def _parse_bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="品类代码，例如 YD。")
    parser.add_argument("--run-id", default=None, help="可选：覆盖自动生成的 run_id。")
    parser.add_argument("--force-rebuild-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-rebuild-context", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()

ARGS = _parse_bootstrap_args()
CONTEXT = ProjectContext.from_category(ARGS.category, project_root=TRENDING_ROOT) if ARGS.category else ProjectContext.active(project_root=TRENDING_ROOT)
if ARGS.run_id:
    CONTEXT = CONTEXT.with_run_id(ARGS.run_id)
TREND_CONFIG = CONTEXT.trend
PIPELINE_CONFIG = CONTEXT.config.get("pipeline", {})
SOURCE_PATH = str(CONTEXT.sampled_products_file)
DESC_COL = str(PIPELINE_CONFIG.get("desc_col", "PROD_DESC_RAW"))
PERIOD_COL = str(PIPELINE_CONFIG.get("period_col", "PERIODCODE"))
BASE_QUARTER = str(CONTEXT.period["base_quarter"])
CURRENT_QUARTER = str(CONTEXT.period["current_quarter"])
MIN_FREQ = int(TREND_CONFIG["min_freq"])
MIN_BASE_COUNT = int(TREND_CONFIG["min_base_count"])
CANDIDATE_CHUNK_ROWS = int(PIPELINE_CONFIG.get("candidate_chunk_rows", 200_000))
CANDIDATE_WORKERS = int(PIPELINE_CONFIG.get("candidate_workers", 4))
CONTEXT_CHUNK_ROWS = int(PIPELINE_CONFIG.get("context_chunk_rows", 1_000_000))
CONTEXT_WORKERS = int(PIPELINE_CONFIG.get("context_workers", 6))
REDUCE_BATCH_FILES = int(PIPELINE_CONFIG.get("reduce_batch_files", 6))
MIN_NGRAM = int(TREND_CONFIG["min_ngram"])
MAX_NGRAM = int(TREND_CONFIG["max_ngram"])
MAX_DESC_CHARS = int(TREND_CONFIG["max_desc_chars"])
MIN_CONTEXT_DIVERSITY = int(TREND_CONFIG["min_context_diversity"])
BOUNDARY_CHAR = str(PIPELINE_CONFIG.get("boundary_char", " "))
TOP_N_DISPLAY = int(PIPELINE_CONFIG.get("top_n_display", 50))
ENABLE_COHESION_FILTER = bool(PIPELINE_CONFIG.get("enable_cohesion_filter", True))
MIN_COHESION = float(TREND_CONFIG["min_cohesion"])
FORCE_REBUILD_CACHE = bool(ARGS.force_rebuild_cache) if ARGS.force_rebuild_cache is not None else bool(PIPELINE_CONFIG.get("force_rebuild_cache", True))
FORCE_REBUILD_CONTEXT = bool(ARGS.force_rebuild_context) if ARGS.force_rebuild_context is not None else bool(PIPELINE_CONFIG.get("force_rebuild_context", True))
CACHE_DIR_PATH = str(CONTEXT.trend_dir / "char_ngram_docfreq_cache_parts")
CACHE_REDUCE_DIR_PATH = str(CONTEXT.trend_dir / "char_ngram_docfreq_reduce_parts")
FINAL_CANDIDATE_PATH = str(CONTEXT.trend_dir / "final_candidate_docfreq.parquet")
CONTEXT_PART_DIR_PATH = str(CONTEXT.trend_dir / "candidate_context_parts_allctx")
UNIGRAM_FREQ_PATH = str(CONTEXT.trend_dir / "unigram_docfreq.parquet")
TOTAL_DOCS_PATH = str(CONTEXT.trend_dir / "total_docs.txt")
TRIPLE_OUTPUT_DIR = str(CONTEXT.trend_dir)
TRIPLE_CONTAINED_NAME = "trend_contained"
TRIPLE_CONTAINER_NAME = "trend_container"
TRIPLE_LOW_COHESION_NAME = "trend_low_cohesion"

# ============================================================
# Global variable for context multiprocessing
# ============================================================

GLOBAL_CANDIDATES_BY_LEN: dict[int, set[str]] | None = None


# ============================================================
# Text processing
# ============================================================

def char_level_t2s(text: str) -> str:
    result = []
    for ch in text:
        result.append(cc.convert(ch))
    return "".join(result)


def desc_clean(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9βBαA\-\+]", "", text)


def normalize_desc(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = char_level_t2s(text)
    text = desc_clean(text)
    text = text.upper()

    if MAX_DESC_CHARS and MAX_DESC_CHARS > 0:
        text = text[:MAX_DESC_CHARS]

    return text


def make_unique_char_ngrams(
    text: str,
    min_n: int = MIN_NGRAM,
    max_n: int = MAX_NGRAM,
) -> list[str]:
    if not text or not isinstance(text, str):
        return []

    text = str(text)
    text_len = len(text)

    grams_set = set()

    for n in range(min_n, max_n + 1):
        if text_len < n:
            continue
        for i in range(text_len - n + 1):
            grams_set.add(text[i:i + n])

    return list(grams_set)


# ============================================================
# Period processing
# ============================================================

def periodcode_to_quarter(period_code) -> str | None:
    if period_code is None:
        return None

    s = str(period_code).strip()
    if not s:
        return None
    if s.endswith(".0"):
        s = s[:-2]

    m = re.fullmatch(r"(\d{4})14(\d{2})", s)
    if not m:
        return None

    year = int(m.group(1))
    month = int(m.group(2))
    if month < 1 or month > 12:
        return None

    quarter = (month - 1) // 3 + 1
    return f"{year}Q{quarter}"


# ============================================================
# Noise filter
# ============================================================

NOISE_KEYWORDS = [
    "官方", "旗舰", "旗舰店", "正品", "包邮", "买一送一", "买1送1",
    "满减", "下单", "现货", "促销", "特价", "优惠", "秒杀", "直播",
    "同款", "专柜", "授权", "保证", "保障", "新款", "爆款", "热卖",
    "推荐", "升级", "礼盒", "组合", "套装", "规格", "春节", "年货",
    "新年", "马年", "节", "代购", "生日", "礼物", "淘金币", "试用",
    "小样", "免费", "积分", "代买","新春","过年","到手","顺丰"
]

SPEC_UNIT_PATTERN = re.compile(
    r"("
    r"\d+(\.\d+)?"
    r"(g|G|kg|KG|克|千克|斤|公斤|mg|MG|毫克|ml|ML|mL|毫升|l|L|升|年|月"
    r"片|粒|颗|袋|包|盒|瓶|支|只|件|罐|枚|抽|卷|"
    r"寸|cm|CM|厘米|mm|MM|毫米|m|M|米)"
    r"|"
    r"\d+(\.\d+)?"
    r"(斤装|克装|kg装|KG装|g装|G装|ml装|ML装|毫升装|升装)"
    r"|"
    r"(一|二|三|四|五|六|七|八|九|十|\d+)"
    r"(件装|瓶装|袋装|盒装|包装|罐装|支装|只装|片装|粒装)"
    r"|"
    r"(斤装|克装|瓶装|袋装|盒装|包装|罐装|支装|只装|片装|粒装)"
    r")",
    re.IGNORECASE,
)

PURE_NUMBER_OR_SYMBOL_PATTERN = re.compile(r"^[0-9A-Za-z\-\+βBαA]+$")


def is_noise_candidate(candidate: str) -> bool:
    if not candidate:
        return True

    candidate = str(candidate).strip()
    if len(candidate) < MIN_NGRAM:
        return True

    for kw in NOISE_KEYWORDS:
        if kw in candidate:
            return True

    if SPEC_UNIT_PATTERN.search(candidate):
        return True

    if PURE_NUMBER_OR_SYMBOL_PATTERN.fullmatch(candidate):
        return True

    return False


# ============================================================
# Utility
# ============================================================

def recreate_dir(dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def clear_parquet_dir(dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return
    for file in dir_path.rglob("*.parquet"):
        file.unlink()


def cache_is_valid(cache_dir_path: str) -> bool:
    cache_dir = Path(cache_dir_path)
    part_files = sorted(cache_dir.glob("part_*.parquet"))
    return cache_dir.exists() and len(part_files) > 0


# === NEW === 帮助函数：找到现有的最终 reduce 文件
def find_final_reduce_file(reduce_dir_path: str) -> Path | None:
    reduce_root = Path(reduce_dir_path)
    if not reduce_root.exists():
        return None

    round_dirs = sorted(
        d for d in reduce_root.iterdir()
        if d.is_dir() and d.name.startswith("round_")
    )
    if not round_dirs:
        return None

    last_round = round_dirs[-1]
    files = sorted(last_round.glob("*.parquet"))
    if len(files) == 1:
        return files[0]
    return None


def print_config() -> None:
    print("=" * 100)
    print("CONFIG")
    print("=" * 100)
    print(f"SOURCE_PATH                 = {SOURCE_PATH}")
    print(f"DESC_COL                    = {DESC_COL}")
    print(f"PERIOD_COL                  = {PERIOD_COL}")
    print(f"BASE_QUARTER                = {BASE_QUARTER}")
    print(f"CURRENT_QUARTER             = {CURRENT_QUARTER}")
    print(f"MIN_FREQ                    = {MIN_FREQ}")
    print(f"MIN_BASE_COUNT              = {MIN_BASE_COUNT}")
    print(f"CANDIDATE_CHUNK_ROWS        = {CANDIDATE_CHUNK_ROWS}")
    print(f"CANDIDATE_WORKERS           = {CANDIDATE_WORKERS}")
    print(f"CONTEXT_CHUNK_ROWS          = {CONTEXT_CHUNK_ROWS}")
    print(f"CONTEXT_WORKERS             = {CONTEXT_WORKERS}")
    print(f"REDUCE_BATCH_FILES          = {REDUCE_BATCH_FILES}")
    print(f"MIN_NGRAM                   = {MIN_NGRAM}")
    print(f"MAX_NGRAM                   = {MAX_NGRAM}")
    print(f"MAX_DESC_CHARS              = {MAX_DESC_CHARS}")
    print(f"MIN_CONTEXT_DIVERSITY       = {MIN_CONTEXT_DIVERSITY}")
    print(f"BOUNDARY_CHAR               = {repr(BOUNDARY_CHAR)}")
    print(f"ENABLE_COHESION_FILTER      = {ENABLE_COHESION_FILTER}")
    print(f"MIN_COHESION                = {MIN_COHESION}")
    print(f"CACHE_DIR_PATH              = {CACHE_DIR_PATH}")
    print(f"CACHE_REDUCE_DIR_PATH       = {CACHE_REDUCE_DIR_PATH}")
    print(f"FINAL_CANDIDATE_PATH        = {FINAL_CANDIDATE_PATH}")
    print(f"CONTEXT_PART_DIR_PATH       = {CONTEXT_PART_DIR_PATH}")
    print(f"UNIGRAM_FREQ_PATH           = {UNIGRAM_FREQ_PATH}")
    print(f"TOTAL_DOCS_PATH             = {TOTAL_DOCS_PATH}")
    print(f"TRIPLE_OUTPUT_DIR           = {TRIPLE_OUTPUT_DIR}")
    print(f"FORCE_REBUILD_CACHE         = {FORCE_REBUILD_CACHE}")
    print(f"FORCE_REBUILD_CONTEXT       = {FORCE_REBUILD_CONTEXT}")
    print("CANDIDATE_FREQ_SCOPE        = all_periods")
    print("CONTEXT_SCOPE               = all_periods")
    print(f"COUNT_SCOPE                 = {BASE_QUARTER}_vs_{CURRENT_QUARTER}")
    print("=" * 100)


# ============================================================
# Build candidate docfreq cache
# ============================================================

def build_candidate_cache_part(
    source_path: str,
    part_path: str,
    start_row: int,
    chunk_rows: int,
) -> None:
    end_row = start_row + chunk_rows

    print(
        f"[PID {os.getpid()}] writing {Path(part_path).name} "
        f"rows {start_row:,}-{end_row:,}"
    )

    (
        pl.scan_parquet(source_path)
        .slice(start_row, chunk_rows)
        .select(DESC_COL)
        .with_columns(
            pl.col(DESC_COL)
            .map_elements(normalize_desc, return_dtype=pl.String)
            .alias("DESC_NORM")
        )
        .with_columns(
            pl.col("DESC_NORM")
            .map_elements(make_unique_char_ngrams, return_dtype=pl.List(pl.String))
            .alias("DESC_NGRAMS")
        )
        .select("DESC_NGRAMS")
        .explode("DESC_NGRAMS")
        .rename({"DESC_NGRAMS": "candidate"})
        .filter(
            pl.col("candidate").is_not_null()
            & (pl.col("candidate") != "")
        )
        .with_columns(
            pl.col("candidate").str.len_chars().alias("ngram_len")
        )
        .group_by(["candidate", "ngram_len"])
        .agg(pl.len().alias("freq"))
        .sink_parquet(part_path)
    )


def build_candidate_cache(
    source_path: str,
    cache_dir_path: str,
    chunk_rows: int = CANDIDATE_CHUNK_ROWS,
    workers: int = CANDIDATE_WORKERS,
) -> None:
    cache_dir = Path(cache_dir_path)
    recreate_dir(cache_dir)

    total_rows = (
        pl.scan_parquet(source_path)
        .select(pl.len())
        .collect()
        .item()
    )

    print(
        f"building candidate docfreq cache in {cache_dir_path} "
        f"with chunk_rows={chunk_rows:,}, workers={workers}, total_rows={total_rows:,}"
    )

    tasks = []

    for part_index, start_row in enumerate(
        range(0, total_rows, chunk_rows),
        start=1,
    ):
        current_chunk_rows = min(chunk_rows, total_rows - start_row)
        part_path = cache_dir / f"part_{part_index:05d}.parquet"

        tasks.append(
            (
                source_path,
                str(part_path),
                start_row,
                current_chunk_rows,
            )
        )

    if workers > 1:
        try:
            with cf.ProcessPoolExecutor(
                max_workers=workers,
                max_tasks_per_child=1,
            ) as executor:
                futures = [
                    executor.submit(build_candidate_cache_part, *task)
                    for task in tasks
                ]
                for future in cf.as_completed(futures):
                    future.result()
            return

        except Exception as exc:
            print(f"parallel candidate build failed: {exc}")
            print("retrying candidate build sequentially...")
            clear_parquet_dir(cache_dir)

    for task in tasks:
        build_candidate_cache_part(*task)


# ============================================================
# Hierarchical reduce for candidates
# ============================================================

def reduce_files_once(
    input_files: list[Path],
    output_dir: Path,
    round_index: int,
    batch_files: int = REDUCE_BATCH_FILES,
    apply_min_freq_filter: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []

    for batch_index, start_index in enumerate(
        range(0, len(input_files), batch_files),
        start=1,
    ):
        batch = input_files[start_index:start_index + batch_files]
        if not batch:
            continue

        if len(batch) == 1:
            output_files.append(batch[0])
            continue

        out_path = output_dir / f"round_{round_index:03d}_reduce_{batch_index:05d}.parquet"

        print(
            f"round {round_index}: reducing files "
            f"{start_index + 1:,}-{start_index + len(batch):,} "
            f"of {len(input_files):,} -> {out_path.name}"
        )

        lf = (
            pl.scan_parquet([str(path) for path in batch])
            .group_by(["candidate", "ngram_len"])
            .agg(pl.col("freq").sum().alias("freq"))
        )

        if apply_min_freq_filter:
            lf = lf.filter(pl.col("freq") >= MIN_FREQ)

        lf.sink_parquet(out_path)
        output_files.append(out_path)

    return output_files


def hierarchical_reduce(
    input_dir_path: str,
    output_root_path: str,
    batch_files: int = REDUCE_BATCH_FILES,
    apply_min_freq_filter_in_middle: bool = False,
) -> Path:
    input_dir = Path(input_dir_path)
    output_root = Path(output_root_path)

    input_files = sorted(input_dir.glob("part_*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"no part_*.parquet found in {input_dir_path}")

    recreate_dir(output_root)

    current_files = input_files
    round_index = 1

    print(
        f"starting hierarchical reduce with {len(current_files):,} files, "
        f"batch_files={batch_files}"
    )

    while len(current_files) > 1:
        round_dir = output_root / f"round_{round_index:03d}"
        current_files = reduce_files_once(
            input_files=current_files,
            output_dir=round_dir,
            round_index=round_index,
            batch_files=batch_files,
            apply_min_freq_filter=apply_min_freq_filter_in_middle,
        )
        print(f"round {round_index} done, files left: {len(current_files):,}")
        round_index += 1

    final_file = current_files[0]
    print(f"hierarchical reduce done, final file: {final_file}")
    return final_file


def build_final_candidate_freq(
    final_reduce_file: Path,
    final_output_path: str,
) -> pl.DataFrame:
    print(f"building final candidate docfreq from {final_reduce_file}")

    candidate_freq = (
        pl.scan_parquet(str(final_reduce_file))
        .filter(pl.col("freq") >= MIN_FREQ)
        .sort("freq", descending=True)
        .collect()
    )

    output_path = Path(final_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_freq.write_parquet(output_path)

    print(f"final candidate docfreq written to: {output_path}")
    print(f"final candidate rows: {candidate_freq.height:,}")

    return candidate_freq


# ============================================================
# Context extraction
# ============================================================

def load_high_freq_candidates(candidate_path: str) -> dict[int, set[str]]:
    df = pl.read_parquet(candidate_path)
    candidates_by_len: dict[int, set[str]] = defaultdict(set)

    for row in df.iter_rows(named=True):
        candidate = row["candidate"]
        ngram_len = int(row["ngram_len"])
        candidates_by_len[ngram_len].add(candidate)

    total = sum(len(v) for v in candidates_by_len.values())
    print(f"loaded high freq candidates: {total:,}")
    for n in sorted(candidates_by_len):
        print(f"  len={n}: {len(candidates_by_len[n]):,}")

    return candidates_by_len


def init_context_worker(candidate_path: str) -> None:
    global GLOBAL_CANDIDATES_BY_LEN
    GLOBAL_CANDIDATES_BY_LEN = load_high_freq_candidates(candidate_path)
    print(
        f"[PID {os.getpid()}] context worker initialized, "
        f"candidate lengths: {sorted(GLOBAL_CANDIDATES_BY_LEN.keys())}"
    )


def extract_candidate_context_from_text(
    text: str,
    candidates_by_len: dict[int, set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    left_context: dict[str, set[str]] = defaultdict(set)
    right_context: dict[str, set[str]] = defaultdict(set)
    appeared_candidates: set[str] = set()

    if not text:
        return left_context, right_context, appeared_candidates

    text_len = len(text)

    for n, candidate_set in candidates_by_len.items():
        if text_len < n:
            continue

        for i in range(0, text_len - n + 1):
            gram = text[i:i + n]
            if gram not in candidate_set:
                continue

            left_char = text[i - 1] if i > 0 else BOUNDARY_CHAR
            right_pos = i + n
            right_char = text[right_pos] if right_pos < text_len else BOUNDARY_CHAR

            left_context[gram].add(left_char)
            right_context[gram].add(right_char)
            appeared_candidates.add(gram)

    return left_context, right_context, appeared_candidates


def build_context_part(
    source_path: str,
    part_path: str,
    start_row: int,
    chunk_rows: int,
    candidates_by_len: dict[int, set[str]],
) -> None:
    end_row = start_row + chunk_rows

    print(
        f"[PID {os.getpid()}] writing context {Path(part_path).name} "
        f"rows {start_row:,}-{end_row:,}"
    )

    df = (
        pl.scan_parquet(source_path)
        .slice(start_row, chunk_rows)
        .select([DESC_COL, PERIOD_COL])
        .collect()
    )

    left_map: dict[str, set[str]] = defaultdict(set)
    right_map: dict[str, set[str]] = defaultdict(set)

    base_counter: Counter[str] = Counter()
    current_counter: Counter[str] = Counter()

    for row in df.iter_rows(named=True):
        raw_text = row.get(DESC_COL)
        text = normalize_desc(raw_text)
        if not text:
            continue

        row_left, row_right, appeared_candidates = extract_candidate_context_from_text(
            text=text,
            candidates_by_len=candidates_by_len,
        )

        if not appeared_candidates:
            continue

        for candidate, chars in row_left.items():
            left_map[candidate].update(chars)
        for candidate, chars in row_right.items():
            right_map[candidate].update(chars)

        quarter = periodcode_to_quarter(row.get(PERIOD_COL))
        if quarter == BASE_QUARTER:
            for candidate in appeared_candidates:
                base_counter[candidate] += 1
        elif quarter == CURRENT_QUARTER:
            for candidate in appeared_candidates:
                current_counter[candidate] += 1

    all_candidates = (
        set(left_map.keys())
        | set(right_map.keys())
        | set(base_counter.keys())
        | set(current_counter.keys())
    )

    rows = []
    for candidate in all_candidates:
        rows.append(
            {
                "candidate": candidate,
                "ngram_len": len(candidate),
                "left_chars": sorted(left_map.get(candidate, set())),
                "right_chars": sorted(right_map.get(candidate, set())),
                "base_count": int(base_counter.get(candidate, 0)),
                "current_count": int(current_counter.get(candidate, 0)),
            }
        )

    out_df = pl.DataFrame(
        rows,
        schema={
            "candidate": pl.String,
            "ngram_len": pl.Int64,
            "left_chars": pl.List(pl.String),
            "right_chars": pl.List(pl.String),
            "base_count": pl.Int64,
            "current_count": pl.Int64,
        },
    )

    out_df.write_parquet(part_path)


def build_context_part_worker(
    source_path: str,
    part_path: str,
    start_row: int,
    chunk_rows: int,
) -> None:
    global GLOBAL_CANDIDATES_BY_LEN
    if GLOBAL_CANDIDATES_BY_LEN is None:
        raise RuntimeError("GLOBAL_CANDIDATES_BY_LEN is not initialized.")

    build_context_part(
        source_path=source_path,
        part_path=part_path,
        start_row=start_row,
        chunk_rows=chunk_rows,
        candidates_by_len=GLOBAL_CANDIDATES_BY_LEN,
    )


def build_context_parts(
    source_path: str,
    context_dir_path: str,
    candidate_path: str,
    chunk_rows: int = CONTEXT_CHUNK_ROWS,
    workers: int = CONTEXT_WORKERS,
) -> list[Path]:
    context_dir = Path(context_dir_path)
    recreate_dir(context_dir)

    total_rows = (
        pl.scan_parquet(source_path)
        .select(pl.len())
        .collect()
        .item()
    )

    print(
        f"building context parts in {context_dir_path}, "
        f"chunk_rows={chunk_rows:,}, workers={workers}, total_rows={total_rows:,}"
    )

    tasks = []
    part_files: list[Path] = []

    for part_index, start_row in enumerate(
        range(0, total_rows, chunk_rows),
        start=1,
    ):
        current_chunk_rows = min(chunk_rows, total_rows - start_row)
        part_path = context_dir / f"context_part_{part_index:05d}.parquet"

        tasks.append(
            (
                source_path,
                str(part_path),
                start_row,
                current_chunk_rows,
            )
        )
        part_files.append(part_path)

    if workers > 1:
        try:
            with cf.ProcessPoolExecutor(
                max_workers=workers,
                max_tasks_per_child=1,
                initializer=init_context_worker,
                initargs=(candidate_path,),
            ) as executor:
                futures = [
                    executor.submit(build_context_part_worker, *task)
                    for task in tasks
                ]
                for future in cf.as_completed(futures):
                    future.result()
            return part_files

        except Exception as exc:
            print(f"parallel context build failed: {exc}")
            print("retrying context build sequentially...")
            recreate_dir(context_dir)

    candidates_by_len = load_high_freq_candidates(candidate_path)
    for task in tasks:
        source_path_i, part_path_i, start_row_i, chunk_rows_i = task
        build_context_part(
            source_path=source_path_i,
            part_path=part_path_i,
            start_row=start_row_i,
            chunk_rows=chunk_rows_i,
            candidates_by_len=candidates_by_len,
        )

    return part_files


# ============================================================
# Context reduce
# ============================================================

def reduce_context_parts(context_dir_path: str) -> pl.DataFrame:
    context_files = sorted(Path(context_dir_path).glob("context_part_*.parquet"))
    if not context_files:
        raise FileNotFoundError(f"no context_part_*.parquet found in {context_dir_path}")

    print(f"reducing context parts: {len(context_files):,} files")

    left_map: dict[str, set[str]] = defaultdict(set)
    right_map: dict[str, set[str]] = defaultdict(set)
    base_counter: Counter[str] = Counter()
    current_counter: Counter[str] = Counter()
    ngram_len_map: dict[str, int] = {}

    for idx, file_path in enumerate(context_files, start=1):
        print(f"reading context part {idx:,}/{len(context_files):,}: {file_path.name}")
        df = pl.read_parquet(file_path)

        for row in df.iter_rows(named=True):
            candidate = row["candidate"]
            ngram_len_map[candidate] = int(row["ngram_len"])

            left_chars = row["left_chars"] or []
            right_chars = row["right_chars"] or []

            left_map[candidate].update(left_chars)
            right_map[candidate].update(right_chars)

            base_counter[candidate] += int(row["base_count"] or 0)
            current_counter[candidate] += int(row["current_count"] or 0)

    all_candidates = (
        set(left_map.keys())
        | set(right_map.keys())
        | set(base_counter.keys())
        | set(current_counter.keys())
    )

    rows = []
    for candidate in all_candidates:
        rows.append(
            {
                "candidate": candidate,
                "ngram_len": ngram_len_map.get(candidate, len(candidate)),
                "left_chars": sorted(left_map.get(candidate, set())),
                "right_chars": sorted(right_map.get(candidate, set())),
                "base_count": int(base_counter.get(candidate, 0)),
                "current_count": int(current_counter.get(candidate, 0)),
            }
        )

    context_df = pl.DataFrame(
        rows,
        schema={
            "candidate": pl.String,
            "ngram_len": pl.Int64,
            "left_chars": pl.List(pl.String),
            "right_chars": pl.List(pl.String),
            "base_count": pl.Int64,
            "current_count": pl.Int64,
        },
    )

    print(f"context reduced rows: {context_df.height:,}")
    return context_df


def context_diversity_pass(left_chars: list[str], right_chars: list[str]) -> bool:
    left_set = set(left_chars or [])
    right_set = set(right_chars or [])

    left_non_boundary = left_set - {BOUNDARY_CHAR}
    right_non_boundary = right_set - {BOUNDARY_CHAR}

    left_boundary_only = len(left_set) > 0 and left_set <= {BOUNDARY_CHAR}
    right_boundary_only = len(right_set) > 0 and right_set <= {BOUNDARY_CHAR}

    left_div = len(left_non_boundary)
    right_div = len(right_non_boundary)

    if left_boundary_only and right_boundary_only:
        return False
    if left_boundary_only:
        return right_div >= MIN_CONTEXT_DIVERSITY
    if right_boundary_only:
        return left_div >= MIN_CONTEXT_DIVERSITY

    return (
        left_div >= MIN_CONTEXT_DIVERSITY
        and right_div >= MIN_CONTEXT_DIVERSITY
    )


def calculate_growth_rate(base_count: int, current_count: int) -> float:
    if base_count == 0:
        if current_count > 0:
            return math.inf
        return 0.0
    return (current_count - base_count) / base_count


# ============================================================
# === NEW === Unigram docfreq + total docs（凝固度查表用）
# ============================================================

def build_unigram_docfreq_and_count(
    source_path: str,
    output_path: str,
    total_docs_path: str,
    chunk_rows: int = 500_000,
) -> int:
    """
    扫一遍源数据：
    - 统计每个单字符的 doc frequency
    - 统计 total_docs（normalize 后非空描述总条数）

    返回 total_docs。
    """
    print(f"building unigram docfreq from {source_path}")

    counter: Counter[str] = Counter()
    total_docs = 0

    total_rows = (
        pl.scan_parquet(source_path)
        .select(pl.len())
        .collect()
        .item()
    )

    for start in range(0, total_rows, chunk_rows):
        df = (
            pl.scan_parquet(source_path)
            .slice(start, chunk_rows)
            .select(DESC_COL)
            .collect()
        )

        for raw in df[DESC_COL].to_list():
            text = normalize_desc(raw)
            if not text:
                continue
            total_docs += 1
            for ch in set(text):
                counter[ch] += 1

        print(f"  unigram scan: {min(start + chunk_rows, total_rows):,}/{total_rows:,}")

    out_df = pl.DataFrame(
        {
            "candidate": list(counter.keys()),
            "doc_count": list(counter.values()),
        },
        schema={"candidate": pl.String, "doc_count": pl.Int64},
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)

    Path(total_docs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(total_docs_path).write_text(str(total_docs))

    print(f"unigram docfreq written: {output_path} (unique chars={len(counter):,})")
    print(f"total_docs={total_docs:,} written to: {total_docs_path}")

    return total_docs


def load_total_docs(total_docs_path: str) -> int:
    return int(Path(total_docs_path).read_text().strip())


# ============================================================
# === NEW === Subgram freq map（凝固度查表）
# ============================================================

def build_subgram_freq_map(
    final_reduce_file: Path,
    unigram_freq_path: str,
) -> dict[str, int]:
    """
    构建凝固度查表用的 {ngram -> doc_count} 字典。
    数据来源：
      - 1gram：unigram_freq_path
      - 2-MAX_NGRAM：final_reduce_file（未施加 MIN_FREQ 的全量 reduce 结果）
    """
    print("building subgram freq map for cohesion lookup")
    freq_map: dict[str, int] = {}

    uni_df = pl.read_parquet(unigram_freq_path)
    for row in uni_df.iter_rows(named=True):
        freq_map[row["candidate"]] = int(row["doc_count"])
    print(f"  unigram entries: {len(freq_map):,}")

    multi_df = pl.read_parquet(str(final_reduce_file))
    for row in multi_df.iter_rows(named=True):
        freq_map[row["candidate"]] = int(row["freq"])
    print(f"  total subgram freq map size: {len(freq_map):,}")

    return freq_map


# ============================================================
# === NEW === Cohesion (min log2 PMI over all binary splits)
# ============================================================

def compute_cohesion(
    ngram: str,
    ngram_doc_count: int | None,
    freq_map: dict[str, int],
    total_docs: int,
) -> tuple[float, str]:
    """
    凝固度 = min over all binary splits of  log2( P(W) / (P(L) * P(R)) )

    返回 (cohesion, splits_debug_str)
    splits_debug_str 形如 "玻|尿酸=4.30; 玻尿|酸=1.40"
    """
    L = len(ngram)
    if L < 2:
        return float("inf"), ""

    if ngram_doc_count is None or ngram_doc_count <= 0 or total_docs <= 0:
        return float("-inf"), ""

    p_w = ngram_doc_count / total_docs
    if p_w <= 0:
        return float("-inf"), ""

    min_pmi = float("inf")
    debug_parts: list[str] = []

    for i in range(1, L):
        left, right = ngram[:i], ngram[i:]
        # 平滑：未见子串记为 1，避免 log(0)
        l_cnt = freq_map.get(left, 1)
        r_cnt = freq_map.get(right, 1)
        p_l = l_cnt / total_docs
        p_r = r_cnt / total_docs
        denom = p_l * p_r
        if denom <= 0:
            pmi = float("inf")
        else:
            pmi = math.log2(p_w / denom)
        debug_parts.append(f"{left}|{right}={pmi:.2f}")
        if pmi < min_pmi:
            min_pmi = pmi

    return min_pmi, "; ".join(debug_parts)


# ============================================================
# === NEW === Containment relations
# ============================================================

def build_containment_relations(
    ngrams: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    在【通过所有过滤】的 candidate 集合内判定包含关系。

    parents_map[w]  = 包含 w 的更长 ngram 列表
    children_map[w] = w 包含的更短 ngram 列表
    """
    ngram_set = set(ngrams)
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)

    for w in ngrams:
        L = len(w)
        for sub_len in range(MIN_NGRAM, L):
            for i in range(L - sub_len + 1):
                sub = w[i:i + sub_len]
                if sub == w:
                    continue
                if sub in ngram_set:
                    parents[sub].add(w)
                    children[w].add(sub)

    parents_sorted = {k: sorted(v) for k, v in parents.items()}
    children_sorted = {k: sorted(v) for k, v in children.items()}
    return parents_sorted, children_sorted


# ============================================================
# === NEW === Final trend outputs: 3 files
# ============================================================

# 关键列顺序：ngram, growth_rate, base_count, current_count 在最前
BASE_COLS = [
    "ngram",
    "growth_rate",
    "base_count",
    "current_count",
    "ngram_len",
    "total_count",
    "cohesion",
    "left_diversity",
    "right_diversity",
]

CONTAINED_COLS = BASE_COLS + ["parents"]
CONTAINER_COLS = BASE_COLS + ["children", "parents"]
LOW_COHESION_COLS = BASE_COLS + ["cohesion_splits_debug"]


def _write_output(
    df_out: pl.DataFrame,
    cols: list[str],
    output_dir: Path,
    basename: str,
    sort_col: str,
    descending: bool,
) -> None:
    existing = [c for c in cols if c in df_out.columns]
    df_out = df_out.select(existing)

    if sort_col in df_out.columns:
        df_out = df_out.sort(sort_col, descending=descending, nulls_last=True)

    round_cols = [c for c in ("growth_rate", "cohesion") if c in df_out.columns]
    if round_cols:
        df_out = df_out.with_columns([
            pl.col(c).round(3) for c in round_cols
        ])

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{basename}.csv"
    parquet_path = output_dir / f"{basename}.parquet"

    df_out.write_csv(csv_path, include_bom=True)
    df_out.write_parquet(parquet_path)

    print(f"  wrote {csv_path.name}  ({df_out.height:,} rows)")
    print(f"  wrote {parquet_path.name}  ({df_out.height:,} rows)")


def build_final_trend_outputs(
    context_df: pl.DataFrame,
    candidate_freq: pl.DataFrame,
    final_reduce_file: Path,
    unigram_freq_path: str,
    total_docs: int,
    output_dir: str,
) -> None:
    """
    最终输出三份文件：

      文件1  trend_contained.csv
          通过所有过滤的 candidate，且【自己没有包含任何其他通过项】
          → 含两类：被别人包含但自己没包含别人；以及孤立词

      文件2  trend_container.csv
          通过所有过滤的 candidate，且【自己包含了至少一个其他通过项】
          → 可能同时也被更长的通过项包含

      文件3  trend_low_cohesion.csv
          通过其他所有过滤、但被凝固度毙掉的 candidate

    包含关系判定仅在【通过所有过滤】的集合内做。
    """
    print("=" * 100)
    print("building 3 final trend outputs")
    print("=" * 100)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. join total_count (全量 doc count) from candidate_freq ----
    total_count_df = candidate_freq.select([
        pl.col("candidate"),
        pl.col("freq").alias("total_count"),
    ])
    df = context_df.join(total_count_df, on="candidate", how="left")
    df = df.with_columns(
        pl.col("total_count").fill_null(0).cast(pl.Int64)
    )

    # ---- 2. base_count 过滤 ----
    before = df.height
    df = df.filter(pl.col("base_count") >= MIN_BASE_COUNT)
    print(f"after MIN_BASE_COUNT>={MIN_BASE_COUNT}: {df.height:,} (from {before:,})")

    # ---- 3. 噪音过滤 ----
    before = df.height
    df = df.with_columns(
        pl.col("candidate").map_elements(
            is_noise_candidate, return_dtype=pl.Boolean
        ).alias("_is_noise")
    ).filter(~pl.col("_is_noise")).drop("_is_noise")
    print(f"after noise filter: {df.height:,} (from {before:,})")

    # ---- 4. context diversity 过滤 ----
    before = df.height
    df = df.with_columns(
        pl.struct(["left_chars", "right_chars"]).map_elements(
            lambda r: context_diversity_pass(r["left_chars"], r["right_chars"]),
            return_dtype=pl.Boolean,
        ).alias("_div_pass")
    ).filter(pl.col("_div_pass")).drop("_div_pass")
    print(f"after context diversity (>= {MIN_CONTEXT_DIVERSITY}): {df.height:,} (from {before:,})")

    # ---- 5. 计算 left/right diversity 数值列 + growth_rate ----
    # === FIXED === 用 polars 原生 list 表达式，避免 udf 触发 Series 真值歧义
    df = df.with_columns([
        pl.col("left_chars")
          .list.set_difference([BOUNDARY_CHAR])
          .list.len()
          .cast(pl.Int64)
          .alias("left_diversity"),
        pl.col("right_chars")
          .list.set_difference([BOUNDARY_CHAR])
          .list.len()
          .cast(pl.Int64)
          .alias("right_diversity"),
        pl.struct(["base_count", "current_count"]).map_elements(
            lambda r: calculate_growth_rate(r["base_count"], r["current_count"]),
            return_dtype=pl.Float64,
        ).alias("growth_rate"),
    ])

    # ---- 6. 凝固度 ----
    if ENABLE_COHESION_FILTER:
        print("computing cohesion (log2 PMI, min over all binary splits)...")
        freq_map = build_subgram_freq_map(final_reduce_file, unigram_freq_path)

        cohesion_vals: list[float] = []
        cohesion_debug: list[str] = []
        for row in df.iter_rows(named=True):
            cv, dbg = compute_cohesion(
                row["candidate"],
                int(row["total_count"]),
                freq_map,
                total_docs,
            )
            cohesion_vals.append(cv)
            cohesion_debug.append(dbg)

        df = df.with_columns([
            pl.Series("cohesion", cohesion_vals, dtype=pl.Float64),
            pl.Series("cohesion_splits_debug", cohesion_debug, dtype=pl.Utf8),
        ])

        low_cohesion_df = df.filter(pl.col("cohesion") < MIN_COHESION)
        passed_df = df.filter(pl.col("cohesion") >= MIN_COHESION)
    else:
        df = df.with_columns([
            pl.lit(float("inf"), dtype=pl.Float64).alias("cohesion"),
            pl.lit("", dtype=pl.Utf8).alias("cohesion_splits_debug"),
        ])
        low_cohesion_df = df.head(0)
        passed_df = df

    print(f"low cohesion (< {MIN_COHESION}): {low_cohesion_df.height:,}")
    print(f"passed all filters: {passed_df.height:,}")

    # ---- 7. 包含关系（仅在 passed_df 上判定）----
    print("computing containment relations on passed candidates...")
    passed_ngrams = passed_df["candidate"].to_list()
    parents_map, children_map = build_containment_relations(passed_ngrams)

    parents_col = ["; ".join(parents_map.get(w, [])) for w in passed_ngrams]
    children_col = ["; ".join(children_map.get(w, [])) for w in passed_ngrams]
    has_children = [bool(children_map.get(w)) for w in passed_ngrams]

    passed_df = passed_df.with_columns([
        pl.Series("parents", parents_col, dtype=pl.Utf8),
        pl.Series("children", children_col, dtype=pl.Utf8),
        pl.Series("_has_children", has_children, dtype=pl.Boolean),
    ])

    # ---- 8. 拆 container / contained ----
    # has_children=True  → 文件2 container
    # has_children=False → 文件1 contained
    container_df = passed_df.filter(pl.col("_has_children")).drop("_has_children")
    contained_df = (
        passed_df
        .filter(~pl.col("_has_children"))
        .drop(["_has_children", "children"])
    )

    print(f"container (文件2 包含别人的): {container_df.height:,}")
    print(f"contained (文件1 没包含别人的): {contained_df.height:,}")

    # ---- 9. 重命名 candidate -> ngram 后写出 ----
    container_df = container_df.rename({"candidate": "ngram"})
    contained_df = contained_df.rename({"candidate": "ngram"})
    low_cohesion_df = low_cohesion_df.rename({"candidate": "ngram"})

    print(f"writing 3 outputs to {out_dir} ...")

    _write_output(
        contained_df,
        CONTAINED_COLS,
        out_dir,
        TRIPLE_CONTAINED_NAME,
        sort_col="growth_rate",
        descending=True,
    )
    _write_output(
        container_df,
        CONTAINER_COLS,
        out_dir,
        TRIPLE_CONTAINER_NAME,
        sort_col="growth_rate",
        descending=True,
    )
    _write_output(
        low_cohesion_df,
        LOW_COHESION_COLS,
        out_dir,
        TRIPLE_LOW_COHESION_NAME,
        sort_col="growth_rate",
        descending=True,
    )

    print("=" * 100)
    print("final trend outputs done")
    print("=" * 100)


# ============================================================
# Main
# ============================================================

def run_trend_pipeline() -> None:
    print_config()

    source_path = Path(SOURCE_PATH)
    if not source_path.exists():
        raise FileNotFoundError(f"source parquet not found: {SOURCE_PATH}")

    # --------------------------------------------------------
    # Step 1: 高频 candidate docfreq，全量数据
    # --------------------------------------------------------

    if FORCE_REBUILD_CACHE and Path(CACHE_DIR_PATH).exists():
        print(f"FORCE_REBUILD_CACHE=True, deleting cache: {CACHE_DIR_PATH}")
        recreate_dir(CACHE_DIR_PATH)

    if not cache_is_valid(CACHE_DIR_PATH):
        print("candidate docfreq cache not found, building it once...")
        build_candidate_cache(
            source_path=SOURCE_PATH,
            cache_dir_path=CACHE_DIR_PATH,
            chunk_rows=CANDIDATE_CHUNK_ROWS,
            workers=CANDIDATE_WORKERS,
        )
    else:
        print(f"using cached candidate docfreq -> {CACHE_DIR_PATH}")

    if FORCE_REBUILD_CACHE or not Path(FINAL_CANDIDATE_PATH).exists():
        final_reduce_file = hierarchical_reduce(
            input_dir_path=CACHE_DIR_PATH,
            output_root_path=CACHE_REDUCE_DIR_PATH,
            batch_files=REDUCE_BATCH_FILES,
            apply_min_freq_filter_in_middle=False,
        )
        candidate_freq = build_final_candidate_freq(
            final_reduce_file=final_reduce_file,
            final_output_path=FINAL_CANDIDATE_PATH,
        )
    else:
        print(f"using final candidate file -> {FINAL_CANDIDATE_PATH}")
        candidate_freq = pl.read_parquet(FINAL_CANDIDATE_PATH)

        final_reduce_file = find_final_reduce_file(CACHE_REDUCE_DIR_PATH)
        if ENABLE_COHESION_FILTER and final_reduce_file is None:
            print("final reduce file not found, rerunning hierarchical_reduce for cohesion lookup...")
            final_reduce_file = hierarchical_reduce(
                input_dir_path=CACHE_DIR_PATH,
                output_root_path=CACHE_REDUCE_DIR_PATH,
                batch_files=REDUCE_BATCH_FILES,
                apply_min_freq_filter_in_middle=False,
            )
        else:
            print(f"using existing final reduce file -> {final_reduce_file}")

    print("top 20 high frequency candidates:")
    print(candidate_freq.head(20))

    # --------------------------------------------------------
    # === NEW === Step 1b: unigram docfreq + total_docs（凝固度查表用）
    # --------------------------------------------------------

    if ENABLE_COHESION_FILTER:
        need_rebuild_unigram = (
            FORCE_REBUILD_CACHE
            or not Path(UNIGRAM_FREQ_PATH).exists()
            or not Path(TOTAL_DOCS_PATH).exists()
        )
        if need_rebuild_unigram:
            total_docs = build_unigram_docfreq_and_count(
                source_path=SOURCE_PATH,
                output_path=UNIGRAM_FREQ_PATH,
                total_docs_path=TOTAL_DOCS_PATH,
            )
        else:
            total_docs = load_total_docs(TOTAL_DOCS_PATH)
            print(f"using cached unigram docfreq -> {UNIGRAM_FREQ_PATH}")
            print(f"using cached total_docs={total_docs:,}")
    else:
        total_docs = 0

    # --------------------------------------------------------
    # Step 2: 高频 candidate 的上下文和两个季度 count
    # --------------------------------------------------------

    context_dir = Path(CONTEXT_PART_DIR_PATH)
    context_exists = (
        context_dir.exists()
        and len(list(context_dir.glob("context_part_*.parquet"))) > 0
    )

    if FORCE_REBUILD_CONTEXT or not context_exists:
        print("building context parts with all-period context scope...")
        build_context_parts(
            source_path=SOURCE_PATH,
            context_dir_path=CONTEXT_PART_DIR_PATH,
            candidate_path=FINAL_CANDIDATE_PATH,
            chunk_rows=CONTEXT_CHUNK_ROWS,
            workers=CONTEXT_WORKERS,
        )
    else:
        print(f"using cached context parts -> {CONTEXT_PART_DIR_PATH}")

    # --------------------------------------------------------
    # Step 3: 汇总 context，过滤残缺词和噪音词，计算趋势，
    #         凝固度过滤，并输出三份最终文件
    # --------------------------------------------------------

    context_df = reduce_context_parts(CONTEXT_PART_DIR_PATH)

    build_final_trend_outputs(
        context_df=context_df,
        candidate_freq=candidate_freq,
        final_reduce_file=final_reduce_file,
        unigram_freq_path=UNIGRAM_FREQ_PATH,
        total_docs=total_docs,
        output_dir=TRIPLE_OUTPUT_DIR,
    )


def main() -> None:
    CONTEXT.ensure_directories()
    create_manifest(CONTEXT)
    if not Path(SOURCE_PATH).exists():
        update_stage(CONTEXT, "trend", "failed")
        raise FileNotFoundError(
            f"未找到当前品类的 ETL 文件：{SOURCE_PATH}\n"
            f"请先运行：uv run etl_test.py --category {CONTEXT.category_code}"
        )
    update_stage(CONTEXT, "trend", "running")
    try:
        run_trend_pipeline()
        artifacts = {
            "trend_contained_parquet": CONTEXT.trend_dir / f"{TRIPLE_CONTAINED_NAME}.parquet",
            "trend_container_parquet": CONTEXT.trend_dir / f"{TRIPLE_CONTAINER_NAME}.parquet",
            "trend_low_cohesion_parquet": CONTEXT.trend_dir / f"{TRIPLE_LOW_COHESION_NAME}.parquet",
            "final_candidate_docfreq": Path(FINAL_CANDIDATE_PATH),
            "unigram_docfreq": Path(UNIGRAM_FREQ_PATH),
            "total_docs": Path(TOTAL_DOCS_PATH),
        }
        update_stage(CONTEXT, "trend", "completed", artifacts=artifacts)
        set_active_run(CONTEXT)
        print(f"[OK] trend stage completed: {CONTEXT.trend_dir}")
    except Exception:
        update_stage(CONTEXT, "trend", "failed")
        raise


if __name__ == "__main__":
    main()
