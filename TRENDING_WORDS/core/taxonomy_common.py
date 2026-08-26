from __future__ import annotations
import json, math, re, unicodedata
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

# These are not product attributes for this opportunity pipeline.
EXCLUDED_DIMENSION_CODES = {"CATEGORY", "BRAND", "SUBBRAND"}
# They remain known attributes, but cannot create a direct new-label opportunity from vague market text.
FACT_BASED_ATTRIBUTE_CODES = {"PACKSIZE", "SHELF_LIFE"}
NON_OPPORTUNITY_ATTRIBUTE_CODES = {"PACKSIZE", "SHELF_LIFE"}

PLACEHOLDER_VALUES = {
    "其他", "其它", "其他类", "其它类", "其他类别", "其它类别", "其他品牌", "其它品牌", "其他牌子",
    "不知道", "不清楚", "不确定", "未知", "未知项", "不详", "暂无", "无资料", "无信息",
    "未注明", "没注明", "未标注", "未说明", "未提供", "无注明", "无标注",
    "不适用", "非适用", "缺失", "待定", "待确认", "unknown", "unk", "other", "others",
    "n/a", "na", "none", "null", "not applicable", "notapplicable",
}
BRAND_FALLBACK_PATTERNS = (
    re.compile(r"^(?:zz2na[_\-]*)?(?:其他|其它)(?:品牌|牌子)?$", re.I),
    re.compile(r"^.+(?:其它|其他)$", re.I),
)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip())


def normalize_key(value: Any) -> str:
    return re.sub(r"[\s_\-—–/\\（）()【】\[\]{}·.,，。:：;；]+", "", clean_text(value).casefold())

PLACEHOLDER_KEYS = {normalize_key(x) for x in PLACEHOLDER_VALUES}


def label_rejection_reason(code: str, name: str, label: str) -> str | None:
    code, name, label = clean_text(code), clean_text(name), clean_text(label)
    if not label:
        return "blank"
    key = normalize_key(label)
    if key in PLACEHOLDER_KEYS:
        return "generic_placeholder"
    if code in {"BRAND", "SUBBRAND"} and any(p.fullmatch(label) for p in BRAND_FALLBACK_PATTERNS):
        return "brand_fallback_bucket"
    # PACKSIZE | 规格 | PACKSIZE is a source placeholder, not a real label.
    if code == "PACKSIZE" and key in {normalize_key(code), normalize_key(name)}:
        return "self_reference_placeholder"
    return None


def load_taxonomy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine="openpyxl")
    required = {"SEGTYPE", "SEGDESC", "CSEGMENT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Taxonomy 缺少字段: {sorted(missing)}")
    out = df[["SEGTYPE", "SEGDESC", "CSEGMENT"]].copy()
    out.columns = ["attribute_code", "attribute_name", "label"]
    for col in out.columns:
        out[col] = out[col].map(clean_text)
    out = out[(out.attribute_code != "") & (out.attribute_name != "")]
    out["label_rejection_reason"] = out.apply(
        lambda r: label_rejection_reason(r.attribute_code, r.attribute_name, r.label), axis=1
    )
    out["is_valid_label"] = out.label_rejection_reason.isna()
    return out.drop_duplicates(["attribute_code", "attribute_name", "label"]).reset_index(drop=True)


def valid_labels_by_code(df: pd.DataFrame) -> dict[str, list[str]]:
    valid = df[df.is_valid_label]
    return {
        str(code): sorted(set(group.label.tolist()))
        for code, group in valid.groupby("attribute_code", sort=False)
    }


def attribute_directory(df: pd.DataFrame) -> list[dict[str, str]]:
    rows = df.loc[~df.attribute_code.isin(EXCLUDED_DIMENSION_CODES), ["attribute_code", "attribute_name"]]
    return rows.drop_duplicates().sort_values(["attribute_code", "attribute_name"]).to_dict("records")


def category_context(df: pd.DataFrame) -> list[dict[str, str]]:
    rows = df.loc[df.attribute_code.eq("CATEGORY"), ["attribute_code", "attribute_name", "label"]]
    rows = rows[rows.isna().sum(axis=1).eq(0)]
    return rows.drop_duplicates().to_dict("records")


def normalize_range_label(label: str) -> str:
    text = clean_text(label)
    text = re.sub(r">\s*(\d+)\s*-\s*<=\s*(\d+)\s*天", r"大于\1天且小于等于\2天", text)
    text = re.sub(r">=\s*(\d+)\s*天", r"大于等于\1天", text)
    text = re.sub(r"(\d+)\s*天\s*-\s*(\d+)\s*天", r"\1天至\2天", text)
    return text


def encode_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(model_name, use_fp16=False)
        vecs = model.encode(texts, batch_size=batch_size, max_length=256)["dense_vecs"]
    except ImportError:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def json_load(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
