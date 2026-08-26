# src/data_contract.py
from pydantic import BaseModel, Field
from typing import List

class ProductRow(BaseModel):
    id: int
    clean_title: str
    cat_id: int
    probability: float
    layer: str

class NGramRecord(BaseModel):
    term: str
    cat_id: int
    df: int
    left_entropy: float = 0.0
    right_entropy: float = 0.0
    attach_ratio: float = 0.0

class HotWordResult(BaseModel):
    cat_id: int
    term: str
    df_old: int
    df_new: int
    growth: float
    neighbor_entropy: float
    attach_ratio: float
    evidence_samples: List[str] = []

class HotWordsReport(BaseModel):
    items: List[HotWordResult]