"""Query-key based JSONL cache for Cluster and Noise Term LLM results."""
from __future__ import annotations

import hashlib, json, math, os, threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CACHE_RECORD_VERSION = "analysis_cache_record_v2"
VALID_PHASES = {"internal", "external"}
VALID_REVIEW_STATUSES = {"PENDING", "APPROVED", "REVISED", "REJECTED"}
VALID_RECORD_STATUSES = {"ACTIVE", "INVALIDATED"}

@dataclass(slots=True)
class CacheLookup:
    hit: bool
    phase: str
    query_key: str
    signature: str
    record: dict[str, Any] | None = None
    reason: str = ""
    @property
    def parsed_result(self):
        return self.record.get("parsed_result") if self.record else None

@dataclass(slots=True)
class CacheLoadReport:
    path: str; total_lines: int=0; valid_records: int=0
    skipped_blank_lines: int=0; invalid_json_lines: int=0
    non_object_lines: int=0; invalid_record_lines: int=0
    warnings: list[str] | None=None
    def __post_init__(self): self.warnings = self.warnings or []

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)

def build_internal_signature(*, evidence, model, prompt_version, schema_version, system_prompt=None):
    return _sha256(canonical_json({"phase":"internal","model":model,"prompt_version":prompt_version,"schema_version":schema_version,"system_prompt_hash":_optional_text_hash(system_prompt),"evidence":evidence}))

def build_external_signature(*, evidence, internal_insight, model, prompt_version, schema_version, tools, system_prompt=None):
    return _sha256(canonical_json({"phase":"external","model":model,"prompt_version":prompt_version,"schema_version":schema_version,"system_prompt_hash":_optional_text_hash(system_prompt),"tools":tools or [],"evidence":evidence,"internal_insight":internal_insight}))

def build_evidence_hash(evidence): return _sha256(canonical_json(evidence))
def utc_now_iso(): return datetime.now(timezone.utc).isoformat(timespec="microseconds")

class ClusterCache:
    def __init__(self, cache_path: str|Path, error_path: str|Path|None=None):
        self.cache_path=Path(cache_path);self.error_path=Path(error_path) if error_path else self.cache_path.with_name(self.cache_path.stem+"_errors.jsonl");self._lock=threading.RLock();self.records=[];self.load_report=CacheLoadReport(str(self.cache_path));self.reload()
    def reload(self):
        self.records,self.load_report=load_jsonl_records(self.cache_path);return self.records
    def lookup_by_query_key(self, *, phase, query_key, signature):
        _validate_phase(phase);_require_nonempty(query_key,"query_key");_require_nonempty(signature,"signature")
        matches=[r for r in self.records if r.get("phase")==phase and _record_query_key(r)==query_key and r.get("signature")==signature and r.get("schema_valid") is True]
        if not matches:return CacheLookup(False,phase,query_key,signature,None,"miss")
        record=max(matches,key=_record_timestamp)
        if record.get("record_status","ACTIVE")!="ACTIVE":return CacheLookup(False,phase,query_key,signature,record,"invalidated")
        return CacheLookup(True,phase,query_key,signature,record,"hit")
    def lookup(self, *, phase, signature, query_key=None, cluster_id=None):
        key=query_key or f"cluster:{_validate_cluster_id(cluster_id)}";return self.lookup_by_query_key(phase=phase,query_key=key,signature=signature)
    def latest_for_query_key(self, *, phase, query_key, active_only=False):
        rows=[r for r in self.records if r.get("phase")==phase and _record_query_key(r)==query_key and (not active_only or r.get("record_status","ACTIVE")=="ACTIVE")]
        return max(rows,key=_record_timestamp) if rows else None
    def latest_for_cluster(self, *, phase, cluster_id): return self.latest_for_query_key(phase=phase,query_key=f"cluster:{_validate_cluster_id(cluster_id)}")
    def list_records(self, *, phase=None, schema_valid_only=False, active_only=True):
        latest={}
        for r in self.records:
            if phase and r.get("phase")!=phase:continue
            if schema_valid_only and r.get("schema_valid") is not True:continue
            key=(r.get("phase"),_record_query_key(r),r.get("signature"))
            if key not in latest or _record_timestamp(r)>=_record_timestamp(latest[key]):latest[key]=r
        rows=list(latest.values())
        if active_only:rows=[r for r in rows if r.get("record_status","ACTIVE")=="ACTIVE"]
        return sorted(rows,key=_record_timestamp)
    def put_success(self, *, phase, query_key=None, cluster_id=None, signature, parsed_result, evidence_hash="", analysis_unit="cluster", raw_output_text="", model="", prompt_version="", schema_version="", validation_warnings=None, api_result=None, internal_signature=None, **metadata):
        key=query_key or f"cluster:{_validate_cluster_id(cluster_id)}";cid=int(cluster_id if cluster_id is not None else metadata.get("cluster_id",-1));now=utc_now_iso();api=api_result or {}
        record={"record_type":"success","record_version":CACHE_RECORD_VERSION,"phase":phase,"analysis_unit":analysis_unit,"query_key":key,"cluster_id":cid,"signature":signature,"evidence_hash":evidence_hash,"schema_valid":True,"parsed_result":parsed_result,"raw_output_text":raw_output_text,"model":model,"prompt_version":prompt_version,"schema_version":schema_version,"validation_errors":[],"validation_warnings":validation_warnings or [],"record_status":"ACTIVE","review_status":"PENDING","reviewer":"","review_comment":"","internal_signature":internal_signature,"created_at_utc":now,"updated_at_utc":now}
        for field in ("input_tokens","output_tokens","total_tokens","elapsed_seconds","thinking_text","degraded","tools"):
            record[field]=api.get(field,metadata.get(field,api.get("usage",{}).get(field,0) if field.endswith("tokens") else None))
        self._append(record);return record
    def put_error(self, *, phase, query_key=None, cluster_id=None, signature="", error_code="ERROR", error_message="", analysis_unit="cluster", raw_output_text="", validation_errors=None, api_error=None, **metadata):
        key=query_key or f"cluster:{_validate_cluster_id(cluster_id)}";now=utc_now_iso();record={"record_type":"error","record_version":CACHE_RECORD_VERSION,"phase":phase,"analysis_unit":analysis_unit,"query_key":key,"cluster_id":int(cluster_id if cluster_id is not None else -1),"signature":signature,"schema_valid":False,"parsed_result":None,"raw_output_text":raw_output_text,"error_code":error_code,"error_message":error_message,"validation_errors":validation_errors or [],"api_error":api_error,"record_status":"ACTIVE","created_at_utc":now,"updated_at_utc":now,**metadata};_append_jsonl_line(self.error_path,record);return record
    def invalidate_query_key(self, *, phase, query_key, reason=""):
        return self._status_event(phase,query_key,"INVALIDATED",reason)
    def restore_query_key(self, *, phase, query_key, reason=""):
        return self._status_event(phase,query_key,"ACTIVE",reason)
    def _status_event(self,phase,key,status,reason):
        base=self.latest_for_query_key(phase=phase,query_key=key)
        if not base:raise KeyError(key)
        event=dict(base);event.update(record_type="status",record_version=CACHE_RECORD_VERSION,record_status=status,status_reason=reason,updated_at_utc=utc_now_iso());self._append(event);return event
    def _append(self,record):
        with self._lock:_append_jsonl_line(self.cache_path,record);self.records.append(record)

def load_jsonl_records(path):
    path=Path(path);report=CacheLoadReport(str(path));records=[]
    if not path.exists():return records,report
    with path.open("r",encoding="utf-8-sig") as h:
        for n,line in enumerate(h,1):
            report.total_lines+=1
            if not line.strip():report.skipped_blank_lines+=1;continue
            try:value=json.loads(line)
            except json.JSONDecodeError:report.invalid_json_lines+=1;report.warnings.append(f"line {n}: invalid json");continue
            if not isinstance(value,dict):report.non_object_lines+=1;continue
            if not _is_valid_cache_record(value):report.invalid_record_lines+=1;continue
            value.setdefault("query_key",f"cluster:{value['cluster_id']}");value.setdefault("analysis_unit","cluster");value.setdefault("record_status","ACTIVE");records.append(value);report.valid_records+=1
    return records,report

def append_error_event(error_path,event):
    enriched=dict(event);enriched.setdefault("created_at_utc",utc_now_iso());_append_jsonl_line(Path(error_path),enriched)
def _append_jsonl_line(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="\n") as h:h.write(canonical_json(value)+"\n");h.flush();os.fsync(h.fileno())
def _is_valid_cache_record(r): return r.get("phase") in VALID_PHASES and isinstance(r.get("cluster_id"),int) and isinstance(r.get("signature"),str) and "schema_valid" in r and "updated_at_utc" in r
def _record_query_key(r): return str(r.get("query_key") or f"cluster:{r.get('cluster_id')}")
def _record_timestamp(r): return str(r.get("updated_at_utc") or r.get("created_at_utc") or "")
def _sha256(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()
def _optional_text_hash(v): return None if v is None else _sha256(v)
def _json_default(v):
    if isinstance(v,Path):return str(v)
    if hasattr(v,"item"):
        x=v.item()
        if isinstance(x,float) and not math.isfinite(x):raise ValueError("NaN/Infinity")
        return x
    if hasattr(v,"to_dict"):return v.to_dict()
    raise TypeError(type(v).__name__)
def _validate_phase(v):
    if v not in VALID_PHASES:raise ValueError(v)
    return v
def _validate_cluster_id(v):
    if not isinstance(v,int) or isinstance(v,bool):raise TypeError("cluster_id必须为int")
    return v
def _require_nonempty(v,name):
    if not isinstance(v,str) or not v.strip():raise ValueError(f"{name}不能为空")
