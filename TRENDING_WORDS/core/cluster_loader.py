"""Load Analysis Evidence and route Cluster/Noise internal LLM tasks."""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
from core.analysis_unit import ANALYSIS_UNIT_CLUSTER,ANALYSIS_UNIT_NOISE_TERM,VALID_ANALYSIS_UNITS,analysis_unit_display_name
from core.cluster_cache import ClusterCache,build_evidence_hash,build_external_signature,build_internal_signature,canonical_json
from core.cluster_prompt import EXTERNAL_SYSTEM_PROMPT,INTERNAL_SYSTEM_PROMPT,build_external_user_prompt,build_internal_user_prompt
from core.noise_term_prompt import NOISE_TERM_SYSTEM_PROMPT,build_noise_term_user_prompt
from core.config import EXTERNAL_PROMPT_VERSION,EXTERNAL_SCHEMA_VERSION,EXTERNAL_TOOLS,INTERNAL_CLUSTER_PROMPT_VERSION,INTERNAL_CLUSTER_SCHEMA_VERSION,INTERNAL_NOISE_PROMPT_VERSION,INTERNAL_NOISE_SCHEMA_VERSION

@dataclass(slots=True)
class EvidenceIssue:
    severity:str;code:str;message:str;cluster_id:int|None=None;query_key:str|None=None;line_number:int|None=None
    def to_dict(self):return asdict(self)
@dataclass(slots=True)
class EvidenceLoadReport:
    path:str;source_format:str;total_records_read:int;valid_records:int;invalid_records:int;duplicate_cluster_ids:list[int];duplicate_query_keys:list[str];issues:list[EvidenceIssue]
    @property
    def has_errors(self):return any(x.severity=='error' for x in self.issues)
    @property
    def error_count(self):return sum(x.severity=='error' for x in self.issues)
    @property
    def warning_count(self):return sum(x.severity=='warning' for x in self.issues)
    def to_dict(self):d=asdict(self);d.update(has_errors=self.has_errors,error_count=self.error_count,warning_count=self.warning_count);return d
@dataclass(slots=True)
class ClusterEvidence:
    cluster_id:int;query_key:str;evidence:dict[str,Any];evidence_hash:str;analysis_unit:str=ANALYSIS_UNIT_CLUSTER;source_cluster_id:int|None=None;term:str|None=None
    @property
    def cluster_metrics(self):return self.evidence.get('cluster_metrics',{})
    @property
    def representative_terms(self):return self.evidence.get('representative_terms',[])
    @property
    def taxonomy_candidates(self):return self.evidence.get('taxonomy_candidates',[])
    @property
    def representative_term_names(self):return [x.get('ngram') for x in self.representative_terms if isinstance(x,dict) and isinstance(x.get('ngram'),str)]
    @property
    def display_name(self):return analysis_unit_display_name(analysis_unit=self.analysis_unit,cluster_id=self.cluster_id,ngram=self.term)
@dataclass(slots=True)
class PreparedTask:
    phase:str;cluster_id:int;query_key:str;signature:str;evidence_hash:str;user_input:str;evidence:dict[str,Any];analysis_unit:str=ANALYSIS_UNIT_CLUSTER;prompt_version:str='';schema_version:str='';system_prompt:str='';internal_insight:dict[str,Any]|None=None;internal_signature:str|None=None;cache_status:str='pending'
    def to_api_item(self):return {'cluster_id':self.cluster_id,'query_key':self.query_key,'signature':self.signature,'evidence_hash':self.evidence_hash,'analysis_unit':self.analysis_unit,'__user_input__':self.user_input}

def load_cluster_evidence(path,*,strict=True):
    path=Path(path);issues=[];raw=[]
    if not path.exists():raise FileNotFoundError(path)
    if path.suffix.lower()=='.jsonl':
        for n,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
            if not line.strip():continue
            try:raw.append((n,json.loads(line)))
            except json.JSONDecodeError as e:issues.append(EvidenceIssue('error','INVALID_JSON_LINE',e.msg,line_number=n))
        fmt='jsonl'
    elif path.suffix.lower()=='.json':
        values=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(values,list):raise ValueError('JSON根节点必须是数组')
        raw=list(enumerate(values,1));fmt='json'
    else:raise ValueError('仅支持.jsonl/.json')
    records=[];seen_ids=set();seen_keys=set();dup_ids=set();dup_keys=set()
    for n,obj in raw:
        if isinstance(obj,dict) and 'analysis_unit' not in obj:
            obj=dict(obj);obj['analysis_unit']=ANALYSIS_UNIT_CLUSTER;obj.setdefault('source_cluster_id',obj.get('cluster_id'));obj.setdefault('term',None)
        local=validate_evidence(obj,line_number=n);issues.extend(local)
        if any(x.severity=='error' for x in local):continue
        cid=int(obj['cluster_id']);key=str(obj['query_key']);unit=str(obj['analysis_unit'])
        if key in seen_keys:dup_keys.add(key);issues.append(EvidenceIssue('error','DUPLICATE_QUERY_KEY',key,cid,key,n));continue
        if unit==ANALYSIS_UNIT_CLUSTER and cid in seen_ids:dup_ids.add(cid);issues.append(EvidenceIssue('error','DUPLICATE_CLUSTER_ID',str(cid),cid,key,n));continue
        seen_keys.add(key)
        if unit==ANALYSIS_UNIT_CLUSTER:seen_ids.add(cid)
        norm=json.loads(canonical_json(obj));records.append(ClusterEvidence(cid,key,norm,build_evidence_hash(norm),unit,int(obj['source_cluster_id']),obj.get('term')))
    report=EvidenceLoadReport(str(path),fmt,len(raw),len(records),len(raw)-len(records),sorted(dup_ids),sorted(dup_keys),issues)
    records.sort(key=lambda r:(r.analysis_unit,r.query_key))
    if strict and report.has_errors:raise ValueError('Evidence校验失败：'+' | '.join(f'{x.code}:{x.message}' for x in issues if x.severity=='error')[:1800])
    return records,report

def validate_evidence(evidence:Any,*,line_number=None):
    issues=[]
    if not isinstance(evidence,dict):return [EvidenceIssue('error','NOT_OBJECT','Evidence必须是对象',line_number=line_number)]
    cid=evidence.get('cluster_id');key=evidence.get('query_key');unit=evidence.get('analysis_unit');source=evidence.get('source_cluster_id')
    def add(code,msg):issues.append(EvidenceIssue('error',code,msg,cid if isinstance(cid,int) else None,key if isinstance(key,str) else None,line_number))
    required={'analysis_unit','query_key','cluster_id','source_cluster_id','cluster_metrics','representative_terms','attribute_diagnostics','all_existing_attributes','taxonomy_candidates','exact_matches','allowed_mapping_types','metric_note'}
    missing=required-set(evidence)
    if missing:add('MISSING_ROOT_KEYS',str(sorted(missing)))
    if unit not in VALID_ANALYSIS_UNITS:add('INVALID_ANALYSIS_UNIT',str(unit))
    if not isinstance(cid,int) or isinstance(cid,bool):add('INVALID_CLUSTER_ID','cluster_id必须为整数')
    if not isinstance(source,int) or isinstance(source,bool):add('INVALID_SOURCE_CLUSTER_ID','source_cluster_id必须为整数')
    if not isinstance(key,str) or not key.strip():add('INVALID_QUERY_KEY','query_key无效')
    if unit==ANALYSIS_UNIT_CLUSTER and (cid is not None and cid<0 or source!=cid):add('INVALID_CLUSTER_IDENTITY','稳定Cluster身份无效')
    if unit==ANALYSIS_UNIT_NOISE_TERM:
        if cid!=-1 or source!=-1:add('INVALID_NOISE_IDENTITY','Noise身份必须为-1')
        if not isinstance(evidence.get('term'),str) or not evidence.get('term','').strip():add('INVALID_NOISE_TERM','term不能为空')
        if not isinstance(evidence.get('term_evidence'),dict):add('MISSING_TERM_EVIDENCE','缺少term_evidence')
    directory=evidence.get('all_existing_attributes');candidates=evidence.get('taxonomy_candidates')
    if not isinstance(directory,list) or not directory:add('INVALID_ATTRIBUTE_DIRECTORY','属性目录不能为空')
    if not isinstance(candidates,list) or not candidates:add('INVALID_TAXONOMY_CANDIDATES','候选属性不能为空')
    return issues

def filter_evidence(records:Iterable[ClusterEvidence],*,cluster_ids=None,analysis_units=None,query_keys=None,min_cluster_size=None,min_growth=None,min_cluster_probability=None,max_low_cohesion_ratio=None,exact_match=None,max_top1_top2_margin=None):
    ids=set(cluster_ids) if cluster_ids is not None else None;units=set(analysis_units) if analysis_units is not None else None;keys=set(query_keys) if query_keys is not None else None;out=[]
    for r in records:
        m=r.cluster_metrics;d=r.evidence.get('attribute_diagnostics',{});e=r.evidence.get('exact_matches',{})
        if ids is not None and r.cluster_id not in ids:continue
        if units is not None and r.analysis_unit not in units:continue
        if keys is not None and r.query_key not in keys:continue
        if min_cluster_size is not None and _num(m.get('cluster_size'),-math.inf)<min_cluster_size:continue
        if min_growth is not None and _num(m.get('aggregate_ngram_docfreq_growth'),-math.inf)<min_growth:continue
        if min_cluster_probability is not None and _num(m.get('mean_cluster_probability'),-math.inf)<min_cluster_probability:continue
        if max_low_cohesion_ratio is not None and _num(m.get('representative_low_cohesion_ratio'),math.inf)>max_low_cohesion_ratio:continue
        if exact_match is not None and bool(e.get('any_candidate_has_exact_match',False))!=exact_match:continue
        if max_top1_top2_margin is not None and _num(d.get('top1_top2_margin'),math.inf)>max_top1_top2_margin:continue
        out.append(r)
    return sorted(out,key=lambda r:(r.analysis_unit,r.query_key))

def _evidence_with_category(record, *, category_code=None, category_name=None):
    evidence = dict(record.evidence)
    embedded = evidence.get("category_context") or {}
    code = str(category_code or embedded.get("code") or "").strip()
    name = str(category_name or embedded.get("name") or "").strip()
    if not code or not name:
        raise ValueError("Internal task 必须提供 category_code 和 category_name")
    evidence["category_context"] = {"code": code, "name": name}
    return evidence


def _internal_contract(record, *, category_code=None, category_name=None):
    prompt_evidence = _evidence_with_category(
        record,
        category_code=category_code,
        category_name=category_name,
    )
    if record.analysis_unit == ANALYSIS_UNIT_CLUSTER:
        return (
            INTERNAL_SYSTEM_PROMPT,
            build_internal_user_prompt(
                prompt_evidence,
                category_code=prompt_evidence["category_context"]["code"],
                category_name=prompt_evidence["category_context"]["name"],
            ),
            INTERNAL_CLUSTER_PROMPT_VERSION,
            INTERNAL_CLUSTER_SCHEMA_VERSION,
            prompt_evidence,
        )
    return (
        NOISE_TERM_SYSTEM_PROMPT,
        build_noise_term_user_prompt(prompt_evidence),
        INTERNAL_NOISE_PROMPT_VERSION,
        INTERNAL_NOISE_SCHEMA_VERSION,
        prompt_evidence,
    )


def prepare_internal_tasks(
    records,
    *,
    cache: ClusterCache,
    model: str,
    force_rerun=False,
    category_code: str,
    category_name: str,
):
    pending = []
    cached = []
    for record in records:
        system, user, prompt_version, schema_version, signature_evidence = (
            _internal_contract(
                record,
                category_code=category_code,
                category_name=category_name,
            )
        )
        signature = build_internal_signature(
            evidence=signature_evidence,
            model=model,
            prompt_version=prompt_version,
            schema_version=schema_version,
            system_prompt=system,
        )
        task = PreparedTask(
            "internal",
            record.cluster_id,
            record.query_key,
            signature,
            record.evidence_hash,
            user,
            record.evidence,
            record.analysis_unit,
            prompt_version,
            schema_version,
            system,
        )
        lookup = cache.lookup_by_query_key(
            phase="internal",
            query_key=record.query_key,
            signature=signature,
        )
        if lookup.hit and not force_rerun:
            task.cache_status = "hit"
            task.internal_insight = lookup.parsed_result
            cached.append(task)
        else:
            task.cache_status = lookup.reason or "pending"
            pending.append(task)
    return pending, cached

def prepare_external_tasks(records,*,internal_cache,external_cache,internal_model,external_model,force_rerun=False,only_recommended=True,skip_low_quality=True):
    pending=[];cached=[];skipped=[]
    for r in records:
        system,_,pv,sv,signature_evidence=_internal_contract(r);isig=build_internal_signature(evidence=signature_evidence,model=internal_model,prompt_version=pv,schema_version=sv,system_prompt=system);ilook=internal_cache.lookup_by_query_key(phase='internal',query_key=r.query_key,signature=isig)
        if not ilook.hit or not ilook.parsed_result:skipped.append({'query_key':r.query_key,'reason':'missing_valid_internal_insight'});continue
        insight=ilook.parsed_result
        if only_recommended and not insight.get('external_research_recommended',False):skipped.append({'query_key':r.query_key,'reason':'external_research_not_recommended'});continue
        if r.analysis_unit==ANALYSIS_UNIT_CLUSTER and skip_low_quality and (insight.get('cluster_quality') in {'low','mixed_or_invalid'} or insight.get('mapping_type')=='mixed_or_invalid_cluster'):skipped.append({'query_key':r.query_key,'reason':'low_or_mixed_cluster_quality'});continue
        sig=build_external_signature(evidence=r.evidence,internal_insight=insight,model=external_model,prompt_version=EXTERNAL_PROMPT_VERSION,schema_version=EXTERNAL_SCHEMA_VERSION,tools=EXTERNAL_TOOLS,system_prompt=EXTERNAL_SYSTEM_PROMPT);task=PreparedTask('external',r.cluster_id,r.query_key,sig,r.evidence_hash,build_external_user_prompt(r.evidence,insight),r.evidence,r.analysis_unit,EXTERNAL_PROMPT_VERSION,EXTERNAL_SCHEMA_VERSION,EXTERNAL_SYSTEM_PROMPT,insight,isig);lookup=external_cache.lookup_by_query_key(phase='external',query_key=r.query_key,signature=sig)
        (cached if lookup.hit and not force_rerun else pending).append(task)
    return pending,cached,skipped

def tasks_to_api_items(tasks):return [x.to_api_item() for x in tasks]
def task_index(tasks):
    out={}
    for t in tasks:
        if t.query_key in out:raise ValueError(f'query_key重复:{t.query_key}')
        out[t.query_key]=t
    return out
def evidence_summary_rows(records):
    rows=[]
    for r in records:
        m=r.cluster_metrics;d=r.evidence.get('attribute_diagnostics',{});e=r.evidence.get('exact_matches',{});c=r.taxonomy_candidates
        rows.append({'analysis_unit':r.analysis_unit,'query_key':r.query_key,'cluster_id':r.cluster_id,'source_cluster_id':r.source_cluster_id,'term':r.term,'display_name':r.display_name,'cluster_size':m.get('cluster_size'),'representative_terms':'; '.join(r.representative_term_names),'aggregate_ngram_docfreq_growth':m.get('aggregate_ngram_docfreq_growth'),'mean_cluster_probability':m.get('mean_cluster_probability'),'top1_attribute_code':c[0].get('attribute_code') if c else None,'top1_attribute_name':c[0].get('attribute_name') if c else None,'top1_similarity':d.get('top1_similarity'),'top1_top2_margin':d.get('top1_top2_margin'),'any_exact_match':e.get('any_candidate_has_exact_match',False),'evidence_hash':r.evidence_hash})
    return rows
def _num(v,d):return float(v) if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v)) else d
