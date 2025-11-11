#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-

"""
TSG → Inhibitors → Ligands pipeline (Ensembl-first), Python 3.12
- Вход: CSV с первой колонкой Ensembl Gene (ENSG...), колонкой 'geneName' с HGNC-символами.
- Ингибиторы учитываем только если они есть среди генов/белков пациента.
- Выход: Excel + CSV; отдельная таблица строк, где найден препарат в РЛС.

Конфигурация в CONFIG (вверху файла) + переменные окружения для токенов.
"""

from __future__ import annotations
import os, re, json, time, math, sys, html
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# =========================
# CONFIG
# =========================
CONFIG = {
    # пути
    "INPUT_CSV": "top_5000_RNA-seq_5_stat_.csv",
    "OUT_DIR": "pipeline_output",

    # режимы (можно включить онлайн-пути по одному)
    "USE_ONLINE_ENSEMBL": True,       # Ensembl REST для маппинга ENSG → HGNC/UniProt
    "USE_ONLINE_ONCOKB": False,       # OncoKB для уточнения TSG (нужен токен)
    "USE_ONLINE_CAUSAL": True,        # искать ингибиторы в SIGNOR/Reactome (при сбое → seed)
    "USE_ONLINE_CHEMBL": True,        # ChEMBL для Ki/Kd/IC50 (при сбое → seed)
    "USE_ONLINE_RLS": False,          # РЛС: Aurora API или best-effort rlsnet (см. ниже)

    # логика
    "REQUIRE_INHIBITOR_IN_PATIENT": True,  # брать ингибиторы только из списка пациента
    "AFFINITY_TYPES": {"KI", "KD", "IC50"},
    "AFFINITY_NM_MAX": 1e6,               # защитный фильтр по верхней границе нМ

    # RLS режим: "aurora" (официальный API), "scrape" (best-effort rlsnet), "off"
    "RLS_MODE": "off",  # options: "off" / "aurora" / "scrape"

    # HTTP
    "TIMEOUT": 30,
    "SLEEP_BETWEEN_CALLS": 0.05,

    # кеш
    "CACHE_DIR": ".cache_pipeline"
}

# Токены/URL берём из переменных окружения (или оставляем пустыми)
ENV = {
    "ONCOKB_TOKEN": os.getenv("ONCOKB_TOKEN", ""),
    "RLS_AURORA_URL": os.getenv("RLS_AURORA_URL", ""),  # например: https://api.rlsnet.ru/aurora
    "RLS_AURORA_KEY": os.getenv("RLS_AURORA_KEY", "")
}

# =========================
# SEED KNOWLEDGE (оффлайн)
# =========================
SEED_TSG = {
    "TP53","RB1","PTEN","CDKN2A","BRCA1","BRCA2","APC","VHL","NF1","NF2",
    "SMAD4","STK11","PTCH1","TSC1","TSC2","WT1","BAP1","ARID1A","PALB2",
    "MEN1","MLH1","MSH2","MSH6","PMS2","ATM","ATR","CHEK2","FBXW7",
    "KEAP1","NFE2L2","CDH1","SMARCB1"
}
SEED_TSG_INHIBITORS: Dict[str, List[str]] = {
    "TP53": ["MDM2","MDM4"],
    "RB1":  ["CDK4","CDK6"],
    "PTEN": ["CSNK2A1"],
    "SMAD4":["SKI","SKIL"],
    "CDKN2A":["CDK4","CDK6"],
}
SEED_INHIBITOR_LIGANDS: Dict[str, List[str]] = {
    "MDM2": ["Nutlin-3a","Idasanutlin","Milademetan"],
    "MDM4": ["ALRN-6924"],
    "CDK4": ["Palbociclib","Ribociclib","Abemaciclib"],
    "CDK6": ["Palbociclib","Ribociclib","Abemaciclib"],
    "CSNK2A1": ["Silmitasertib"],
    "SKI": [], "SKIL": [],
}

# =========================
# UTILS
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_csv_any(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")

def clean_ensembl(x: str) -> Optional[str]:
    x = str(x).strip()
    m = re.match(r"^(ENSG\d+)", x)
    return m.group(1) if m else None

def normalize_hgnc(s: str) -> Optional[str]:
    s2 = re.sub(r"\s+", "", str(s))
    s2 = s2.replace("–","-").replace("—","-")
    s2 = re.sub(r"[^A-Za-z0-9\-]", "", s2).upper()
    if re.match(r"^[A-Z][A-Z0-9\-]{1,}$", s2) and len(s2) <= 20:
        return s2
    return None

def session_json(headers: Optional[dict]=None) -> requests.Session:
    s = requests.Session()
    base_headers = {"User-Agent": "bio-tsg-pipeline/1.0"}
    if headers:
        base_headers.update(headers)
    s.headers.update(base_headers)
    return s

def cache_path(name: str) -> Path:
    return Path(CONFIG["CACHE_DIR"]) / f"{name}.json"

def load_cache(name: str) -> Optional[dict]:
    p = cache_path(name)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def save_cache(name: str, data: dict):
    ensure_dir(Path(CONFIG["CACHE_DIR"]))
    cache_path(name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# =========================
# 1) LOAD & NORMALIZE INPUT
# =========================
def load_and_extract_ids(csv_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = read_csv_any(csv_path)
    first_col = df.columns[0]
    df["ensembl_id"] = df[first_col].map(clean_ensembl)
    if "geneName" in df.columns:
        df["geneName_norm"] = df["geneName"].map(normalize_hgnc)
    else:
        df["geneName_norm"] = None
    ens_list = sorted(set([e for e in df["ensembl_id"].dropna().tolist() if e]))
    patient_symbols = sorted(set([s for s in df["geneName_norm"].dropna().tolist() if s]))
    return df, ens_list, patient_symbols

# =========================
# 2) MAPPING ENSG → HGNC / UniProt (Ensembl REST)
# =========================
def ensembl_xrefs(ensg: str) -> dict:
    """
    Ensembl REST:
      GET https://rest.ensembl.org/xrefs/id/{ENSG}?content-type=application/json
    Ищем dbname: 'HGNC' (display_id = символ), UniProt: 'Uniprot/SWISSPROT'|'Uniprot/SPTREMBL'
    """
    cache_key = f"ensembl_xrefs_{ensg}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached
    if not CONFIG["USE_ONLINE_ENSEMBL"]:
        return {"hgnc": "", "uniprot": ""}

    url = f"https://rest.ensembl.org/xrefs/id/{ensg}"
    s = session_json({"Content-Type": "application/json"})
    try:
        r = s.get(url, timeout=CONFIG["TIMEOUT"], headers={"Content-Type":"application/json"})
        if r.status_code != 200:
            return {"hgnc":"", "uniprot":""}
        data = r.json()
    except Exception:
        return {"hgnc":"", "uniprot":""}

    hgnc_symbol = ""
    uniprot_acc = ""
    for it in data:
        db = it.get("dbname","")
        if not hgnc_symbol and db in ("HGNC", "HGNC_gene"):
            hgnc_symbol = (it.get("display_id") or "").upper()
        if not uniprot_acc and db in ("Uniprot/SWISSPROT","Uniprot/SPTREMBL","Uniprot_gn"):
            uniprot_acc = it.get("primary_id") or ""
    out = {"hgnc": hgnc_symbol, "uniprot": uniprot_acc}
    save_cache(cache_key, out)
    time.sleep(CONFIG["SLEEP_BETWEEN_CALLS"])
    return out

def build_mapping_table(df_raw: pd.DataFrame, ens_list: List[str]) -> pd.DataFrame:
    # Base: candidate from geneName_norm
    base = (df_raw.dropna(subset=["ensembl_id"])
                  .groupby("ensembl_id", as_index=False)
                  .agg(candidate_hgnc=("geneName_norm", lambda x: next((y for y in x if isinstance(y,str) and y), ""))))
    # Online map:
    rows = []
    for ensg in tqdm(ens_list, desc="Ensembl mapping"):
        x = ensembl_xrefs(ensg)
        rows.append({"ensembl_id": ensg, "mapped_hgnc": x.get("hgnc",""), "mapped_uniprot": x.get("uniprot","")})
    mapped = pd.DataFrame(rows)
    m = mapped.merge(base, on="ensembl_id", how="left")
    m["final_hgnc"] = m["mapped_hgnc"].where(m["mapped_hgnc"].astype(str).str.len() > 0,
                                             m["candidate_hgnc"])
    m["final_hgnc"] = m["final_hgnc"].fillna("")
    return m

# =========================
# 3) TSG ANNOTATION (seed + OncoKB опционально)
# =========================
def onco_kb_is_tsg(hgnc: str) -> Tuple[bool, List[str]]:
    if not (CONFIG["USE_ONLINE_ONCOKB"] and ENV["ONCOKB_TOKEN"]):
        return False, []
    cache_key = f"oncokb_{hgnc}"
    cached = load_cache(cache_key)
    if cached is None:
        url = f"https://www.oncokb.org/api/v1/genes?hugoSymbol={hgnc}"
        s = session_json({"Authorization": f"Bearer {ENV['ONCOKB_TOKEN']}"})
        try:
            r = s.get(url, timeout=CONFIG["TIMEOUT"])
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = {}
        save_cache(cache_key, data)
    else:
        data = cached
    try:
        if isinstance(data, list) and data:
            g = data[0]
            if g.get("tumorSuppressor", False):
                return True, ["OncoKB"]
    except Exception:
        pass
    return False, []

def annotate_tsg(hgnc_list: List[str]) -> Dict[str, Dict]:
    out = {}
    for g in hgnc_list:
        is_seed = g in SEED_TSG
        sources = ["SEED_TSG"] if is_seed else []
        # дополнить OncoKB при наличии токена
        okb, okb_src = onco_kb_is_tsg(g)
        is_tsg = is_seed or okb
        if okb_src:
            sources.extend(okb_src)
        out[g] = {"is_tsg": is_tsg, "sources": sorted(set(sources))}
    return out

# =========================
# 4) INHIBITORS (SIGNOR / Reactome) + fallback
# =========================
def inhibitors_offline(tsg: str) -> List[Dict]:
    return [{"inhibitor": inh, "db":"SEED_DEMO", "edge_type":"inhibits"}
            for inh in SEED_TSG_INHIBITORS.get(tsg, [])]

def signor_inhibitors(tsg: str) -> List[Dict]:
    """
    SIGNOR best-effort:
      https://signor.uniroma2.it/api/search.php?entity=TP53   (JSON)
    Фильтруем по relations где effect ~ 'down-regulates'/'inhibits' и target == tsg.
    (Эндпоинты у SIGNOR иногда меняются; делаем tolerant parsing.)
    """
    url = f"https://signor.uniroma2.it/api/search.php?entity={tsg}"
    s = session_json()
    try:
        r = s.get(url, timeout=CONFIG["TIMEOUT"])
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []
    out = []
    # ожидаем список связей; поле имён может отличаться — перехватываем варианты
    for it in data if isinstance(data, list) else []:
        eff = (it.get("EFFECT") or it.get("effect") or "").lower()
        target = (it.get("TARGET") or it.get("to") or "").upper()
        source = (it.get("ENTITYA") or it.get("from") or "").upper()
        if target == tsg.upper() and any(k in eff for k in ["inhib","down"]):
            out.append({"inhibitor": source, "db": "SIGNOR", "edge_type": eff})
    time.sleep(CONFIG["SLEEP_BETWEEN_CALLS"])
    return out

def reactome_inhibitors(tsg: str) -> List[Dict]:
    """
    Reactome content service (best-effort, неоднозначно каузальность):
      Пройдём simplified: поиск партнёров и, если есть описание regulation со словами inhibit/downreg, отметить.
      Это приближённый метод; основную каузальность ждём от SIGNOR.
    """
    out = []
    # Поиск идентификатора Reactome/UniProt? Реализация очень вариативна; оставим пустым если нет чёткого пути.
    return out

def inhibitors_online(tsg: str) -> List[Dict]:
    got = []
    got.extend(signor_inhibitors(tsg))
    if not got:
        got.extend(reactome_inhibitors(tsg))
    # de-dup
    uniq = {}
    for x in got:
        k = x["inhibitor"]
        if k not in uniq:
            uniq[k] = x
    return list(uniq.values())

def fetch_inhibitors_for_tsg(tsg: str) -> List[Dict]:
    items: List[Dict] = []
    if CONFIG["USE_ONLINE_CAUSAL"]:
        try:
            items = inhibitors_online(tsg)
        except Exception:
            items = []
    if not items:
        items = inhibitors_offline(tsg)
    return items

# =========================
# 5) LIGANDS + AFFINITIES (ChEMBL) + fallback
# =========================
def chembl_find_target_id_by_name(name: str) -> Optional[str]:
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?query={requests.utils.quote(name)}"
    s = session_json()
    try:
        r = s.get(url, timeout=CONFIG["TIMEOUT"])
        r.raise_for_status()
        js = r.json()
        arr = js.get("targets", [])
        if not arr:
            return None
        # берём первый exact/наиболее релевантный
        return arr[0].get("target_chembl_id")
    except Exception:
        return None

def chembl_best_activities_for_target(target_chembl_id: str) -> List[Dict]:
    """
    Возвращает лучшие (минимальные) Ki/Kd/IC50 по каждому лиганду для данного target_chembl_id.
    """
    s = session_json()
    fields = "molecule_chembl_id,standard_type,standard_units,standard_value,standard_relation,document_chembl_id"
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_chembl_id}&limit=2000&fields={fields}"
    try:
        r = s.get(url, timeout=CONFIG["TIMEOUT"])
        r.raise_for_status()
        acts = r.json().get("activities", [])
    except Exception:
        return []
    def to_nm(val: str, units: str) -> Optional[float]:
        try:
            v = float(val)
        except Exception:
            return None
        u = (units or "").lower()
        if u in ("nm","nм"):
            return v
        if u in ("um","µm","μm"):
            return v * 1e3
        if u in ("pm",):
            return v * 1e-3
        if u in ("m","mol/l","mol"):
            return v * 1e9
        return None

    rows = []
    for a in acts:
        t = (a.get("standard_type") or "").upper()
        if t not in CONFIG["AFFINITY_TYPES"]:
            continue
        nm = to_nm(a.get("standard_value"), a.get("standard_units"))
        if nm is None or not (0 < nm <= CONFIG["AFFINITY_NM_MAX"]):
            continue
        rows.append({
            "ligand_chembl_id": a.get("molecule_chembl_id"),
            "affinity_type": t,
            "affinity_value_nM": nm,
            "reference": a.get("document_chembl_id")
        })
    # взять лучший по каждому лиганду
    best = {}
    for rec in rows:
        k = rec["ligand_chembl_id"]
        if k not in best or rec["affinity_value_nM"] < best[k]["affinity_value_nM"]:
            best[k] = rec
    # добавить имена молекул
    out = []
    for chembl_id, rec in best.items():
        name = chembl_molecule_name(chembl_id)  # может вернуть None
        out.append({
            "ligand": name or chembl_id,
            "ligand_chembl_id": chembl_id,
            "affinity_type": rec["affinity_type"],
            "affinity_value_nM": rec["affinity_value_nM"],
            "reference": rec["reference"]
        })
    return out

def chembl_molecule_name(chembl_id: str) -> Optional[str]:
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    s = session_json()
    try:
        r = s.get(url, timeout=CONFIG["TIMEOUT"])
        r.raise_for_status()
        js = r.json()
        pref = js.get("pref_name")
        if pref:
            return pref
        # иначе может быть у синонимов
        syns = js.get("molecule_synonyms", [])
        if syns:
            return syns[0].get("synonyms")
    except Exception:
        return None
    return None

def fetch_ligands_for_target(target_name: str) -> List[Dict]:
    if CONFIG["USE_ONLINE_CHEMBL"]:
        tid = chembl_find_target_id_by_name(target_name)
        if tid:
            acts = chembl_best_activities_for_target(tid)
            if acts:
                time.sleep(CONFIG["SLEEP_BETWEEN_CALLS"])
                return acts
    # fallback seed
    return [{"ligand": L, "affinity_type": None, "affinity_value_nM": None, "reference": None}
            for L in SEED_INHIBITOR_LIGANDS.get(target_name, [])]

# =========================
# 6) RLS (Aurora / scrape / off)
# =========================
def rls_status(inn_or_name: str) -> Dict:
    """Возвращает {has_RLS_drug: bool|None, brands: List[str]}"""
    mode = CONFIG["RLS_MODE"]
    if mode == "off" or not CONFIG["USE_ONLINE_RLS"]:
        return {"has_RLS_drug": None, "brands": []}
    if mode == "aurora":
        return rls_aurora(inn_or_name)
    if mode == "scrape":
        return rls_scrape_rlsnet(inn_or_name)
    return {"has_RLS_drug": None, "brands": []}

def rls_aurora(query: str) -> Dict:
    """Требует ENV['RLS_AURORA_URL'] и ENV['RLS_AURORA_KEY']."""
    base = ENV["RLS_AURORA_URL"]; key = ENV["RLS_AURORA_KEY"]
    if not (base and key):
        return {"has_RLS_drug": None, "brands": []}
    # пример: POST {base}/api/drugchoice2 (схема может отличаться у провайдера)
    url = f"{base.rstrip('/')}/api/drugchoice2"
    s = session_json({"X-Api-Key": key, "Content-Type": "application/json"})
    try:
        r = s.post(url, json={"q": query}, timeout=CONFIG["TIMEOUT"])
        r.raise_for_status()
        js = r.json()
        # допустим, js["items"] — список найденных препаратов
        items = js.get("items", [])
        brands = [it.get("tradeName") or it.get("name") for it in items if it.get("name") or it.get("tradeName")]
        return {"has_RLS_drug": bool(items), "brands": list(sorted(set([b for b in brands if b])))}
    except Exception:
        return {"has_RLS_drug": None, "brands": []}

def rls_scrape_rlsnet(query: str) -> Dict:
    """
    Очень грубый best-effort HTML-поиск rlsnet.ru (может ломаться).
    Мы делаем GET на /search? , ищем в HTML наличие блоков результатов.
    """
    s = session_json()
    try:
        url = f"https://www.rlsnet.ru/search_result.htm?word={requests.utils.quote(query)}"
        r = s.get(url, timeout=CONFIG["TIMEOUT"])
        if r.status_code != 200:
            return {"has_RLS_drug": None, "brands": []}
        text = r.text
        # примитивная эвристика: наличие фразы "Препараты" и элементов результатов
        has = ("Препараты" in text) or ("Лекарственные средства" in text)
        # вытащим первые несколько названий (если попадутся)
        brand_matches = re.findall(r'<a[^>]+class="search__title"[^>]*>(.*?)</a>', text, flags=re.IGNORECASE)
        brands = [html.unescape(re.sub(r"<.*?>", "", b)).strip() for b in brand_matches]
        brands = [b for b in brands if b]
        return {"has_RLS_drug": bool(brands) or has, "brands": list(dict.fromkeys(brands))[:10]}
    except Exception:
        return {"has_RLS_drug": None, "brands": []}

# =========================
# 7) PIPELINE
# =========================
def run_pipeline() -> None:
    in_path = Path(CONFIG["INPUT_CSV"])
    out_dir = Path(CONFIG["OUT_DIR"])
    ensure_dir(out_dir)
    ensure_dir(Path(CONFIG["CACHE_DIR"]))

    # 1) load
    df_raw, ens_list, patient_symbols = load_and_extract_ids(in_path)

    # 2) mapping
    mapping = build_mapping_table(df_raw, ens_list)
    # annotate TSG
    tsg_anno = annotate_tsg([x for x in mapping["final_hgnc"].tolist() if x])
    mapping["is_tsg"] = mapping["final_hgnc"].map(lambda x: tsg_anno.get(x, {}).get("is_tsg", False))
    mapping["tsg_sources"] = mapping["final_hgnc"].map(lambda x: ";".join(tsg_anno.get(x, {}).get("sources", [])))

    # Список TSG у пациента
    tsg_rows = mapping[mapping["is_tsg"] & mapping["final_hgnc"].astype(bool)]

    # 3) inhibitors per TSG (и фильтрация по наличию у пациента)
    patient_symbol_set = set(patient_symbols)
    pipeline_rows = []
    for _, row in tsg_rows.iterrows():
        tsg = row["final_hgnc"]
        rels = fetch_inhibitors_for_tsg(tsg)
        if CONFIG["REQUIRE_INHIBITOR_IN_PATIENT"]:
            rels = [r for r in rels if r["inhibitor"] in patient_symbol_set]
        if not rels:
            pipeline_rows.append({
                "ensembl_id": row["ensembl_id"],
                "tsg_symbol": tsg,
                "evidence_db": None,
                "inhibitor": None,
                "ligand": None,
                "affinity_type": None,
                "affinity_value_nM": None,
                "reference": None,
                "has_RLS_drug": None,
                "RLS_brand_names": None
            })
            continue
        for r in rels:
            inh = r["inhibitor"]
            ligs = fetch_ligands_for_target(inh)
            if not ligs:
                pipeline_rows.append({
                    "ensembl_id": row["ensembl_id"],
                    "tsg_symbol": tsg,
                    "evidence_db": r["db"],
                    "inhibitor": inh,
                    "ligand": None,
                    "affinity_type": None,
                    "affinity_value_nM": None,
                    "reference": None,
                    "has_RLS_drug": None,
                    "RLS_brand_names": None
                })
                continue
            for L in ligs:
                ligand_name = L.get("ligand") or L.get("ligand_chembl_id")
                rls = rls_status(str(ligand_name))
                pipeline_rows.append({
                    "ensembl_id": row["ensembl_id"],
                    "tsg_symbol": tsg,
                    "evidence_db": r["db"],
                    "inhibitor": inh,
                    "ligand": ligand_name,
                    "affinity_type": L.get("affinity_type"),
                    "affinity_value_nM": L.get("affinity_value_nM"),
                    "reference": L.get("reference"),
                    "has_RLS_drug": rls["has_RLS_drug"],
                    "RLS_brand_names": ",".join(rls["brands"]) if rls["brands"] else None
                })

    pipeline_df = pd.DataFrame(pipeline_rows)
    if pipeline_df.empty:
        pipeline_df = pd.DataFrame(columns=[
            "ensembl_id","tsg_symbol","evidence_db","inhibitor","ligand",
            "affinity_type","affinity_value_nM","reference","has_RLS_drug","RLS_brand_names"
        ])

    # 4) отдельная таблица строк с найденным препаратом в РЛС
    rls_hits = pipeline_df.copy()
    # True/False/None → берём только True
    rls_hits = rls_hits[rls_hits["has_RLS_drug"] == True]

    # 5) save
    out_mapping_csv = out_dir / "ensembl_mapping_table.csv"
    out_pipeline_xlsx = out_dir / "tsg_inhibitor_ligand.xlsx"
    out_rls_csv = out_dir / "rls_hits.csv"

    mapping.to_csv(out_mapping_csv, index=False)
    rls_hits.to_csv(out_rls_csv, index=False)

    with pd.ExcelWriter(out_pipeline_xlsx, engine="openpyxl") as w:
        mapping.to_excel(w, index=False, sheet_name="ensembl_mapping")
        pipeline_df.to_excel(w, index=False, sheet_name="tsg_inhibitor_ligand")
        rls_hits.to_excel(w, index=False, sheet_name="rls_hits")

    print(f"[OK] mapping:  {out_mapping_csv}")
    print(f"[OK] pipeline: {out_pipeline_xlsx}")
    print(f"[OK] rls_hits: {out_rls_csv}")
    print("Done.")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 12):
        print(f"Python 3.12+ required, found {major}.{minor}", file=sys.stderr)
        sys.exit(1)
    run_pipeline()