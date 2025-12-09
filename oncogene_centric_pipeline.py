#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Онкоген-центричный пайплайн (с DrugBank full database.xml):

1) Для данных пациента ищем онкогены.
2) Для этих онкогенов ищем, какие супрессоры они блокируют (TRRUST, Repression).
3) Для найденных супрессоров ищем лиганды (лекарственные средства), которые их активируют.
4) Для онкогенов, блокирующих какие-либо из этих супрессоров, ищем лиганды, блокирующие онкогены.

Используем те же источники, что и в TSG-пайплайне:

Входные файлы
-------------
1) RNA-seq: top_5000_RNA-seq_5_stat_.csv
   - разделитель: ';'
   - обязательная колонка: geneName (HGNC-символы)

2) cancerGeneList.tsv
   - минимум: "Hugo Symbol", "Gene Type"
   - онкогены: Gene Type ∈ {"ONCOGENE", "ONCOGENE_AND_TSG"} (можно расширить)

3) trrust_rawdata.human.tsv
   - таб-разделённые колонки:
     0: TF (регулятор)
     1: Target gene (мишень)
     2: Regulation ("Activation", "Repression", "Unknown")
     3: PubMed IDs

4) interactions.csv (GtoPdb Interactions Dataset)
   - разделитель: запятая
   - первая строка — комментарий ("# GtoPdb Version: ...") → header=1
   - нужные колонки:
       "Target Gene Symbol"
       "Target"
       "Target ID"
       "Target Species"
       "Primary Target"
       "Ligand"
       "Ligand ID"
       "Ligand Type"
       "Affinity Median"
       "Affinity Units"
       "Original Affinity Median nm"
       "Original Affinity Relation"
       "PubMed ID"
       "Action"

5) full database.xml (DrugBank full database)
   - используем:
       <drugbank-id>      → drugbank_id
       <name>             → drugbank_name
       <groups><group>    → drugbank_groups (множество статусов)
   - препарат считается "одобренным", если среди group есть "approved"
     (регистр не важен).

Выходные файлы (в --out-dir)
----------------------------
1) patient_oncogenes.tsv
   - oncogene_symbol, gene_type

2) oncogene_tsg_repressions.tsv
   - oncogene_symbol, tsg_symbol, regulation, pmids
   (онкоген репрессирует TSG в TRRUST)

3) tsg_activating_drugs.tsv
   - супрессоры и лиганды, их активирующие (agonist/activator/positive modulator/partial agonist)

4) tsg_activating_drugs_approved.tsv  (DrugBank-approved)
   - (3) ∩ approved по DrugBank (матч по ligand_name ↔ DrugBank name)
   - + drugbank_id, drugbank_name, drugbank_groups

5) oncogene_blocking_drugs.tsv
   - онкогены (из п.2), для которых найдены активируемые TSG из п.3, и лиганды, эти онкогены блокирующие
     (Action ∈ {inhibitor, antagonist, blocker})

6) oncogene_blocking_drugs_approved.tsv (DrugBank-approved)
   - аналогично (5) + DrugBank-колонки

7) tsg_activator_drug_list.tsv
   - одна колонка ligand_name, уникальные препараты-активаторы супрессоров

8) oncogene_blocker_drug_list.tsv
   - одна колонка ligand_name, уникальные препараты-блокаторы онкогенов

9) all_oncogene_blocking_drugs.tsv / all_oncogene_blocking_drugs_approved.tsv
   - блокаторы любых онкогенов пациента (независимо от связи с TSG)

10) all_oncogene_blocker_drug_list.tsv
   - список всех блокаторов онкогенов (любых)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set, Tuple

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# --- Общие вспомогательные функции -----------------------------------------


def load_patient_genes(rna_path: Path) -> Set[str]:
    """
    Загрузка генов пациента из RNA-seq файла.

    Ожидается CSV с разделителем ';' и колонкой 'geneName'.
    Значения NA/пустые строки отбрасываются.
    """
    df = pd.read_csv(rna_path, sep=";")
    if "geneName" not in df.columns:
        raise ValueError(f"'geneName' column not found in {rna_path}")

    genes = (
        df["geneName"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    genes = genes[genes.str.upper() != "NA"]

    return set(genes)


def load_cancer_genes(
    cancer_path: Path,
    oncogene_types: Iterable[str] = ("ONCOGENE", "ONCOGENE_AND_TSG"),
    tsg_types: Iterable[str] = ("TSG", "ONCOGENE_AND_TSG"),
) -> Tuple[Set[str], Set[str], pd.DataFrame]:
    """
    Загрузка онкогенов и TSG из cancerGeneList.tsv.
    """
    df = pd.read_csv(cancer_path, sep="\t")

    required_cols = {"Hugo Symbol", "Gene Type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Columns {missing} not found in cancer gene list {cancer_path}"
        )

    # Онкогены
    mask_onco = df["Gene Type"].isin(list(oncogene_types))
    oncogene_symbols = (
        df.loc[mask_onco, "Hugo Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Супрессоры
    mask_tsg = df["Gene Type"].isin(list(tsg_types))
    tsg_symbols = (
        df.loc[mask_tsg, "Hugo Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return set(oncogene_symbols), set(tsg_symbols), df


def load_trrust(trrust_path: Path) -> pd.DataFrame:
    """
    Загрузка TRRUST (human).

    Формат: 4 таб-разделённых столбца:
        TF, Target, Regulation, PubMed IDs
    """
    cols = ["regulator", "target", "regulation", "pmids"]
    df = pd.read_csv(trrust_path, sep="\t", header=None, names=cols)

    df["regulation"] = df["regulation"].astype(str).str.strip()
    df["regulator"] = df["regulator"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    return df


def load_interactions(path: Path) -> pd.DataFrame:
    """
    Загрузка interactions.csv из Guide to PHARMACOLOGY.
    """
    df = pd.read_csv(path, sep=",", header=1, low_memory=False)

    required_cols = [
        "Target Gene Symbol",
        "Target",
        "Target ID",
        "Target Species",
        "Primary Target",
        "Ligand",
        "Ligand ID",
        "Ligand Type",
        "Affinity Median",
        "Affinity Units",
        "Original Affinity Median nm",
        "Original Affinity Relation",
        "PubMed ID",
        "Action",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Нет колонок: {missing}\nНайдено: {list(df.columns)}")

    df["Target_Gene_upper"] = (
        df["Target Gene Symbol"].astype(str).str.upper().str.strip()
    )
    df["Action_upper"] = df["Action"].astype(str).str.upper().str.strip()

    return df


# --- DrugBank XML → approved препараты -------------------------------------


def load_approved_ligands(path: Path) -> pd.DataFrame:
    """
    Загрузка DrugBank full database XML.

    Извлекаем:
      - drugbank_id  (primary=true, если есть, иначе первый <drugbank-id>)
      - name         (основное имя препарата)
      - drugbank_groups (список group, объединённый через ';')
      - drugbank_name_norm (NAME в верхнем регистре, для матчинга)
      - is_approved  (True, если среди groups есть 'approved')
    """
    ns = {"db": "http://www.drugbank.ca"}

    tree = ET.parse(path)
    root = tree.getroot()

    rows = []

    for drug in root.findall("db:drug", ns):
        # drugbank-id (primary, если есть)
        primary_id_elem = drug.find("db:drugbank-id[@primary='true']", ns)
        if primary_id_elem is not None and primary_id_elem.text:
            drugbank_id = primary_id_elem.text.strip()
        else:
            # fallback: первый <drugbank-id>
            first_id = drug.find("db:drugbank-id", ns)
            drugbank_id = (
                first_id.text.strip()
                if first_id is not None and first_id.text
                else None
            )

        name_elem = drug.find("db:name", ns)
        if name_elem is None or not name_elem.text:
            continue
        name = name_elem.text.strip()

        groups_elem = drug.find("db:groups", ns)
        groups_list = []
        if groups_elem is not None:
            for g in groups_elem.findall("db:group", ns):
                if g.text:
                    groups_list.append(g.text.strip())

        if not groups_list:
            # Можно не пропускать, но для наших целей нужны хотя бы какие-то группы
            continue

        groups_str = ";".join(groups_list)

        rows.append(
            {
                "drugbank_id": drugbank_id,
                "name": name,
                "drugbank_groups": groups_str,
            }
        )

    if not rows:
        raise RuntimeError(f"Не удалось извлечь ни одного препарата из {path}")

    df = pd.DataFrame(rows)

    df["drugbank_name_norm"] = (
        df["name"].astype(str).str.strip().str.upper()
    )
    df["is_approved"] = df["drugbank_groups"].astype(str).str.contains(
        "approved", case=False, na=False
    )

    return df


def filter_approved(
    drugs_df: pd.DataFrame,
    approved_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Оставляем только те строки из drugs_df, чьи ligand_name присутствуют
    среди approved-препаратов в DrugBank (по имени, без регистра).

    Возвращает drugs_df + добавленные колонки:
      - drugbank_id
      - drugbank_name
      - drugbank_groups
    """
    approved_only = approved_df[approved_df["is_approved"]].copy()

    if approved_only.empty or drugs_df.empty:
        return drugs_df.iloc[0:0].copy()

    # Уникальные DrugBank-записи по нормализованному имени
    approved_unique = (
        approved_only
        .loc[:, ["drugbank_id", "name", "drugbank_name_norm", "drugbank_groups"]]
        .drop_duplicates(subset=["drugbank_name_norm"])
    )

    tsg = drugs_df.copy()
    tsg["ligand_name_norm"] = (
        tsg["ligand_name"].astype(str).str.strip().str.upper()
    )

    merged = tsg.merge(
        approved_unique,
        left_on="ligand_name_norm",
        right_on="drugbank_name_norm",
        how="inner",
    )

    if merged.empty:
        return merged

    merged = merged.rename(
        columns={
            "name": "drugbank_name",
        }
    )

    merged = merged.drop(columns=["ligand_name_norm", "drugbank_name_norm"])

    return merged


def parse_affinity(value):
    """Преобразование строк вида '<1', '>1000' и чисел в float (нМ)."""
    if pd.isna(value):
        return np.inf

    s = str(value).strip()

    if s.startswith("<"):
        try:
            return float(s[1:])
        except Exception:
            return np.inf

    if s.startswith(">"):
        try:
            return float(s[1:])
        except Exception:
            return np.inf

    try:
        return float(s)
    except Exception:
        return np.inf


# --- Специфическая логика пайплайна ----------------------------------------


# действия, активирующие таргет
ACTIVATING_ACTIONS = {
    "AGONIST",
    "ACTIVATOR",
    "POSITIVE MODULATOR",
    "PARTIAL AGONIST",
}

# действия, блокирующие таргет
INHIBITING_ACTIONS = {
    "INHIBITOR",
    "ANTAGONIST",
    "BLOCKER",
}


def build_oncogene_tsg_repressions(
    trrust_df: pd.DataFrame,
    patient_oncogenes: Set[str],
    all_tsg_symbols: Set[str],
) -> pd.DataFrame:
    """
    Находим, какие супрессоры блокируют (репрессируют) онкогены пациента.

    Условие:
    - regulator ∈ patient_oncogenes
    - target ∈ all_tsg_symbols
    - regulation == "Repression" (без учёта регистра)
    """
    if not patient_oncogenes or not all_tsg_symbols:
        return pd.DataFrame(
            columns=["oncogene_symbol", "tsg_symbol", "regulation", "pmids"]
        )

    onco_upper = {g.upper() for g in patient_oncogenes}
    tsg_upper = {g.upper() for g in all_tsg_symbols}

    df = trrust_df.copy()
    df["regulator_upper"] = df["regulator"].str.upper()
    df["target_upper"] = df["target"].str.upper()
    df["regulation_lower"] = df["regulation"].str.lower()

    mask = (
        df["regulator_upper"].isin(onco_upper)
        & df["target_upper"].isin(tsg_upper)
        & (df["regulation_lower"] == "repression")
    )

    sub = df.loc[mask, ["regulator", "target", "regulation", "pmids"]].copy()
    sub = sub.rename(
        columns={"regulator": "oncogene_symbol", "target": "tsg_symbol"}
    )
    sub = sub.drop_duplicates().sort_values(["oncogene_symbol", "tsg_symbol"])

    return sub


def build_tsg_activating_drugs(
    oncogene_tsg_df: pd.DataFrame,
    interactions: pd.DataFrame,
    human_only: bool = True,
    primary_only: bool = True,
) -> pd.DataFrame:
    """
    Для найденных супрессоров (tsg_symbol) ищем лиганды, их активирующие.

    - фильтр по Target Species == Human (опционально)
    - фильтр по Primary Target == True (опционально)
    - Action ∈ ACTIVATING_ACTIONS
    """
    if oncogene_tsg_df.empty:
        return pd.DataFrame(
            columns=[
                "tsg_symbol",
                "target_name",
                "target_id",
                "target_gene_symbol",
                "ligand_name",
                "ligand_id",
                "ligand_type",
                "affinity_median",
                "affinity_units",
                "original_affinity_median_nm",
                "original_affinity_relation",
                "interaction_pubmed_id",
                "action",
            ]
        )

    tsgs = set(oncogene_tsg_df["tsg_symbol"].astype(str).str.upper())

    df_int = interactions.copy()

    if human_only:
        df_int = df_int[df_int["Target Species"] == "Human"]

    if primary_only:
        df_int = df_int[df_int["Primary Target"] == True]

    df_int = df_int[df_int["Action_upper"].isin(ACTIVATING_ACTIONS)]
    df_int = df_int[df_int["Target_Gene_upper"].isin(tsgs)]

    if df_int.empty:
        return pd.DataFrame(
            columns=[
                "tsg_symbol",
                "target_name",
                "target_id",
                "target_gene_symbol",
                "ligand_name",
                "ligand_id",
                "ligand_type",
                "affinity_median",
                "affinity_units",
                "original_affinity_median_nm",
                "original_affinity_relation",
                "interaction_pubmed_id",
                "action",
            ]
        )

    df_int = df_int.rename(
        columns={
            "Target": "target_name",
            "Target ID": "target_id",
            "Target Gene Symbol": "target_gene_symbol",
            "Ligand": "ligand_name",
            "Ligand ID": "ligand_id",
            "Ligand Type": "ligand_type",
            "Affinity Median": "affinity_median",
            "Affinity Units": "affinity_units",
            "Original Affinity Median nm": "original_affinity_median_nm",
            "Original Affinity Relation": "original_affinity_relation",
            "PubMed ID": "interaction_pubmed_id",
            "Action": "action",
        }
    )

    df_int["tsg_symbol"] = df_int["target_gene_symbol"].astype(str).str.strip()

    cols = [
        "tsg_symbol",
        "target_name",
        "target_id",
        "target_gene_symbol",
        "ligand_name",
        "ligand_id",
        "ligand_type",
        "affinity_median",
        "affinity_units",
        "original_affinity_median_nm",
        "original_affinity_relation",
        "interaction_pubmed_id",
        "action",
    ]

    return df_int[cols].sort_values(["tsg_symbol", "ligand_name"])


def build_oncogene_blocking_drugs(
    oncogene_tsg_df: pd.DataFrame,
    tsg_activating_drugs_df: pd.DataFrame,
    interactions: pd.DataFrame,
    human_only: bool = True,
    primary_only: bool = True,
) -> pd.DataFrame:
    """
    Для онкогенов, блокирующих супрессоры с найденными активирующими лигандами,
    ищем лиганды, блокирующие эти онкогены.

    - берём TSG, для которых есть активирующие лиганды
    - смотрим, какие онкогены их репрессируют (из oncogene_tsg_df)
    - для таких онкогенов смотрим interactions с Action ∈ INHIBITING_ACTIONS
    """
    if oncogene_tsg_df.empty or tsg_activating_drugs_df.empty:
        return pd.DataFrame(
            columns=[
                "oncogene_symbol",
                "target_name",
                "target_id",
                "target_gene_symbol",
                "ligand_name",
                "ligand_id",
                "ligand_type",
                "affinity_median",
                "affinity_units",
                "original_affinity_median_nm",
                "original_affinity_relation",
                "interaction_pubmed_id",
                "action",
            ]
        )

    # TSG, для которых есть активирующие лиганды
    tsg_with_drugs = set(
        tsg_activating_drugs_df["tsg_symbol"].astype(str).str.upper()
    )

    # Онкогены, которые репрессируют эти TSG
    tmp = oncogene_tsg_df.copy()
    tmp["tsg_upper"] = tmp["tsg_symbol"].astype(str).str.upper()
    tmp["oncogene_upper"] = tmp["oncogene_symbol"].astype(str).str.upper()

    tmp = tmp[tmp["tsg_upper"].isin(tsg_with_drugs)]

    if tmp.empty:
        return pd.DataFrame(
            columns=[
                "oncogene_symbol",
                "target_name",
                "target_id",
                "target_gene_symbol",
                "ligand_name",
                "ligand_id",
                "ligand_type",
                "affinity_median",
                "affinity_units",
                "original_affinity_median_nm",
                "original_affinity_relation",
                "interaction_pubmed_id",
                "action",
            ]
        )

    oncogenes_of_interest = set(tmp["oncogene_upper"])

    df_int = interactions.copy()

    if human_only:
        df_int = df_int[df_int["Target Species"] == "Human"]

    if primary_only:
        df_int = df_int[df_int["Primary Target"] == True]

    df_int = df_int[df_int["Action_upper"].isin(INHIBITING_ACTIONS)]
    df_int = df_int[df_int["Target_Gene_upper"].isin(oncogenes_of_interest)]

    if df_int.empty:
        return pd.DataFrame(
            columns=[
                "oncogene_symbol",
                "target_name",
                "target_id",
                "target_gene_symbol",
                "ligand_name",
                "ligand_id",
                "ligand_type",
                "affinity_median",
                "affinity_units",
                "original_affinity_median_nm",
                "original_affinity_relation",
                "interaction_pubmed_id",
                "action",
            ]
        )

    df_int = df_int.rename(
        columns={
            "Target": "target_name",
            "Target ID": "target_id",
            "Target Gene Symbol": "target_gene_symbol",
            "Ligand": "ligand_name",
            "Ligand ID": "ligand_id",
            "Ligand Type": "ligand_type",
            "Affinity Median": "affinity_median",
            "Affinity Units": "affinity_units",
            "Original Affinity Median nm": "original_affinity_median_nm",
            "Original Affinity Relation": "original_affinity_relation",
            "PubMed ID": "interaction_pubmed_id",
            "Action": "action",
        }
    )

    df_int["oncogene_symbol"] = df_int["target_gene_symbol"].astype(str).str.strip()

    cols = [
        "oncogene_symbol",
        "target_name",
        "target_id",
        "target_gene_symbol",
        "ligand_name",
        "ligand_id",
        "ligand_type",
        "affinity_median",
        "affinity_units",
        "original_affinity_median_nm",
        "original_affinity_relation",
        "interaction_pubmed_id",
        "action",
    ]

    return df_int[cols].sort_values(["oncogene_symbol", "ligand_name"])


def build_all_oncogene_blocking_drugs(
    patient_oncogenes: Set[str],
    interactions: pd.DataFrame,
    human_only: bool = True,
    primary_only: bool = True,
) -> pd.DataFrame:
    """
    Ищем лиганды, блокирующие ЛЮБЫЕ онкогены пациента,
    независимо от того, блокируют ли они супрессоры.

    - Target Gene Symbol ∈ patient_oncogenes
    - Action ∈ INHIBITING_ACTIONS
    """
    if not patient_oncogenes:
        return pd.DataFrame(
            columns=[
                "oncogene_symbol",
                "target_name",
                "target_id",
                "target_gene_symbol",
                "ligand_name",
                "ligand_id",
                "ligand_type",
                "affinity_median",
                "affinity_units",
                "original_affinity_median_nm",
                "original_affinity_relation",
                "interaction_pubmed_id",
                "action",
            ]
        )

    onco_upper = {g.upper() for g in patient_oncogenes}

    df_int = interactions.copy()

    if human_only:
        df_int = df_int[df_int["Target Species"] == "Human"]

    if primary_only:
        df_int = df_int[df_int["Primary Target"] == True]

    df_int = df_int[df_int["Action_upper"].isin(INHIBITING_ACTIONS)]
    df_int = df_int[df_int["Target_Gene_upper"].isin(onco_upper)]

    if df_int.empty:
        return pd.DataFrame(
            columns=[
                "oncogene_symbol",
                "target_name",
                "target_id",
                "target_gene_symbol",
                "ligand_name",
                "ligand_id",
                "ligand_type",
                "affinity_median",
                "affinity_units",
                "original_affinity_median_nm",
                "original_affinity_relation",
                "interaction_pubmed_id",
                "action",
            ]
        )

    df_int = df_int.rename(
        columns={
            "Target": "target_name",
            "Target ID": "target_id",
            "Target Gene Symbol": "target_gene_symbol",
            "Ligand": "ligand_name",
            "Ligand ID": "ligand_id",
            "Ligand Type": "ligand_type",
            "Affinity Median": "affinity_median",
            "Affinity Units": "affinity_units",
            "Original Affinity Median nm": "original_affinity_median_nm",
            "Original Affinity Relation": "original_affinity_relation",
            "PubMed ID": "interaction_pubmed_id",
            "Action": "action",
        }
    )

    df_int["oncogene_symbol"] = df_int["target_gene_symbol"].astype(str).str.strip()

    cols = [
        "oncogene_symbol",
        "target_name",
        "target_id",
        "target_gene_symbol",
        "ligand_name",
        "ligand_id",
        "ligand_type",
        "affinity_median",
        "affinity_units",
        "original_affinity_median_nm",
        "original_affinity_relation",
        "interaction_pubmed_id",
        "action",
    ]

    return df_int[cols].sort_values(["oncogene_symbol", "ligand_name"])


# --- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Онкоген-центричный пайплайн: "
            "1) онкогены пациента → 2) супрессоры, которые они блокируют (TRRUST) → "
            "3) активирующие супрессоры препараты → 4) препараты, блокирующие онкогены. "
            "(DrugBank full database.xml как источник approved-препаратов)"
        )
    )

    parser.add_argument("--rna", type=Path, required=True,
                        help="Путь к RNA-seq CSV (top_5000_RNA-seq_5_stat_.csv).")
    parser.add_argument("--cancer-list", type=Path, required=True,
                        help="Путь к cancerGeneList.tsv.")
    parser.add_argument("--trrust", type=Path, required=True,
                        help="Путь к trrust_rawdata.human.tsv.")
    parser.add_argument("--interactions", type=Path, required=True,
                        help="Путь к GtoPdb interactions.csv.")

    # DrugBank XML (оставляем старое имя параметра как алиас)
    parser.add_argument(
        "--drugbank-xml",
        "--approved-interactions",
        dest="drugbank_xml",
        type=Path,
        required=False,
        help="(Опционально) Путь к DrugBank full database XML (full database.xml).",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Каталог для записи результатов (по умолчанию: текущий).",
    )

    parser.add_argument("--no-human-filter", action="store_true",
                        help="Не ограничивать Target Species == 'Human'.")
    parser.add_argument("--no-primary-only", action="store_true",
                        help="Не ограничивать Primary Target == True.")

    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Гены пациента, онкогены и TSG -----------------------------------
    patient_genes = load_patient_genes(args.rna)

    oncogenes_all, tsg_all, cancer_df = load_cancer_genes(args.cancer_list)

    # онкогены пациента
    patient_oncogenes = sorted(oncogenes_all.intersection(patient_genes))

    if patient_oncogenes:
        onco_df = (
            cancer_df[cancer_df["Hugo Symbol"].isin(patient_oncogenes)]
            .loc[:, ["Hugo Symbol", "Gene Type"]]
            .rename(
                columns={
                    "Hugo Symbol": "oncogene_symbol",
                    "Gene Type": "gene_type",
                }
            )
            .drop_duplicates()
            .sort_values("oncogene_symbol")
        )
    else:
        onco_df = pd.DataFrame(columns=["oncogene_symbol", "gene_type"])

    onco_path = out_dir / "patient_oncogenes.tsv"
    onco_df.to_csv(onco_path, sep="\t", index=False)
    print(f"Онкогены пациента записаны в: {onco_path} "
          f"({len(onco_df)} генов)")

    # --- 2) Какие супрессоры они блокируют (TRRUST, Repression) ------------
    trrust_df = load_trrust(args.trrust)

    oncogene_tsg_df = build_oncogene_tsg_repressions(
        trrust_df=trrust_df,
        patient_oncogenes=set(patient_oncogenes),
        all_tsg_symbols=tsg_all,
    )

    oncogene_tsg_path = out_dir / "oncogene_tsg_repressions.tsv"
    oncogene_tsg_df.to_csv(oncogene_tsg_path, sep="\t", index=False)
    print(
        f"Онкогенов → TSG (Repression) найдено строк: {len(oncogene_tsg_df)} "
        f"→ {oncogene_tsg_path}"
    )

    # --- 3) Для найденных супрессоров ищем активирующие лиганды ------------
    interactions = load_interactions(args.interactions)

    human_only = not args.no_human_filter
    primary_only = not args.no_primary_only

    # --- 3a) ЛЮБЫЕ блокаторы всех онкогенов пациента -----------------------
    all_onco_blocking_drugs_df = build_all_oncogene_blocking_drugs(
        patient_oncogenes=set(patient_oncogenes),
        interactions=interactions,
        human_only=human_only,
        primary_only=primary_only,
    )

    all_onco_drugs_path = out_dir / "all_oncogene_blocking_drugs.tsv"
    all_onco_blocking_drugs_df.to_csv(all_onco_drugs_path, sep="\t", index=False)
    print(
        f"Лекарства, блокирующие любые онкогены пациента: "
        f"{len(all_onco_blocking_drugs_df)} строк → {all_onco_drugs_path}"
    )

    # (опц.) фильтр по DrugBank-approved
    if args.drugbank_xml is not None:
        approved_df = load_approved_ligands(args.drugbank_xml)
        n_total = len(approved_df)
        n_approved = int(approved_df["is_approved"].sum())
        print(
            f"DrugBank: извлечено препаратов {n_total}, "
            f"из них с группой 'approved': {n_approved}"
        )

        all_onco_blocking_approved = filter_approved(
            all_onco_blocking_drugs_df, approved_df
        )
        all_onco_blocking_approved_path = (
            out_dir / "all_oncogene_blocking_drugs_approved.tsv"
        )
        all_onco_blocking_approved.to_csv(
            all_onco_blocking_approved_path, sep="\t", index=False
        )
        print(
            f"Лекарства (DrugBank-approved), блокирующие любые онкогены пациента: "
            f"{len(all_onco_blocking_approved)} строк → {all_onco_blocking_approved_path}"
        )

        if not all_onco_blocking_approved.empty:
            df_sorted_all = all_onco_blocking_approved.copy()
            df_sorted_all["affinity_nM"] = df_sorted_all[
                "original_affinity_median_nm"
            ].apply(parse_affinity)
            df_sorted_all = df_sorted_all.sort_values(
                by=["affinity_nM", "oncogene_symbol", "ligand_name"],
                ascending=[True, True, True],
            )
            sorted_all_path = (
                out_dir / "all_oncogene_blocking_drugs_approved_sorted_by_affinity.tsv"
            )
            df_sorted_all.to_csv(sorted_all_path, sep="\t", index=False)
            print(
                f"Все онкогены: approved-блокаторы, отсортированные по аффинности "
                f"→ {sorted_all_path}"
            )

    tsg_activating_drugs_df = build_tsg_activating_drugs(
        oncogene_tsg_df=oncogene_tsg_df,
        interactions=interactions,
        human_only=human_only,
        primary_only=primary_only,
    )

    tsg_drugs_path = out_dir / "tsg_activating_drugs.tsv"
    tsg_activating_drugs_df.to_csv(tsg_drugs_path, sep="\t", index=False)
    print(
        f"Супрессоры с активирующими лигандами: {len(tsg_activating_drugs_df)} "
        f"строк → {tsg_drugs_path}"
    )

    # при наличии DrugBank XML — фильтрация
    if args.drugbank_xml is not None:
        approved_df = load_approved_ligands(args.drugbank_xml)
        tsg_activating_approved = filter_approved(
            tsg_activating_drugs_df, approved_df
        )
        tsg_activating_approved_path = out_dir / "tsg_activating_drugs_approved.tsv"
        tsg_activating_approved.to_csv(
            tsg_activating_approved_path, sep="\t", index=False
        )
        print(
            f"Супрессоры с активирующими DrugBank-approved препаратами: "
            f"{len(tsg_activating_approved)} строк → {tsg_activating_approved_path}"
        )

        # для сортировки по аффинности (нМ)
        if not tsg_activating_approved.empty:
            df_sorted = tsg_activating_approved.copy()
            df_sorted["affinity_nM"] = df_sorted[
                "original_affinity_median_nm"
            ].apply(parse_affinity)
            df_sorted = df_sorted.sort_values(
                by=["affinity_nM", "tsg_symbol", "ligand_name"],
                ascending=[True, True, True],
            )
            sorted_path = out_dir / "tsg_activating_drugs_approved_sorted_by_affinity.tsv"
            df_sorted.to_csv(sorted_path, sep="\t", index=False)
            print(
                f"Супрессоры: approved-препараты, отсортированные по аффинности → "
                f"{sorted_path}"
            )

    # --- 4) Для онкогенов, блокирующих эти супрессоры, ищем блокирующие их лиганды ---
    oncogene_blocking_drugs_df = build_oncogene_blocking_drugs(
        oncogene_tsg_df=oncogene_tsg_df,
        tsg_activating_drugs_df=tsg_activating_drugs_df,
        interactions=interactions,
        human_only=human_only,
        primary_only=primary_only,
    )

    onco_drugs_path = out_dir / "oncogene_blocking_drugs.tsv"
    oncogene_blocking_drugs_df.to_csv(onco_drugs_path, sep="\t", index=False)
    print(
        f"Онкогены с блокирующими лигандами: {len(oncogene_blocking_drugs_df)} "
        f"строк → {onco_drugs_path}"
    )

    if args.drugbank_xml is not None:
        approved_df = load_approved_ligands(args.drugbank_xml)
        onco_blocking_approved = filter_approved(
            oncogene_blocking_drugs_df, approved_df
        )
        onco_blocking_approved_path = (
            out_dir / "oncogene_blocking_drugs_approved.tsv"
        )
        onco_blocking_approved.to_csv(
            onco_blocking_approved_path, sep="\t", index=False
        )
        print(
            f"Онкогены: блокирующие DrugBank-approved препараты: "
            f"{len(onco_blocking_approved)} строк → {onco_blocking_approved_path}"
        )

        if not onco_blocking_approved.empty:
            df_sorted = onco_blocking_approved.copy()
            df_sorted["affinity_nM"] = df_sorted[
                "original_affinity_median_nm"
            ].apply(parse_affinity)
            sorted_path = (
                out_dir / "oncogene_blocking_drugs_approved_sorted_by_affinity.tsv"
            )
            df_sorted = df_sorted.sort_values(
                by=["affinity_nM", "oncogene_symbol", "ligand_name"],
                ascending=[True, True, True],
            )
            df_sorted.to_csv(sorted_path, sep="\t", index=False)
            print(
                f"Онкогены: approved-блокаторы, отсортированные по аффинности → {sorted_path}"
            )

    # --- 5) Итоговые списки препаратов -------------------------------------
    # а) препараты-активаторы супрессоров
    activator_list_path = out_dir / "tsg_activator_drug_list.tsv"
    if not tsg_activating_drugs_df.empty:
        activator_list = (
            tsg_activating_drugs_df["ligand_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .sort_values()
            .to_frame(name="ligand_name")
        )
    else:
        activator_list = pd.DataFrame(columns=["ligand_name"])
    activator_list.to_csv(activator_list_path, sep="\t", index=False)
    print(
        f"Итоговый список активаторов супрессоров: {len(activator_list)} препаратов "
        f"→ {activator_list_path}"
    )

    # б) препараты-блокаторы онкогенов (связаны с супрессорами)
    blocker_list_path = out_dir / "oncogene_blocker_drug_list.tsv"
    if not oncogene_blocking_drugs_df.empty:
        blocker_list = (
            oncogene_blocking_drugs_df["ligand_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .sort_values()
            .to_frame(name="ligand_name")
        )
    else:
        blocker_list = pd.DataFrame(columns=["ligand_name"])
    blocker_list.to_csv(blocker_list_path, sep="\t", index=False)
    print(
        f"Итоговый список блокаторов онкогенов: {len(blocker_list)} препаратов "
        f"→ {blocker_list_path}"
    )

    # в) блокаторы любых онкогенов пациента
    all_onco_blocker_list_path = out_dir / "all_oncogene_blocker_drug_list.tsv"
    if not all_onco_blocking_drugs_df.empty:
        all_blocker_list = (
            all_onco_blocking_drugs_df["ligand_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .sort_values()
            .to_frame(name="ligand_name")
        )
    else:
        all_blocker_list = pd.DataFrame(columns=["ligand_name"])
    all_blocker_list.to_csv(all_onco_blocker_list_path, sep="\t", index=False)
    print(
        f"Итоговый список блокаторов любых онкогенов: "
        f"{len(all_blocker_list)} препаратов → {all_onco_blocker_list_path}"
    )

    print("\nГотово.\n")


if __name__ == "__main__":
    main()