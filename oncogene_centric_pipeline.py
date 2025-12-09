#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Онкоген-центричный пайплайн (через approved_drug_detailed_interactions, без DrugBank XML):

1) Для данных пациента ищем онкогены.
2) Для этих онкогенов ищем, какие супрессоры они блокируют (TRRUST, Repression).
3) Для найденных супрессоров ищем approved-препараты из GtoPdb,
   которые таргетируют эти супрессоры (approved_drug_detailed_interactions).
4) Для онкогенов, блокирующих какие-либо из этих супрессоров, ищем approved-препараты,
   таргетирующие онкогены (approved_drug_detailed_interactions).

⚠️ В этом пайплайне мы НЕ отличаем «активаторы» и «ингибиторы» по типу действия,
потому что в approved_drug_detailed_interactions нет отдельной колонки Action.
Все найденные препараты — просто препараты, таргетирующие данный ген.

Дополнительно:
- из RNA-seq берётся колонка log2FoldChange;
- полные таблицы и финальные списки препаратов сортируются по модулю log2FoldChange
  соответствующего гена:
    * для TSG — по |log2FoldChange(tsg_symbol)|
    * для онкогенов — по |log2FoldChange(oncogene_symbol)|

Входные файлы
-------------
1) RNA-seq: top_5000_RNA-seq_5_stat_.csv
   - разделитель: ';'
   - обязательные колонки: geneName, log2FoldChange (HGNC-символы)

2) cancerGeneList.tsv
   - минимум: "Hugo Symbol", "Gene Type"
   - онкогены: Gene Type ∈ {"ONCOGENE", "ONCOGENE_AND_TSG"} (можно расширить)
   - TSG: Gene Type ∈ {"TSG", "ONCOGENE_AND_TSG"}

3) TRRUST-файл (например Processed_TRRUST.tsv)
   - новый формат:
       Gene   – регулятор (TF)
       Target – мишень
       Effect – 'Activation' / 'Repression' / 'Unknown'
   - также поддерживаются старые RegNetwork-форматы (TF/Target/Regulation/PMID).
   - внутри приводится к колонкам: regulator, target, regulation, pmids

4) approved_drug_detailed_interactions.csv
   - Детализированный датасет approved-препаратов и их взаимодействий (GtoPdb).
   - первая строка — комментарий с версией, поэтому header=1.
   - ключевые колонки:
       "Ligand"
       "Ligand ID"
       "Type"
       "Clinical Use Comment"
       "Bioactivity Comment"
       "Target"
       "Target ID"
       "Target Gene Name"
       "Target Species"

Выходные файлы (в --out-dir)
----------------------------
1) patient_oncogenes.tsv
   - oncogene_symbol, gene_type

2) oncogene_tsg_repressions.tsv
   - oncogene_symbol, tsg_symbol, regulation, pmids
   (онкоген репрессирует TSG в TRRUST)

3) tsg_activating_drugs.tsv
   - на самом деле: супрессоры и approved-препараты, их таргетирующие
   - колонки:
       tsg_symbol,
       ligand_name, ligand_id, ligand_type,
       clinical_use_comment, bioactivity_comment,
       target_name, target_id, target_gene_symbol, target_species,
       log2FoldChange, abs_log2FoldChange
   - отсортировано по убыванию abs_log2FoldChange, затем tsg_symbol, ligand_name

4) tsg_activating_drugs_approved.tsv
   - совпадает с (3), оставлено для совместимости.

5) oncogene_blocking_drugs.tsv
   - онкогены (из п.2), для которых найдены таргетируемые TSG из п.3,
     и approved-препараты, таргетирующие эти онкогены
   - колонки:
       oncogene_symbol,
       ligand_name, ligand_id, ligand_type,
       clinical_use_comment, bioactivity_comment,
       target_name, target_id, target_gene_symbol, target_species,
       log2FoldChange, abs_log2FoldChange
   - отсортировано по убыванию abs_log2FoldChange, затем oncogene_symbol, ligand_name

6) oncogene_blocking_drugs_approved.tsv
   - совпадает с (5), оставлено для совместимости.

7) tsg_activator_drug_list.tsv
   - одна колонка ligand_name
   - уникальные препараты, таргетирующие супрессоры,
     отсортированы по убыванию максимального |log2FoldChange(tsg_symbol)|

8) oncogene_blocker_drug_list.tsv
   - одна колонка ligand_name
   - уникальные препараты, таргетирующие онкогены
     (связанные с супрессорами), отсортированы по убыванию
     максимального |log2FoldChange(oncogene_symbol)|

9) all_oncogene_blocking_drugs.tsv / all_oncogene_blocking_drugs_approved.tsv
   - все approved-препараты, таргетирующие любые онкогены пациента
     (независимо от связи с TSG)
   - колонки:
       oncogene_symbol,
       ligand_name, ligand_id, ligand_type,
       clinical_use_comment, bioactivity_comment,
       target_name, target_id, target_gene_symbol, target_species,
       log2FoldChange, abs_log2FoldChange
   - отсортировано по убыванию abs_log2FoldChange, затем oncogene_symbol, ligand_name

10) all_oncogene_blocker_drug_list.tsv
   - одна колонка ligand_name
   - список всех препаратов, таргетирующих любые онкогены,
     отсортированных по убыванию максимального |log2FoldChange(oncogene_symbol)|
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set, Tuple, Dict

import pandas as pd


# --- Общие вспомогательные функции -----------------------------------------


def load_patient_genes(rna_path: Path) -> tuple[Set[str], Dict[str, float]]:
    """
    Загрузка генов пациента из RNA-seq файла.

    Ожидается CSV с разделителем ';' и колонками:
      - geneName
      - log2FoldChange

    Возвращает:
      - множество генов пациента
      - словарь geneName -> log2FoldChange
    """
    df = pd.read_csv(rna_path, sep=";")

    required = {"geneName", "log2FoldChange"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"RNA-seq file {rna_path} must contain columns {required}, "
            f"missing: {missing}"
        )

    df = df.dropna(subset=["geneName"])
    df["geneName"] = df["geneName"].astype(str).str.strip()
    df = df[df["geneName"].str.upper() != "NA"]

    df = df.copy()
    df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")

    genes = set(df["geneName"])
    logfc_map: Dict[str, float] = (
        df.set_index("geneName")["log2FoldChange"].to_dict()
    )

    return genes, logfc_map


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
    Загрузка TRRUST-файла.

    Поддерживаются варианты:
      1) Новый Processed_TRRUST.tsv:
         колонки 'Gene', 'Target', 'Effect'.
         Маппинг:
             Gene   → regulator
             Target → target
             Effect → regulation
         pmids заполняется пустой строкой.

      2) Старые RegNetwork-форматы:
         TF, Target, Regulation, PMID (или Target gene / PubMed ID).
         На выходе всегда:
             regulator, target, regulation, pmids
    """
    cols_std = ["regulator", "target", "regulation", "pmids"]
    df = pd.read_csv(trrust_path, sep="\t")

    # Вариант 1: новый Processed_TRRUST.tsv
    if {"Gene", "Target", "Effect"}.issubset(df.columns):
        df = df.rename(
            columns={
                "Gene": "regulator",
                "Target": "target",
                "Effect": "regulation",
            }
        )
        df["pmids"] = ""
        df = df[cols_std].copy()
    else:
        # Вариант 2: старый RegNetwork или уже приведённый формат
        col_maps = [
            {"TF": "regulator", "Target": "target", "Regulation": "regulation", "PMID": "pmids"},
            {"TF": "regulator", "Target gene": "target", "Regulation": "regulation", "PMID": "pmids"},
            {"TF": "regulator", "Target": "target", "Regulation": "regulation", "PubMed ID": "pmids"},
            {"TF": "regulator", "Target gene": "target", "Regulation": "regulation", "PubMed ID": "pmids"},
        ]

        used_map = None
        for cmap in col_maps:
            if set(cmap.keys()).issubset(df.columns):
                df = df.rename(columns=cmap)
                used_map = cmap
                break

        if used_map is None:
            # Возможно, это старый raw-файл без заголовка (4 столбца)
            if df.shape[1] == 4 and set(df.columns) == {0, 1, 2, 3}:
                df = pd.read_csv(trrust_path, sep="\t", header=None, names=cols_std)
            else:
                existing = {}
                for c in cols_std:
                    if c in df.columns:
                        existing[c] = c
                df = df.rename(columns=existing)

        missing = [c for c in cols_std if c not in df.columns]
        if missing:
            raise ValueError(
                f"Не удалось привести TRRUST-файл {trrust_path} к стандартным колонкам. "
                f"Отсутствуют: {missing}. Найдены колонки: {list(df.columns)}"
            )

        df = df[cols_std].copy()

    # Нормализуем строки
    df["regulation"] = df["regulation"].astype(str).str.strip()
    df["regulator"] = df["regulator"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["pmids"] = df["pmids"].astype(str).str.strip()

    return df


# --- approved_drug_detailed_interactions → таргеты и препараты ------------


def load_approved_interactions(path: Path) -> pd.DataFrame:
    """
    Загрузка approved_drug_detailed_interactions.csv (на основе GtoPdb).

    Первая строка — комментарий → header=1.

    Возвращает DataFrame с ключевыми колонками и дополнительным полем:
      Target_Gene_upper (нормализованный HGNC).
    """
    df = pd.read_csv(path, header=1, low_memory=False)

    required_cols = [
        "Ligand",
        "Ligand ID",
        "Type",
        "Clinical Use Comment",
        "Bioactivity Comment",
        "Target",
        "Target ID",
        "Target Gene Name",
        "Target Species",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"В approved_drug_detailed_interactions нет колонок: {missing}\n"
            f"Найдено: {list(df.columns)}"
        )

    df = df.copy()
    df["Target_Gene_upper"] = (
        df["Target Gene Name"].astype(str).str.strip().str.upper()
    )

    return df


# --- Специфическая логика пайплайна ----------------------------------------


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
    logfc_map: Dict[str, float],
    human_only: bool = True,
) -> pd.DataFrame:
    """
    Для найденных супрессоров (tsg_symbol) ищем approved-препараты,
    таргетирующие эти TSG (по Target Gene Name).

    Фильтр:
      - Target Species == 'Human' (опционально).

    Возвращает таблицу с колонками:
      tsg_symbol,
      ligand_name, ligand_id, ligand_type,
      clinical_use_comment, bioactivity_comment,
      target_name, target_id, target_gene_symbol, target_species,
      log2FoldChange, abs_log2FoldChange.

    Отсортировано по убыванию abs_log2FoldChange, затем tsg_symbol, ligand_name.
    """
    cols = [
        "tsg_symbol",
        "ligand_name",
        "ligand_id",
        "ligand_type",
        "clinical_use_comment",
        "bioactivity_comment",
        "target_name",
        "target_id",
        "target_gene_symbol",
        "target_species",
        "log2FoldChange",
        "abs_log2FoldChange",
    ]

    if oncogene_tsg_df.empty or interactions.empty:
        return pd.DataFrame(columns=cols)

    tsgs_upper = set(
        oncogene_tsg_df["tsg_symbol"].astype(str).str.upper()
    )

    df_int = interactions.copy()
    if human_only:
        df_int = df_int[df_int["Target Species"] == "Human"]

    df_int = df_int[df_int["Target_Gene_upper"].isin(tsgs_upper)]

    if df_int.empty:
        return pd.DataFrame(columns=cols)

    df_int = df_int.copy()
    df_int["tsg_symbol"] = df_int["Target Gene Name"].astype(str).str.strip()

    result = df_int.assign(
        ligand_name=lambda d: d["Ligand"],
        ligand_id=lambda d: d["Ligand ID"],
        ligand_type=lambda d: d["Type"],
        clinical_use_comment=lambda d: d["Clinical Use Comment"],
        bioactivity_comment=lambda d: d["Bioactivity Comment"],
        target_name=lambda d: d["Target"],
        target_id=lambda d: d["Target ID"],
        target_gene_symbol=lambda d: d["Target Gene Name"],
        target_species=lambda d: d["Target Species"],
    )[
        [
            "tsg_symbol",
            "ligand_name",
            "ligand_id",
            "ligand_type",
            "clinical_use_comment",
            "bioactivity_comment",
            "target_name",
            "target_id",
            "target_gene_symbol",
            "target_species",
        ]
    ].drop_duplicates()

    # добавляем log2FoldChange и модуль
    result["log2FoldChange"] = result["tsg_symbol"].map(logfc_map)
    result["abs_log2FoldChange"] = result["log2FoldChange"].abs()

    return result.sort_values(
        ["abs_log2FoldChange", "tsg_symbol", "ligand_name"],
        ascending=[False, True, True],
    )


def build_oncogene_blocking_drugs(
    oncogene_tsg_df: pd.DataFrame,
    tsg_activating_drugs_df: pd.DataFrame,
    interactions: pd.DataFrame,
    logfc_map: Dict[str, float],
    human_only: bool = True,
) -> pd.DataFrame:
    """
    Для онкогенов, блокирующих супрессоры с найденными таргетируемыми препаратами,
    ищем approved-препараты, таргетирующие эти онкогены.

    - Берём TSG, для которых есть препараты (tsg_activating_drugs_df).
    - Смотрим, какие онкогены их репрессируют (oncogene_tsg_df).
    - Для таких онкогенов фильтруем interactions по:
        Target_Gene_upper ∈ онкогены
        (и опц. human_only).

    Возвращает таблицу:
      oncogene_symbol,
      ligand_name, ligand_id, ligand_type,
      clinical_use_comment, bioactivity_comment,
      target_name, target_id, target_gene_symbol, target_species,
      log2FoldChange, abs_log2FoldChange.

    Отсортировано по убыванию abs_log2FoldChange, затем oncogene_symbol, ligand_name.
    """
    cols = [
        "oncogene_symbol",
        "ligand_name",
        "ligand_id",
        "ligand_type",
        "clinical_use_comment",
        "bioactivity_comment",
        "target_name",
        "target_id",
        "target_gene_symbol",
        "target_species",
        "log2FoldChange",
        "abs_log2FoldChange",
    ]

    if (
        oncogene_tsg_df.empty
        or tsg_activating_drugs_df.empty
        or interactions.empty
    ):
        return pd.DataFrame(columns=cols)

    # TSG, для которых есть препараты
    tsg_with_drugs = set(
        tsg_activating_drugs_df["tsg_symbol"].astype(str).str.upper()
    )

    tmp = oncogene_tsg_df.copy()
    tmp["tsg_upper"] = tmp["tsg_symbol"].astype(str).str.upper()
    tmp["oncogene_upper"] = tmp["oncogene_symbol"].astype(str).str.upper()

    tmp = tmp[tmp["tsg_upper"].isin(tsg_with_drugs)]

    if tmp.empty:
        return pd.DataFrame(columns=cols)

    oncogenes_of_interest = set(tmp["oncogene_upper"])

    df_int = interactions.copy()
    if human_only:
        df_int = df_int[df_int["Target Species"] == "Human"]

    df_int = df_int[df_int["Target_Gene_upper"].isin(oncogenes_of_interest)]

    if df_int.empty:
        return pd.DataFrame(columns=cols)

    df_int = df_int.copy()
    df_int["oncogene_symbol"] = df_int["Target Gene Name"].astype(str).str.strip()

    result = df_int.assign(
        ligand_name=lambda d: d["Ligand"],
        ligand_id=lambda d: d["Ligand ID"],
        ligand_type=lambda d: d["Type"],
        clinical_use_comment=lambda d: d["Clinical Use Comment"],
        bioactivity_comment=lambda d: d["Bioactivity Comment"],
        target_name=lambda d: d["Target"],
        target_id=lambda d: d["Target ID"],
        target_gene_symbol=lambda d: d["Target Gene Name"],
        target_species=lambda d: d["Target Species"],
    )[
        [
            "oncogene_symbol",
            "ligand_name",
            "ligand_id",
            "ligand_type",
            "clinical_use_comment",
            "bioactivity_comment",
            "target_name",
            "target_id",
            "target_gene_symbol",
            "target_species",
        ]
    ].drop_duplicates()

    result["log2FoldChange"] = result["oncogene_symbol"].map(logfc_map)
    result["abs_log2FoldChange"] = result["log2FoldChange"].abs()

    return result.sort_values(
        ["abs_log2FoldChange", "oncogene_symbol", "ligand_name"],
        ascending=[False, True, True],
    )


def build_all_oncogene_blocking_drugs(
    patient_oncogenes: Set[str],
    interactions: pd.DataFrame,
    logfc_map: Dict[str, float],
    human_only: bool = True,
) -> pd.DataFrame:
    """
    Ищем approved-препараты, таргетирующие ЛЮБЫЕ онкогены пациента,
    независимо от связи с TSG.

    - Target_Gene_upper ∈ patient_oncogenes

    Возвращает:
      oncogene_symbol,
      ligand_name, ligand_id, ligand_type,
      clinical_use_comment, bioactivity_comment,
      target_name, target_id, target_gene_symbol, target_species,
      log2FoldChange, abs_log2FoldChange.

    Отсортировано по убыванию abs_log2FoldChange, затем oncogene_symbol, ligand_name.
    """
    cols = [
        "oncogene_symbol",
        "ligand_name",
        "ligand_id",
        "ligand_type",
        "clinical_use_comment",
        "bioactivity_comment",
        "target_name",
        "target_id",
        "target_gene_symbol",
        "target_species",
        "log2FoldChange",
        "abs_log2FoldChange",
    ]

    if not patient_oncogenes or interactions.empty:
        return pd.DataFrame(columns=cols)

    onco_upper = {g.upper() for g in patient_oncogenes}

    df_int = interactions.copy()
    if human_only:
        df_int = df_int[df_int["Target Species"] == "Human"]

    df_int = df_int[df_int["Target_Gene_upper"].isin(onco_upper)]

    if df_int.empty:
        return pd.DataFrame(columns=cols)

    df_int = df_int.copy()
    df_int["oncogene_symbol"] = df_int["Target Gene Name"].astype(str).str.strip()

    result = df_int.assign(
        ligand_name=lambda d: d["Ligand"],
        ligand_id=lambda d: d["Ligand ID"],
        ligand_type=lambda d: d["Type"],
        clinical_use_comment=lambda d: d["Clinical Use Comment"],
        bioactivity_comment=lambda d: d["Bioactivity Comment"],
        target_name=lambda d: d["Target"],
        target_id=lambda d: d["Target ID"],
        target_gene_symbol=lambda d: d["Target Gene Name"],
        target_species=lambda d: d["Target Species"],
    )[
        [
            "oncogene_symbol",
            "ligand_name",
            "ligand_id",
            "ligand_type",
            "clinical_use_comment",
            "bioactivity_comment",
            "target_name",
            "target_id",
            "target_gene_symbol",
            "target_species",
        ]
    ].drop_duplicates()

    result["log2FoldChange"] = result["oncogene_symbol"].map(logfc_map)
    result["abs_log2FoldChange"] = result["log2FoldChange"].abs()

    return result.sort_values(
        ["abs_log2FoldChange", "oncogene_symbol", "ligand_name"],
        ascending=[False, True, True],
    )


# --- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Онкоген-центричный пайплайн (approved_drug_detailed_interactions): "
            "1) онкогены пациента → 2) супрессоры, которые они блокируют (TRRUST) → "
            "3) approved-препараты для супрессоров → "
            "4) approved-препараты для онкогенов, с сортировкой по |log2FoldChange|."
        )
    )

    parser.add_argument(
        "--rna",
        type=Path,
        required=True,
        help="Путь к RNA-seq CSV (top_5000_RNA-seq_5_stat_.csv).",
    )
    parser.add_argument(
        "--cancer-list",
        type=Path,
        required=True,
        help="Путь к cancerGeneList.tsv.",
    )
    parser.add_argument(
        "--trrust",
        type=Path,
        required=True,
        help="Путь к TRRUST-файлу (например, Processed_TRRUST.tsv).",
    )
    parser.add_argument(
        "--approved-interactions",
        type=Path,
        required=True,
        help="Путь к approved_drug_detailed_interactions.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Каталог для записи результатов (по умолчанию: текущий).",
    )
    parser.add_argument(
        "--no-human-filter",
        action="store_true",
        help="Не ограничивать Target Species == 'Human'.",
    )

    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    human_only = not args.no_human_filter

    # --- 1) Гены пациента, онкогены и TSG -----------------------------------
    patient_genes, rna_logfc = load_patient_genes(args.rna)

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
    print(
        f"Онкогены пациента записаны в: {onco_path} "
        f"({len(onco_df)} генов)"
    )

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

    # --- 3) approved_drug_detailed_interactions ----------------------------
    interactions = load_approved_interactions(args.approved_interactions)

    n_drugs = interactions["Ligand"].nunique()
    print(
        f"Из approved_drug_detailed_interactions извлечено записей: {len(interactions)}, "
        f"уникальных препаратов (Ligand): {n_drugs}"
    )

    # --- 3a) ЛЮБЫЕ препараты для всех онкогенов пациента -------------------
    all_onco_blocking_drugs_df = build_all_oncogene_blocking_drugs(
        patient_oncogenes=set(patient_oncogenes),
        interactions=interactions,
        logfc_map=rna_logfc,
        human_only=human_only,
    )

    all_onco_drugs_path = out_dir / "all_oncogene_blocking_drugs.tsv"
    all_onco_blocking_drugs_df.to_csv(all_onco_drugs_path, sep="\t", index=False)
    print(
        f"Препараты, таргетирующие любые онкогены пациента: "
        f"{len(all_onco_blocking_drugs_df)} строк → {all_onco_drugs_path}"
    )

    # approved-only: файл с суффиксом _approved оставляем для совместимости
    all_onco_blocking_approved_path = (
        out_dir / "all_oncogene_blocking_drugs_approved.tsv"
    )
    all_onco_blocking_drugs_df.to_csv(
        all_onco_blocking_approved_path, sep="\t", index=False
    )
    print(
        f"(Approved) препараты, таргетирующие любые онкогены пациента: "
        f"{len(all_onco_blocking_drugs_df)} строк → "
        f"{all_onco_blocking_approved_path}"
    )

    # --- 4) Супрессоры с препаратами ---------------------------------------
    tsg_activating_drugs_df = build_tsg_activating_drugs(
        oncogene_tsg_df=oncogene_tsg_df,
        interactions=interactions,
        logfc_map=rna_logfc,
        human_only=human_only,
    )

    tsg_drugs_path = out_dir / "tsg_activating_drugs.tsv"
    tsg_activating_drugs_df.to_csv(tsg_drugs_path, sep="\t", index=False)
    print(
        f"Супрессоры с препаратами: "
        f"{len(tsg_activating_drugs_df)} строк → {tsg_drugs_path}"
    )

    tsg_activating_approved_path = out_dir / "tsg_activating_drugs_approved.tsv"
    tsg_activating_drugs_df.to_csv(
        tsg_activating_approved_path, sep="\t", index=False
    )
    print(
        f"(Approved) супрессоры с препаратами: "
        f"{len(tsg_activating_drugs_df)} строк → "
        f"{tsg_activating_approved_path}"
    )

    # --- 5) Онкогены с препаратами (через TSG) -----------------------------
    oncogene_blocking_drugs_df = build_oncogene_blocking_drugs(
        oncogene_tsg_df=oncogene_tsg_df,
        tsg_activating_drugs_df=tsg_activating_drugs_df,
        interactions=interactions,
        logfc_map=rna_logfc,
        human_only=human_only,
    )

    onco_drugs_path = out_dir / "oncogene_blocking_drugs.tsv"
    oncogene_blocking_drugs_df.to_csv(onco_drugs_path, sep="\t", index=False)
    print(
        f"Онкогены с препаратами: "
        f"{len(oncogene_blocking_drugs_df)} строк → {onco_drugs_path}"
    )

    onco_blocking_approved_path = (
        out_dir / "oncogene_blocking_drugs_approved.tsv"
    )
    oncogene_blocking_drugs_df.to_csv(
        onco_blocking_approved_path, sep="\t", index=False
    )
    print(
        f"(Approved) онкогены с препаратами: "
        f"{len(oncogene_blocking_drugs_df)} строк → "
        f"{onco_blocking_approved_path}"
    )

    # --- 6) Итоговые списки препаратов с сортировкой по |log2FoldChange| ---

    # a) препараты для супрессоров
    activator_list_path = out_dir / "tsg_activator_drug_list.tsv"
    if not tsg_activating_drugs_df.empty:
        tmp = tsg_activating_drugs_df.copy()
        if "abs_log2FoldChange" not in tmp.columns:
            tmp["log2FoldChange"] = tmp["tsg_symbol"].map(rna_logfc)
            tmp["abs_log2FoldChange"] = tmp["log2FoldChange"].abs()

        activator_list = (
            tmp.groupby("ligand_name", as_index=False)
               .agg(max_abs_log2FoldChange=("abs_log2FoldChange", "max"))
               .sort_values("max_abs_log2FoldChange", ascending=False)
        )
        activator_list = activator_list[["ligand_name"]]
    else:
        activator_list = pd.DataFrame(columns=["ligand_name"])
    activator_list.to_csv(activator_list_path, sep="\t", index=False)
    print(
        f"Итоговый список препаратов для супрессоров (отсортирован по |log2FoldChange(TSG)|): "
        f"{len(activator_list)} препаратов → {activator_list_path}"
    )

    # б) препараты для онкогенов (связанных с супрессорами)
    blocker_list_path = out_dir / "oncogene_blocker_drug_list.tsv"
    if not oncogene_blocking_drugs_df.empty:
        tmp = oncogene_blocking_drugs_df.copy()
        if "abs_log2FoldChange" not in tmp.columns:
            tmp["log2FoldChange"] = tmp["oncogene_symbol"].map(rna_logfc)
            tmp["abs_log2FoldChange"] = tmp["log2FoldChange"].abs()

        blocker_list = (
            tmp.groupby("ligand_name", as_index=False)
               .agg(max_abs_log2FoldChange=("abs_log2FoldChange", "max"))
               .sort_values("max_abs_log2FoldChange", ascending=False)
        )
        blocker_list = blocker_list[["ligand_name"]]
    else:
        blocker_list = pd.DataFrame(columns=["ligand_name"])
    blocker_list.to_csv(blocker_list_path, sep="\t", index=False)
    print(
        f"Итоговый список препаратов для онкогенов (через TSG, отсортирован по |log2FoldChange(онкогена)|): "
        f"{len(blocker_list)} препаратов → {blocker_list_path}"
    )

    # в) все препараты для любых онкогенов пациента
    all_onco_blocker_list_path = out_dir / "all_oncogene_blocker_drug_list.tsv"
    if not all_onco_blocking_drugs_df.empty:
        tmp = all_onco_blocking_drugs_df.copy()
        if "abs_log2FoldChange" not in tmp.columns:
            tmp["log2FoldChange"] = tmp["oncogene_symbol"].map(rna_logfc)
            tmp["abs_log2FoldChange"] = tmp["log2FoldChange"].abs()

        all_blocker_list = (
            tmp.groupby("ligand_name", as_index=False)
               .agg(max_abs_log2FoldChange=("abs_log2FoldChange", "max"))
               .sort_values("max_abs_log2FoldChange", ascending=False)
        )
        all_blocker_list = all_blocker_list[["ligand_name"]]
    else:
        all_blocker_list = pd.DataFrame(columns=["ligand_name"])
    all_blocker_list.to_csv(all_onco_blocker_list_path, sep="\t", index=False)
    print(
        f"Итоговый список препаратов для любых онкогенов (отсортирован по |log2FoldChange(онкогена)|): "
        f"{len(all_blocker_list)} препаратов → {all_onco_blocker_list_path}"
    )

    print("\nГотово.\n")


if __name__ == "__main__":
    main()