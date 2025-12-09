#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Единый пайплайн:

RNA-seq → TSG (cancerGeneList) → TRRUST (ингибиторы) → GtoPdb (interactions)
→ DrugBank full database.xml (approved drugs) → сортировка по аффинности.

Входные файлы
-------------
1) RNA-seq: top_5000_RNA-seq_5_stat_.csv
   - Разделитель: ';'
   - Обязательная колонка: geneName (HGNC-символы).
   - "NA" и пустые строки отбрасываются.

2) cancerGeneList.tsv
   - минимально: колонки "Hugo Symbol", "Gene Type"
   - TSG определяются как Gene Type ∈ {"TSG", "ONCOGENE_AND_TSG"}.

3) trrust_rawdata.human.tsv
   - 4 таб-разделённых столбца:
     0: TF (регулятор)
     1: Target gene (мишень)
     2: Regulation ("Activation", "Repression", "Unknown")
     3: PubMed IDs (через запятую)

4) interactions.csv
   - GtoPdb Interactions Dataset
   - первая строка — комментарий с версией, поэтому header=1
   - ключевые колонки:
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

5) full database.xml
   - DrugBank full database
   - Используем:
       <drugbank-id>      → drugbank_id
       <name>             → drugbank_name
       <groups><group>    → drugbank_groups (множество статусов)
   - Препарат считается "одобренным", если среди group есть "approved"
     (регистр не важен).

Выходные файлы (в --out-dir)
----------------------------
1) patient_tsgs.tsv
   - tsg_symbol, gene_type

2) tsg_regulators_all.tsv
   - regulator, tsg_symbol, regulation, pmids

3) tsg_activators.tsv
   - regulator, tsg_symbol, regulation, pmids (Activation)

4) tsg_inhibitors.tsv
   - regulator, tsg_symbol, regulation, pmids (Repression)

5) tsg_inhibitor_drugs.tsv
   - подробная таблица:
     tsg_symbol, inhibitor_gene, regulation_type, trrust_pmids,
     target_name, target_id, target_gene_symbol,
     ligand_name, ligand_id, ligand_type,
     affinity_median, affinity_units,
     original_affinity_median_nm, original_affinity_relation,
     interaction_pubmed_id, action

   (по умолчанию: только Human, Primary Target == True, Action ∈ {inhibitor, antagonist, blocker})

6) tsg_inhibitor_drugs_approved.tsv
   - подтаблица из (5), где ligand_name матчится на approved-препараты
     из DrugBank (по name, без учёта регистра).

   + дополнительные колонки:
     drugbank_id, drugbank_name, drugbank_groups

7) tsg_inhibitor_drugs_approved_sorted_by_affinity.tsv  ← ОСНОВНОЙ РЕЗУЛЬТИРУЮЩИЙ ФАЙЛ
   - (6), плюс колонка affinity_nM, отсортированная по возрастанию (сильнейшее связывание).

8) tsg_inhibitor_drug_list.tsv  ← ИТОГОВЫЙ СПИСОК ЛЕКАРСТВ
   - одна колонка: ligand_name
   - уникальные названия препаратов из (7), без дополнительных колонок.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set, Tuple

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# --- Часть 1. RNA-seq → TSG → TRRUST ---------------------------------------


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
    # Убираем буквальное "NA", если оно есть
    genes = genes[genes.str.upper() != "NA"]

    return set(genes)


def load_tsgs(
    cancer_path: Path,
    tsg_types: Iterable[str] = ("TSG", "ONCOGENE_AND_TSG"),
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Загрузка онкосупрессоров из cancerGeneList.tsv.

    Parameters
    ----------
    cancer_path : Path
        Путь к cancerGeneList.tsv.
    tsg_types : iterable of str
        Какие значения столбца 'Gene Type' считать TSG.

    Returns
    -------
    tsg_symbols : set of str
        Набор символов генов-онкосупрессоров.
    tsg_table : DataFrame
        Подтаблица только по TSG-генам.
    """
    df = pd.read_csv(cancer_path, sep="\t")

    required_cols = {"Hugo Symbol", "Gene Type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Columns {missing} not found in cancer gene list {cancer_path}"
        )

    mask = df["Gene Type"].isin(list(tsg_types))
    tsg_table = df.loc[mask].copy()

    tsg_symbols = (
        tsg_table["Hugo Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return set(tsg_symbols), tsg_table


def load_trrust(trrust_path: Path) -> pd.DataFrame:
    """
    Загрузка TRRUST (human).

    Формат: 4 таб-разделённых столбца:
        TF, Target, Regulation, PubMed IDs
    """
    cols = ["regulator", "target", "regulation", "pmids"]
    df = pd.read_csv(trrust_path, sep="\t", header=None, names=cols)

    # Нормализуем строки
    df["regulation"] = df["regulation"].astype(str).str.strip()
    df["regulator"] = df["regulator"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    return df


def run_trrust_pipeline(
    rna: Path,
    cancer_list: Path,
    trrust_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Выполнить шаги:
    - найти TSG у пациента
    - найти их регуляторов в TRRUST
    - разложить на активаторов и ингибиторов
    - всё сохранить в out_dir

    Возвращает DataFrame inhibitors (колонки: regulator, tsg_symbol, regulation, pmids).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Гены пациента
    patient_genes = load_patient_genes(rna)
    if not patient_genes:
        raise RuntimeError("Не найдено ни одного гена в RNA-seq файле.")

    # 2) TSG из cancerGeneList
    all_tsg_symbols, tsg_table = load_tsgs(cancer_list)

    # Пересечение: TSG, которые реально есть у пациента
    patient_tsgs = sorted(all_tsg_symbols.intersection(patient_genes))

    if not patient_tsgs:
        print(
            "В списке генов пациента отсутствуют онкосупрессоры "
            "из cancerGeneList.tsv."
        )
    else:
        print(
            f"Найдено {len(patient_tsgs)} онкосупрессоров "
            f"в RNA-seq пациента."
        )

    # 3) Сохраняем список TSG пациента
    if patient_tsgs:
        tsg_df = (
            tsg_table[tsg_table["Hugo Symbol"].isin(patient_tsgs)]
            .loc[:, ["Hugo Symbol", "Gene Type"]]
            .rename(
                columns={
                    "Hugo Symbol": "tsg_symbol",
                    "Gene Type": "gene_type",
                }
            )
            .drop_duplicates()
            .sort_values("tsg_symbol")
        )
    else:
        tsg_df = pd.DataFrame(columns=["tsg_symbol", "gene_type"])

    tsg_list_path = out_dir / "patient_tsgs.tsv"
    tsg_df.to_csv(tsg_list_path, sep="\t", index=False)
    print(f"Список онкосупрессоров пациента записан в: {tsg_list_path}")

    # 4) TRRUST: регуляторы для этих TSG
    trrust_df = load_trrust(trrust_path)

    if patient_tsgs:
        regulators_df = trrust_df[trrust_df["target"].isin(patient_tsgs)].copy()
    else:
        regulators_df = trrust_df.iloc[0:0].copy()

    # Переименуем target -> tsg_symbol и сохраним полный список
    all_reg_path = out_dir / "tsg_regulators_all.tsv"
    if not regulators_df.empty:
        regulators_df = regulators_df.rename(columns={"target": "tsg_symbol"})
        regulators_df.to_csv(all_reg_path, sep="\t", index=False)
    else:
        pd.DataFrame(
            columns=["regulator", "tsg_symbol", "regulation", "pmids"]
        ).to_csv(all_reg_path, sep="\t", index=False)

    print(
        "Все регуляторы TSG (Activation/Repression/Unknown) "
        f"записаны в: {all_reg_path}"
    )

    # 5) Делим на активаторов и ингибиторов
    if not regulators_df.empty:
        activators = regulators_df[
            regulators_df["regulation"].str.lower() == "activation"
        ].copy()
        inhibitors = regulators_df[
            regulators_df["regulation"].str.lower() == "repression"
        ].copy()
    else:
        activators = pd.DataFrame(
            columns=["regulator", "tsg_symbol", "regulation", "pmids"]
        )
        inhibitors = activators.copy()

    act_path = out_dir / "tsg_activators.tsv"
    inh_path = out_dir / "tsg_inhibitors.tsv"

    activators.to_csv(act_path, sep="\t", index=False)
    inhibitors.to_csv(inh_path, sep="\t", index=False)

    print(f"Активаторы TSG записаны в:  {act_path}")
    print(f"Ингибиторы TSG записаны в: {inh_path}")

    return inhibitors  # для дальнейшего шага TSG → ингибитор → препараты


# --- Часть 2. TSG ингибиторы → GtoPdb interactions -------------------------


# Какие действия считаем подавляющими активность ингибитора TSG
INHIBITING_ACTIONS = {"INHIBITOR", "ANTAGONIST", "BLOCKER"}


def prepare_tsg_inhibitors_for_gtopdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовить tsg_inhibitors к маппингу с GtoPdb:
    - добавить колонку regulator_upper.
    """
    required = {"regulator", "tsg_symbol", "regulation", "pmids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"В tsg_inhibitors отсутствуют колонки {missing}. "
            f"Найдены: {list(df.columns)}"
        )

    df = df.copy()
    df["regulator_upper"] = (
        df["regulator"].astype(str).str.strip().str.upper()
    )
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


def build_tsg_inhibitor_drug_table(
    tsg_inhibitors: pd.DataFrame,
    interactions: pd.DataFrame,
    human_only: bool = True,
    primary_only: bool = True,
    inhibiting_actions_only: bool = True,
) -> pd.DataFrame:
    """
    Мержим:
    TSG (мишень) ← ингибитор (регулятор из TRRUST) ← GtoPdb (interactions)
    и выбираем только нужные Action / вид / Primary Target.

    Возвращает таблицу с колонками:
    tsg_symbol, inhibitor_gene, regulation_type, trrust_pmids,
    target_name, target_id, target_gene_symbol,
    ligand_name, ligand_id, ligand_type,
    affinity_median, affinity_units,
    original_affinity_median_nm, original_affinity_relation,
    interaction_pubmed_id, action.
    """
    df_int = interactions.copy()

    if human_only:
        df_int = df_int[df_int["Target Species"] == "Human"]

    if primary_only:
        df_int = df_int[df_int["Primary Target"] == True]

    if inhibiting_actions_only:
        df_int = df_int[df_int["Action_upper"].isin(INHIBITING_ACTIONS)]

    # соединяем TSG ингибиторы → транс-фактор → препараты
    merged = tsg_inhibitors.merge(
        df_int,
        left_on="regulator_upper",
        right_on="Target_Gene_upper",
        how="inner",
    )

    if merged.empty:
        return merged

    out_cols = {
        "tsg_symbol": "tsg_symbol",
        "regulator": "inhibitor_gene",
        "regulation": "regulation_type",
        "pmids": "trrust_pmids",

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

    result = merged[list(out_cols.keys())].rename(columns=out_cols)

    # вернём отсортированный по TSG / ингибитору / лиганду df
    return result.sort_values(["tsg_symbol", "inhibitor_gene", "ligand_name"])


# --- Часть 3. DrugBank XML → approved препараты и фильтрация ---------------


def load_approved_ligands(path: Path) -> pd.DataFrame:
    """
    Загрузка DrugBank full database XML.

    Извлекаем:
      - drugbank_id  (primary=true, если есть, иначе первый <drugbank-id>)
      - name         (основное имя препарата)
      - groups       (список group, объединённый через ';')
      - name_norm    (NAME в верхнем регистре, для матчинга)
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
            drugbank_id = first_id.text.strip() if first_id is not None and first_id.text else None

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
    tsg_drugs: pd.DataFrame,
    approved_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Оставляем только те строки из tsg_drugs, чьи ligand_name присутствуют
    среди approved-препаратов в DrugBank (по имени, без регистра).

    Возвращает tsg_drugs + добавленные колонки:
      - drugbank_id
      - drugbank_name
      - drugbank_groups
    """
    approved_only = approved_df[approved_df["is_approved"]].copy()

    if approved_only.empty:
        # Нет ни одного approved-препарата → всё пусто
        return tsg_drugs.iloc[0:0].copy()

    # Оставляем уникальные по нормализованному имени
    approved_unique = (
        approved_only
        .loc[:, ["drugbank_id", "name", "drugbank_name_norm", "drugbank_groups"]]
        .drop_duplicates(subset=["drugbank_name_norm"])
    )

    tsg = tsg_drugs.copy()
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

    # Переименуем для явности
    merged = merged.rename(
        columns={
            "name": "drugbank_name",
        }
    )

    # Временные тех.колонки больше не нужны
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


# --- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Полный пайплайн: RNA-seq → TSG → TRRUST ингибиторы → "
            "GtoPdb interactions → DrugBank approved → сортировка по аффинности."
        )
    )

    # Вход для TRRUST-пайплайна
    parser.add_argument("--rna", type=Path, required=True,
                        help="Путь к RNA-seq CSV (top_5000_RNA-seq_5_stat_.csv).")
    parser.add_argument("--cancer-list", type=Path, required=True,
                        help="Путь к cancerGeneList.tsv.")
    parser.add_argument("--trrust", type=Path, required=True,
                        help="Путь к trrust_rawdata.human.tsv.")

    # GtoPdb interactions
    parser.add_argument("--interactions", type=Path, required=True,
                        help="Путь к GtoPdb interactions.csv.")

    # DrugBank XML (поддерживаем старое имя параметра как алиас)
    parser.add_argument(
        "--drugbank-xml",
        "--approved-interactions",
        dest="drugbank_xml",
        type=Path,
        required=True,
        help="Путь к DrugBank full database XML (full database.xml).",
    )

    # Общий каталог вывода
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Каталог для записи всех результатов (по умолчанию: текущий).",
    )

    # Флаги фильтрации GtoPdb
    parser.add_argument("--no-human-filter", action="store_true",
                        help="Не ограничивать Target Species == 'Human'.")
    parser.add_argument("--no-primary-only", action="store_true",
                        help="Не ограничивать Primary Target == True.")
    parser.add_argument(
        "--include-all-actions",
        action="store_true",
        help=(
            "Не ограничиваться только Action ∈ {inhibitor, antagonist, blocker}; "
            "включить все действия."
        ),
    )

    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Шаг 1. RNA-seq → TSG → TRRUST ингибиторы ===
    inhibitors_raw = run_trrust_pipeline(
        rna=args.rna,
        cancer_list=args.cancer_list,
        trrust_path=args.trrust,
        out_dir=out_dir,
    )

    tsg_inhibitors = prepare_tsg_inhibitors_for_gtopdb(inhibitors_raw)

    # === Шаг 2. TSG ингибиторы → GtoPdb interactions ===
    interactions = load_interactions(args.interactions)

    human_only = not args.no_human_filter
    primary_only = not args.no_primary_only
    inhibiting_actions_only = not args.include_all_actions

    tsg_drugs_full = build_tsg_inhibitor_drug_table(
        tsg_inhibitors=tsg_inhibitors,
        interactions=interactions,
        human_only=human_only,
        primary_only=primary_only,
        inhibiting_actions_only=inhibiting_actions_only,
    )

    tsg_drugs_path = out_dir / "tsg_inhibitor_drugs.tsv"
    if tsg_drugs_full.empty:
        tsg_drugs_full.to_csv(tsg_drugs_path, sep="\t", index=False)
        print(
            "\n❗ Не найдено ни одного препарата "
            "с заданными фильтрами в interactions.csv.\n"
        )
        print(f"Пустой файл записан в: {tsg_drugs_path}")
        return

    tsg_drugs_full.to_csv(tsg_drugs_path, sep="\t", index=False)
    print(f"\n✔️ Найдено строк (TSG → ингибитор → препарат): {len(tsg_drugs_full)}")
    print(f"Полная таблица записана в: {tsg_drugs_path}")

    # === Шаг 3. Фильтрация по DrugBank approved ===
    approved_df = load_approved_ligands(args.drugbank_xml)

    n_total = len(approved_df)
    n_approved = int(approved_df["is_approved"].sum())

    print(
        f"Из DrugBank XML извлечено препаратов: {n_total}, "
        f"из них с группой 'approved': {n_approved}"
    )

    tsg_drugs_approved = filter_approved(tsg_drugs_full, approved_df)

    approved_path = out_dir / "tsg_inhibitor_drugs_approved.tsv"
    if tsg_drugs_approved.empty:
        tsg_drugs_approved.to_csv(approved_path, sep="\t", index=False)
        print(
            "❗ Не найдено ни одного пересечения между ligand_name "
            "и approved-препаратами из DrugBank."
        )
        print(f"Пустой файл записан в: {approved_path}")
        return

    tsg_drugs_approved.to_csv(approved_path, sep="\t", index=False)
    print(
        f"Клинически одобренные (по DrugBank 'approved') препараты: "
        f"{len(tsg_drugs_approved)} строк → {approved_path}"
    )

    # === Шаг 4. Сортировка по аффинности ===
    sorted_path = (
        out_dir / "tsg_inhibitor_drugs_approved_sorted_by_affinity.tsv"
    )

    df_sorted = tsg_drugs_approved.copy()
    df_sorted["affinity_nM"] = df_sorted[
        "original_affinity_median_nm"
    ].apply(parse_affinity)

    df_sorted = df_sorted.sort_values(
        by=["affinity_nM", "tsg_symbol", "inhibitor_gene", "ligand_name"],
        ascending=[True, True, True, True],
    )

    df_sorted.to_csv(sorted_path, sep="\t", index=False)
    print(
        f"Отсортировано по аффинности (сильнейшее связывание сначала): "
        f"{len(df_sorted)} строк → {sorted_path}"
    )

    # === Шаг 5. Итоговый список препаратов (только названия) ===
    drug_list_path = out_dir / "tsg_inhibitor_drug_list.tsv"

    drug_list = (
        df_sorted["ligand_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .to_frame(name="ligand_name")
    )

    drug_list.to_csv(drug_list_path, sep="\t", index=False)
    print(
        f"\nИтоговый список препаратов (уникальные ligand_name, без доп. колонок): "
        f"{len(drug_list)} строк → {drug_list_path}"
    )

    print("\nГотово.\n")


if __name__ == "__main__":
    main()