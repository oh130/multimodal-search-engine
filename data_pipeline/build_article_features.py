import csv
import logging
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "articles.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "articles_feature.csv"
OUTPUT_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "category",
    "main_category",
    "color",
]

UNKNOWN_VALUE = "UNKNOWN"
INPUT_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "index_name",
    "index_group_name",
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def validate_required_columns(file_path: Path, required_columns: Sequence[str]) -> None:
    with file_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []

    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing_columns)}"
        )


def normalize_text(value: str) -> str:
    normalized = " ".join((value or "").strip().split())
    return normalized if normalized else UNKNOWN_VALUE


def normalize_main_category(row: dict[str, str]) -> str:
    for column in ("index_group_name", "index_name", "product_group_name"):
        value = normalize_text(row.get(column, ""))
        if value != UNKNOWN_VALUE:
            return value
    return UNKNOWN_VALUE


def normalize_category(row: dict[str, str]) -> str:
    for column in ("product_type_name", "product_group_name", "department_name"):
        value = normalize_text(row.get(column, ""))
        if value != UNKNOWN_VALUE:
            return value
    return UNKNOWN_VALUE


def normalize_color(row: dict[str, str]) -> str:
    for column in ("perceived_colour_master_name", "colour_group_name"):
        value = normalize_text(row.get(column, ""))
        if value != UNKNOWN_VALUE:
            return value
    return UNKNOWN_VALUE


def main() -> None:
    configure_logging()
    validate_required_columns(INPUT_FILE, INPUT_COLUMNS)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    unknown_main_category_rows = 0
    unknown_category_rows = 0
    unknown_color_rows = 0

    with INPUT_FILE.open(newline="", encoding="utf-8") as infile, OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in reader:
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                continue

            category = normalize_category(row)
            main_category = normalize_main_category(row)
            color = normalize_color(row)

            if main_category == UNKNOWN_VALUE:
                unknown_main_category_rows += 1
            if category == UNKNOWN_VALUE:
                unknown_category_rows += 1
            if color == UNKNOWN_VALUE:
                unknown_color_rows += 1

            writer.writerow(
                {
                    "article_id": article_id,
                    "prod_name": normalize_text(row.get("prod_name", "")),
                    "product_type_name": normalize_text(row.get("product_type_name", "")),
                    "product_group_name": normalize_text(row.get("product_group_name", "")),
                    "colour_group_name": normalize_text(row.get("colour_group_name", "")),
                    "perceived_colour_master_name": normalize_text(row.get("perceived_colour_master_name", "")),
                    "department_name": normalize_text(row.get("department_name", "")),
                    "section_name": normalize_text(row.get("section_name", "")),
                    "garment_group_name": normalize_text(row.get("garment_group_name", "")),
                    "category": category,
                    "main_category": main_category,
                    "color": color,
                }
            )
            rows_written += 1

    logging.info(
        "articles_feature_complete output=%s rows_written=%s unknown_main_category_rows=%s unknown_category_rows=%s unknown_color_rows=%s",
        OUTPUT_FILE,
        rows_written,
        unknown_main_category_rows,
        unknown_category_rows,
        unknown_color_rows,
    )


if __name__ == "__main__":
    main()
