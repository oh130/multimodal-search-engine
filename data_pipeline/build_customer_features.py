import csv
import logging
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "customers.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "customer_features.csv"
OUTPUT_COLUMNS = [
    "customer_id",
    "age",
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
]

UNKNOWN_VALUE = "UNKNOWN"
INPUT_COLUMNS = [
    "customer_id",
    "age",
    "fashion_news_frequency",
    "club_member_status",
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


def normalize_fashion_news_frequency(value: str) -> str:
    normalized = normalize_text(value).upper()
    aliases = {
        "NONE": "NONE",
        "REGULARLY": "REGULARLY",
        "MONTHLY": "MONTHLY",
    }
    return aliases.get(normalized, UNKNOWN_VALUE)


def normalize_club_member_status(value: str) -> str:
    normalized = normalize_text(value).upper()
    aliases = {
        "ACTIVE": "ACTIVE",
        "PRE-CREATE": "PRE-CREATE",
        "LEFT CLUB": "LEFT CLUB",
    }
    return aliases.get(normalized, normalized if normalized != UNKNOWN_VALUE else UNKNOWN_VALUE)


def parse_age(raw_value: str) -> int:
    value = (raw_value or "").strip()
    if not value:
        return -1

    try:
        age = int(float(value))
    except ValueError:
        return -1

    if age < 0 or age > 120:
        return -1
    return age


def make_age_bucket(age: int) -> str:
    if age < 0:
        return "unknown"
    if age < 20:
        return "under_20"
    if age < 30:
        return "20s"
    if age < 40:
        return "30s"
    if age < 50:
        return "40s"
    if age < 60:
        return "50s"
    return "60_plus"


def main() -> None:
    configure_logging()
    validate_required_columns(INPUT_FILE, INPUT_COLUMNS)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    missing_age_rows = 0
    unknown_fashion_rows = 0
    unknown_member_rows = 0

    with INPUT_FILE.open(newline="", encoding="utf-8") as infile, OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in reader:
            customer_id = (row.get("customer_id") or "").strip()
            if not customer_id:
                continue

            age = parse_age(row.get("age", ""))
            if age < 0:
                missing_age_rows += 1

            fashion_news_frequency = normalize_fashion_news_frequency(row.get("fashion_news_frequency", ""))
            if fashion_news_frequency == UNKNOWN_VALUE:
                unknown_fashion_rows += 1

            club_member_status = normalize_club_member_status(row.get("club_member_status", ""))
            if club_member_status == UNKNOWN_VALUE:
                unknown_member_rows += 1

            writer.writerow(
                {
                    "customer_id": customer_id,
                    "age": age,
                    "age_bucket": make_age_bucket(age),
                    "fashion_news_frequency": fashion_news_frequency,
                    "club_member_status": club_member_status,
                }
            )
            rows_written += 1

    logging.info(
        "customer_features_complete output=%s rows_written=%s missing_age_rows=%s unknown_fashion_rows=%s unknown_member_rows=%s",
        OUTPUT_FILE,
        rows_written,
        missing_age_rows,
        unknown_fashion_rows,
        unknown_member_rows,
    )


if __name__ == "__main__":
    main()
