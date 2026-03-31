import csv

input_file = "data/raw/customers.csv"
output_file = "data/processed/customer_features.csv"

def make_age_bucket(age):
    if age == -1:
        return "unknown"
    elif age < 20:
        return "10s"
    elif age < 30:
        return "20s"
    elif age < 40:
        return "30s"
    else:
        return "40+"

with open(input_file, newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = [
        "customer_id",
        "age",
        "age_bucket",
        "fashion_news_frequency",
        "club_member_status"
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        # age 처리
        raw_age = row["age"].strip() if row["age"] else ""
        if raw_age == "":
            age = -1
        else:
            try:
                age = int(float(raw_age))
            except ValueError:
                age = -1

        age_bucket = make_age_bucket(age)

        # categorical 결측 처리
        fashion_news_frequency = (
            row["fashion_news_frequency"].strip()
            if row["fashion_news_frequency"] else "UNKNOWN"
        )
        club_member_status = (
            row["club_member_status"].strip()
            if row["club_member_status"] else "UNKNOWN"
        )

        writer.writerow({
            "customer_id": row["customer_id"],
            "age": age,
            "age_bucket": age_bucket,
            "fashion_news_frequency": fashion_news_frequency,
            "club_member_status": club_member_status
        })

print("완료: customer_features.csv 생성됨")