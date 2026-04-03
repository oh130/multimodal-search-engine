import csv

input_file = "../data/raw/customers.csv"
output_file = "../data/processed/customers_extract_features.csv"

with open(input_file, newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = [
        "customer_id",
        "age",
        "fashion_news_frequency",
        "club_member_status"
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        writer.writerow({
            "customer_id": row["customer_id"],
            "age": row["age"],
            "fashion_news_frequency": row["fashion_news_frequency"],
            "club_member_status": row["club_member_status"]
        })

print("완료: customer_extract_features.csv 생성됨")