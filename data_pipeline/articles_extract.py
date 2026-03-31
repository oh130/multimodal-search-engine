import csv
from collections import Counter

input_file = "data/raw/articles.csv"
output_file = "data/processed/articles_feature.csv"

# 1차: product_type_name 빈도 세기
category_counter = Counter()

with open(input_file, newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        category = row["product_type_name"].strip() if row["product_type_name"] else "UNKNOWN"
        category_counter[category] += 1

top_k = 20
top_categories = set([cat for cat, _ in category_counter.most_common(top_k)])

# 2차: feature csv 생성
with open(input_file, newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = [
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
        "color"
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        product_type = row["product_type_name"].strip() if row["product_type_name"] else "UNKNOWN"
        product_group = row["product_group_name"].strip() if row["product_group_name"] else "UNKNOWN"
        color = row["colour_group_name"].strip() if row["colour_group_name"] else "UNKNOWN"

        category = product_type if product_type in top_categories else "other"

        writer.writerow({
            "article_id": row["article_id"],
            "prod_name": row["prod_name"],
            "product_type_name": product_type,
            "product_group_name": product_group,
            "colour_group_name": color,
            "perceived_colour_master_name": row["perceived_colour_master_name"].strip() if row["perceived_colour_master_name"] else "UNKNOWN",
            "department_name": row["department_name"].strip() if row["department_name"] else "UNKNOWN",
            "section_name": row["section_name"].strip() if row["section_name"] else "UNKNOWN",
            "garment_group_name": row["garment_group_name"].strip() if row["garment_group_name"] else "UNKNOWN",
            "category": category,
            "main_category": product_group,
            "color": color
        })

print("완료: articles_feature.csv 생성됨")