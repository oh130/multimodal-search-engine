import csv

input_file = "../data/raw/articles.csv"
output_file = "../data/processed/articles_extract_feature.csv"

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
        "garment_group_name"
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        writer.writerow({
            "article_id": row["article_id"],
            "prod_name": row["prod_name"],
            "product_type_name": row["product_type_name"],
            "product_group_name": row["product_group_name"],
            "colour_group_name": row["colour_group_name"],
            "perceived_colour_master_name": row["perceived_colour_master_name"],
            "department_name": row["department_name"],
            "section_name": row["section_name"],
            "garment_group_name": row["garment_group_name"]
        })

print("완료: articles_extract_features.csv 생성됨")