import pandas as pd

# 1) transactions 데이터 일부만 읽기
df = pd.read_csv("transactions_train.csv", nrows=100000)

# 2) 날짜 형식 변환
df["t_dat"] = pd.to_datetime(df["t_dat"])

# 3) 고객 feature 만들기
user_purchase_count = df.groupby("customer_id")["article_id"].count().reset_index()
user_purchase_count.columns = ["customer_id", "purchase_count"]

user_avg_price = df.groupby("customer_id")["price"].mean().reset_index()
user_avg_price.columns = ["customer_id", "avg_price"]

user_feat = user_purchase_count.merge(user_avg_price, on="customer_id")

# 4) 상품 feature 만들기
item_popularity = df.groupby("article_id")["customer_id"].count().reset_index()
item_popularity.columns = ["article_id", "popularity"]

item_price = df.groupby("article_id")["price"].mean().reset_index()
item_price.columns = ["article_id", "item_price"]

item_feat = item_popularity.merge(item_price, on="article_id")

# 5) csv로 저장
user_feat.to_csv("user_features.csv", index=False)
item_feat.to_csv("item_features.csv", index=False)

print("완료")