import sqlite3
import csv
import pandas as pd
import os

def valid_page(page):
    crawl_id, url, is_pdf, cur_depth, body = page
    return True

pickled_df = "../../processed_df.pkl"

conn = sqlite3.connect('/vol_b/data/scrapy_cluster_data/data.db', timeout=30)
c = conn.cursor()
print("Beginning data fetch")
data = c.execute("""SELECT crawlid, response_url, is_pdf, curdepth, body FROM dump ORDER BY crawlid""").fetchall()
print("Finished fetching data")
c.close()

filtered_data = []
cur_crawlid, cur_row = data[0][0], []

print("Beginning grouping and filtering")
for i in range(len(data)):
    if data[i][0] != cur_crawlid:
        filtered_data.append((cur_crawlid, cur_row))
        cur_row = []
        cur_crawlid = data[i][0]
    if valid_page(data[i]):
        cur_row.append(data[i][1:])
filtered_data.append((cur_crawlid, cur_row))
data = filtered_data
data = [[int(data[i][0].split("_")[-1]), data[i][1]] for i in range(len(data))]
print("Finished grouping and filtering")

print("Creating df")
df = pd.DataFrame(data)
df.columns = ["crawl_id", "pages"]
df = df.sort_values(by="crawl_id")
print("Finished creating df")

df_csv = pd.read_csv('../data/charter_URLs_2016.csv')
df_csv["data"] = ""

print("Merging dataframes")
merged_df = df_csv.copy()
i = 0
for index, row in df_csv.iterrows():
    if df.iloc[i][0] == index + 1:
        merged_df.at[index, "data"] = df.iloc[i][1]
        i += 1
merged_df.to_pickle(pickled_df)
