# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 23:09:09 2025

@author: geneh
"""

import pandas as pd
import glob
import os

folder_path = r"C:/Users/geneh/Desktop/stockprices"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file, dtype={'日期': str}, encoding="utf-8")
    df_list.append(temp_df)

final_df = pd.concat(df_list, ignore_index=True)

# ---- Handle mixed date formats ----
# If date contains '-' (YYYY-MM-DD) → treat as ISO
# If date contains '/' (DD/MM/YYYY) → treat as day first
final_df['_日期_dt'] = pd.to_datetime(
    final_df['日期'].str.replace('-', '/'),
    dayfirst=True,
    errors='coerce'
)

# ---- Sort by 股票代码 then date ----
final_df = final_df.sort_values(by=['股票代码', '_日期_dt'])

# ---- Drop helper column, keep original 日期 as-is ----
final_df = final_df.drop(columns=['_日期_dt'])

# ---- Save result ----
final_df.to_csv("all_stock_data_sorted.csv", index=False, encoding="utf-8-sig")

print("✅ Done — sorted by 股票代码 then 日期")
print("✅ Original 日期 text preserved")
