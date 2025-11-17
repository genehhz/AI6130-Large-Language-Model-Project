# =====================================
# 🚀 股票情绪特征提取主程序（双输出版）
# 基于 reference_keys_2024 对齐交易日
# ✅ 可选归一化 (closing 时段除以 √(跨日天数))
# ✅ 支持 CSV + Parquet 双输出
# =====================================
import os
import glob
import json
import pandas as pd
import numpy as np


# ---------- 工具函数 ----------
def get_prev_trade_date(date, stock_ref_dates):
    """在该股票交易日列表中找到上一个交易日"""
    date = pd.Timestamp(date)
    prev = [d for d in stock_ref_dates if pd.Timestamp(d) < date]
    return prev[-1] if prev else None


def map_time_period_vectorized(df, stock_ref_set, stock_ref_dates):
    """向量化时间段映射"""
    df["hour"] = df["post_publish_time"].dt.hour
    df["minute"] = df["post_publish_time"].dt.minute
    df["date_only"] = df["post_publish_time"].dt.date
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    df["time_period"] = np.where(
        ((df["minute_of_day"] >= 9 * 60 + 30) & (df["minute_of_day"] < 15 * 60)),
        "trading", "closing"
    )

    mapped_dates = []
    for d, t in zip(df["date_only"], df["minute_of_day"]):
        if d not in stock_ref_set:
            mapped_dates.append(get_prev_trade_date(d, stock_ref_dates))
        elif t < 9 * 60 + 30:  # 早盘前
            mapped_dates.append(get_prev_trade_date(d, stock_ref_dates))
        else:
            mapped_dates.append(d)
    df["mapped_date"] = mapped_dates
    return df


def get_closing_day_span(mapped_dates, stock_ref_dates):
    """计算收盘阶段跨日天数（如周五→周一=3）"""
    stock_ref_dates = sorted(pd.to_datetime(stock_ref_dates))
    mapped_dates = pd.to_datetime(mapped_dates)
    date_to_span = {}
    for i, d in enumerate(stock_ref_dates[:-1]):
        span = (stock_ref_dates[i + 1] - d).days
        date_to_span[d.date()] = span
    date_to_span[stock_ref_dates[-1].date()] = 1
    return mapped_dates.map(lambda d: date_to_span.get(d.date(), 1))


# ---------- 主处理函数 ----------
def process_single_file(file_path, ref_df, output_dir,
                        normalize_closing=False, save_parquet=True):
    stock_code = os.path.basename(file_path).split(".")[0].zfill(6)

    # 股票交易日
    stock_ref_dates = sorted(ref_df[ref_df["stockbar_code"] == stock_code]["date"].tolist())
    if not stock_ref_dates:
        return None
    stock_ref_set = set(stock_ref_dates)

    df = pd.read_parquet(file_path)
    df["post_publish_time"] = pd.to_datetime(df["post_publish_time"])
    df = map_time_period_vectorized(df, stock_ref_set, stock_ref_dates)

    # 聚合
    df["emotion"] = df[["neg", "neu", "pos"]].idxmax(axis=1)
    daily = (
        df.groupby(["stockbar_code", "mapped_date", "time_period"])
        .agg(
            Total_Positive=("emotion", lambda x: (x == "pos").sum()),
            Total_Negative=("emotion", lambda x: (x == "neg").sum()),
            Total_Neutral=("emotion", lambda x: (x == "neu").sum()),
            Total_Posts=("emotion", "count"),
            Total_Click_Count=("click_count", "sum"),
        )
        .reset_index()
    )

    # ✅ 可选归一化（√天数版本）
    if normalize_closing:
        daily["Total_Posts"] = daily["Total_Posts"].astype(float)
        daily["Total_Click_Count"] = daily["Total_Click_Count"].astype(float)
        closing_mask = daily["time_period"] == "closing"
        spans = get_closing_day_span(daily.loc[closing_mask, "mapped_date"], stock_ref_dates)
        spans_sqrt = np.sqrt(spans.values)
        daily.loc[closing_mask, "Total_Posts"] /= spans_sqrt
        daily.loc[closing_mask, "Total_Click_Count"] /= spans_sqrt

    # 情绪指标计算
    daily["Emotion_Index"] = (
        (daily["Total_Positive"] - daily["Total_Negative"]) /
        (daily["Total_Positive"] + daily["Total_Negative"]).replace(0, np.nan)
    )

    # 滚动动量
    daily = daily.sort_values(["stockbar_code", "time_period", "mapped_date"])
    for w in [3, 5]:
        daily[f"Emotion_Momentum_{w}d"] = (
            daily.groupby(["stockbar_code", "time_period"])["Emotion_Index"]
            .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
        )

    # 转宽表
    pivoted = (
        daily.pivot(index=["stockbar_code", "mapped_date"], columns="time_period")
        .reset_index()
    )
    pivoted.columns = ["_".join([c for c in col if c]) for col in pivoted.columns.values]

    # 按参考交易日补齐
    full_df = pd.DataFrame({
        "stockbar_code": stock_code,
        "mapped_date": stock_ref_dates
    })
    pivoted = pd.merge(full_df, pivoted, on=["stockbar_code", "mapped_date"], how="left")

    # 填充缺失
    for base in ["Total_Posts", "Total_Click_Count"]:
        for period in ["trading", "closing"]:
            col = f"{base}_{period}"
            pivoted[col] = pivoted.get(col, 0).fillna(0)
    for base in ["Emotion_Index", "Emotion_Momentum_3d", "Emotion_Momentum_5d"]:
        for period in ["trading", "closing"]:
            col = f"{base}_{period}"
            pivoted[col] = pivoted.get(col, 0).ffill().fillna(0)

    # 保存
    csv_dir = os.path.join(output_dir, "csv")
    parquet_dir = os.path.join(output_dir, "parquet")
    os.makedirs(csv_dir, exist_ok=True)
    if save_parquet:
        os.makedirs(parquet_dir, exist_ok=True)

    out_csv = os.path.join(csv_dir, f"{stock_code}_features.csv")
    pivoted.round(4).to_csv(out_csv, index=False, encoding="utf-8-sig")

    if save_parquet:
        out_parquet = os.path.join(parquet_dir, f"{stock_code}_features.parquet")
        pivoted.to_parquet(out_parquet, index=False)

    return stock_code


# ---------- 批量执行 ----------
def process_multiple_files(input_dir, ref_path, output_dir,
                           normalize_closing=False, save_parquet=True):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    ref_df = pd.read_csv(ref_path)
    ref_df["stockbar_code"] = ref_df["stockbar_code"].astype(str).str.zfill(6)
    ref_df["date"] = pd.to_datetime(ref_df["date"]).dt.date
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "processed_log.json")
    processed = set(json.load(open(log_file)) if os.path.exists(log_file) else [])

    print(f"🔍 检测到 {len(files)} 个文件，已处理 {len(processed)} 个。")

    for f in files:
        stock_code = os.path.splitext(os.path.basename(f))[0].zfill(6)
        if stock_code in processed:
            continue
        try:
            result = process_single_file(
                f, ref_df, output_dir,
                normalize_closing=normalize_closing,
                save_parquet=save_parquet
            )
            if result:
                processed.add(result)
                json.dump(list(processed), open(log_file, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
                print(f"✅ {result}")
            else:
                print(f"⚠️ 无参考数据，跳过 {stock_code}")
        except Exception as e:
            print(f"❌ {stock_code}: {e}")

    print(f"\n🎉 完成！共处理 {len(processed)} 只股票。")


# ---------- 主入口 ----------
if __name__ == "__main__":
    input_dir = "csi300_senti_with_comments"
    ref_path = "reference_keys_2024.csv"
    output_dir = "processed_emo_2"
    # ⚙️ normalize_closing=True 启用跨日√归一化
    # ⚙️ save_parquet=True 同时输出 Parquet 文件
    process_multiple_files(input_dir, ref_path, output_dir,
                           normalize_closing=True, save_parquet=True)
