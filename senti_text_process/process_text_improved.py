# =====================================
# 🚀 股票文本特征编码主程序（对齐 reference_keys_2024）
# ✅ 节假日归入上一个交易日 closing
# ✅ 输出 CSV + Parquet
# ✅ 固定输出所有参考交易日
# =====================================
import os
import glob
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch.nn as nn

pool = None

# ---------- 工具函数 ----------
def get_prev_trade_date(date, stock_ref_dates):
    """在该股票交易日列表中找到上一个交易日"""
    date = pd.Timestamp(date)
    prev = [d for d in stock_ref_dates if pd.Timestamp(d) < date]
    return prev[-1] if prev else None


def map_time_period_vectorized(df, stock_ref_set, stock_ref_dates):
    """向量化时间段映射（节假日归入上一个交易日 closing）"""
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
        elif t < 9 * 60 + 30:  # 早盘前发帖 → 前一交易日 closing
            mapped_dates.append(get_prev_trade_date(d, stock_ref_dates))
        else:
            mapped_dates.append(d)
    df["mapped_date"] = mapped_dates
    return df


'''
def encode_texts(texts, model, batch_size=1024):
    """将文本列表编码为平均 embedding"""
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return np.zeros(model.get_sentence_embedding_dimension())
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return embeddings.mean(axis=0)
'''
def encode_texts(texts, model, batch_size=1024):
    """
    将文本列表编码为平均 embedding。
    如果 'pool' 存在，则使用多进程编码。
    """
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return np.zeros(model.get_sentence_embedding_dimension())

    # 决定是否进行归一化
    # (因为 encode_multi_process 不支持，我们需要手动做)
    do_normalization = True 

    global pool
    if pool:
        # --- 多 GPU 路径 ---
        embeddings = model.encode_multi_process(
            texts,
            pool=pool,
            batch_size=batch_size, # 这是 *每个 GPU* 的 batch_size
        )
    else:
        # --- 单 GPU / CPU 路径 ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device) # 确保模型在正确设备上
        embeddings = model.encode(
            texts,
            batch_size=batch_size, # 这是总的 batch_size
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=do_normalization, # 单 GPU 支持自动归一化
            device=device
        )
    
    # 如果使用了多进程，我们需要手动归一化
    if pool and do_normalization:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 避免除以 0
        norms[norms == 0] = 1e-12 
        embeddings = embeddings / norms
    elif pool and not do_normalization:
        # 此时 embeddings 已经 ok
        pass
    elif not pool:
        # 单 GPU 路径已经处理了归一化 (normalize_embeddings=do_normalization)
        pass

    return embeddings.mean(axis=0)


# ---------- 主处理函数 ----------
def process_text_encoder(file_path, ref_df, model, output_dir="processed_text_ref2024", save_parquet=True):
    stock_code = os.path.basename(file_path).split(".")[0].zfill(6)
    # print(f"\n🚀 正在处理股票 {stock_code} ...")

    # 参考交易日表
    stock_ref_dates = sorted(ref_df[ref_df["stockbar_code"] == stock_code]["date"].tolist())
    if not stock_ref_dates:
        print(f"⚠️ 股票 {stock_code} 无参考数据，跳过。")
        return None
    stock_ref_set = set(stock_ref_dates)

    # 读取 parquet
    df = pd.read_parquet(file_path)
    df["post_publish_time"] = pd.to_datetime(df["post_publish_time"])

    # 应用时间映射
    df = map_time_period_vectorized(df, stock_ref_set, stock_ref_dates)

    # 检测文本列
    TEXT_COL = next((c for c in ["text", "content", "post_content", "comment", "body"] if c in df.columns), None)
    if TEXT_COL is None:
        raise KeyError(f"❌ 找不到文本列：{df.columns.tolist()}")

    # 分组聚合
    grouped = df.groupby(["stockbar_code", "mapped_date", "time_period"])[TEXT_COL].apply(list).reset_index()

    # 编码
    embeddings = []
    for _, row in grouped.iterrows():
        emb = encode_texts(row[TEXT_COL], model)
        embeddings.append(emb)
    emb_array = np.vstack(embeddings)
    emb_cols = [f"dim_{i}" for i in range(emb_array.shape[1])]

    emb_df = pd.concat([
        grouped[["stockbar_code", "mapped_date", "time_period"]],
        pd.DataFrame(emb_array, columns=emb_cols)
    ], axis=1)

    # 转宽表 (trading / closing)
    pivoted = (
        emb_df.pivot(index=["stockbar_code", "mapped_date"], columns="time_period", values=emb_cols)
        .reset_index()
    )
    pivoted.columns = ["_".join([c for c in col if c]) for col in pivoted.columns.values]

    # === 按参考日期补齐 ===
    full_df = pd.DataFrame({
        "stockbar_code": stock_code,
        "mapped_date": stock_ref_dates
    })
    pivoted = pd.merge(full_df, pivoted, on=["stockbar_code", "mapped_date"], how="left")

    # 缺失填充 0
    for col in pivoted.columns:
        if col not in ["stockbar_code", "mapped_date"]:
            pivoted[col] = pivoted[col].fillna(0.0)

    # 保存
    csv_dir = os.path.join(output_dir, "csv")
    parquet_dir = os.path.join(output_dir, "parquet")
    os.makedirs(csv_dir, exist_ok=True)
    if save_parquet:
        os.makedirs(parquet_dir, exist_ok=True)

    out_csv = os.path.join(csv_dir, f"{stock_code}_text_features.csv")
    pivoted.round(6).to_csv(out_csv, index=False, encoding="utf-8-sig")

    if save_parquet:
        out_parquet = os.path.join(parquet_dir, f"{stock_code}_text_features.parquet")
        pivoted.to_parquet(out_parquet, index=False)

    # print(f"✅ 完成 {stock_code} ({len(pivoted)} 行)")
    return stock_code


# ---------- 批量执行 ----------
def process_multiple_files(input_dir, ref_path, output_dir, save_parquet=True):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    ref_df = pd.read_csv(ref_path)
    ref_df["stockbar_code"] = ref_df["stockbar_code"].astype(str).str.zfill(6)
    ref_df["date"] = pd.to_datetime(ref_df["date"]).dt.date

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "processed_log.json")
    processed = set(json.load(open(log_file)) if os.path.exists(log_file) else [])

    print(f"🔍 检测到 {len(files)} 个文件，已处理 {len(processed)} 个。")

    # 加载模型一次
    model_name = "BAAI/bge-large-zh-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 加载模型: {model_name} ({device})")
    model = SentenceTransformer(model_name, device=device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"🚀 启动多 GPU 编码，共 {torch.cuda.device_count()} 个 GPU。")

        # 2. 定义目标设备
        target_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    
        # 3. 启动多进程池
        # pool 是一个字典，key 是设备名，value 是进程
        global pool
        pool = model.start_multi_process_pool(target_devices=target_devices)

    for f in tqdm(files):
        stock_code = os.path.splitext(os.path.basename(f))[0].zfill(6)
        if stock_code in processed:
            continue
        try:
            result = process_text_encoder(f, ref_df, model, output_dir, save_parquet=save_parquet)
            if result:
                processed.add(result)
                json.dump(list(processed), open(log_file, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ {stock_code}: {e}")

    print(f"\n🎉 完成！共处理 {len(processed)} 只股票。")


# ---------- 主入口 ----------
if __name__ == "__main__":
    root_path = '..'
    input_dir = f"{root_path}/csi300_senti_with_comments"                       # ← 输入目录
    ref_path = f"{root_path}/pack/reference_keys_2024.csv"  # ← 参考交易日
    output_dir = f"{root_path}/processed_text_ref2024"                # ← 输出目录
    process_multiple_files(input_dir, ref_path, output_dir, save_parquet=True)
