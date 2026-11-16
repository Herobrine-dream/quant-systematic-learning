# -*- coding: utf-8 -*-
"""爬取 A 股「不复权(adjust="")」日线，逐只存 parquet，支持断点续传/重试/多线程。

用途：市值 = 不复权收盘价 × 总股本。后复权价算市值会混入复权因子（估偏），
      所以练习03的「市值中性化 + ln(市值)因子」需要这套不复权数据。

接口：akshare(东方财富)  stock_zh_a_hist(symbol, period="daily", start_date, end_date, adjust="")
存储：practice03/data/raw_daily/{code}.parquet   (逐只，断点续传)
      末尾合并 -> practice03/data/raw_close_all.parquet
用法：
  python 02_crawl_raw_close.py --limit 20            # 前20只(测试)
  python 02_crawl_raw_close.py --only 600519,000001  # 指定代码
  python 02_crawl_raw_close.py --workers 3           # 全量(默认)
"""
import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice03/scripts
ROOT = os.path.dirname(BASE_DIR)                                # practice03
UNIVERSE = os.path.join(os.path.dirname(ROOT), "practice01", "data",
                        "universe", "universe_final.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw_daily")
MERGED = os.path.join(ROOT, "data", "raw_close_all.parquet")

START, END = "20150101", "20241231"     # 与练习1的后复权日线同区间(2015-2024)

# 东财接口返回的中文列 -> 英文列（只保留与后复权日线一致的 8 列）
RENAME = {
    "日期": "date", "股票代码": "code", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
}
KEEP = ["date", "code", "open", "close", "high", "low", "volume", "amount"]


def fetch_raw(code: str) -> pd.DataFrame:
    """拉取单只股票的不复权日线（2015-2024），转英文列。"""
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=START, end_date=END, adjust="")
            if df is None or len(df) == 0:
                return pd.DataFrame()
            df = df.rename(columns=RENAME)[KEEP]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["code"] = code
            df = df[df["close"] > 0]          # 剔除非正价行（防御性）
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1) + random.random())
            else:
                raise
    return pd.DataFrame()


def process(code: str, sleep_range=(0.1, 0.25)):
    """处理单只：拉取->写 parquet。返回 (code, rows, error或None)。"""
    path = os.path.join(RAW_DIR, f"{code}.parquet")
    try:
        df = fetch_raw(code)
        if len(df) == 0:
            return (code, 0, None)
        df.to_parquet(path, index=False)
        time.sleep(random.uniform(*sleep_range))
        return (code, len(df), None)
    except Exception as e:
        return (code, -1, f"{type(e).__name__}: {e}")


def merge():
    """把逐只 parquet 合并成单个 raw_close_all.parquet（供因子脚本快速加载）。"""
    files = sorted(os.listdir(RAW_DIR))
    files = [f for f in files if f.endswith(".parquet")]
    parts = []
    for f in files:
        d = pd.read_parquet(os.path.join(RAW_DIR, f))
        parts.append(d)
    all_df = pd.concat(parts, ignore_index=True)
    all_df = all_df.drop_duplicates(["date", "code"]).sort_values(["date", "code"])
    all_df.to_parquet(MERGED, index=False)
    print(f"合并完成：{len(all_df)} 行 -> {MERGED}", flush=True)


def render_bar(processed, total, width=30):
    """文字进度条，如：[##########--------------------] 20.0% (1044/5221)"""
    pct = processed / total if total else 1.0
    filled = int(width * pct)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct * 100:5.1f}% ({processed}/{total})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前N只")
    ap.add_argument("--only", type=str, default="", help="逗号分隔的指定代码")
    ap.add_argument("--workers", type=int, default=3, help="线程数")
    ap.add_argument("--sleep", type=float, default=0.15, help="每任务平均sleep秒")
    ap.add_argument("--no-merge", action="store_true", help="不执行末尾合并")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    codes = list(uni.loc[uni["status"] == "current", "code"])   # 现存股(与财报口径一致)
    if args.only:
        codes = [c.strip().zfill(6) for c in args.only.split(",") if c.strip()]
    elif args.limit > 0:
        codes = codes[:args.limit]

    # 断点续传：已存在且非空则跳过
    todo = []
    for code in codes:
        p = os.path.join(RAW_DIR, f"{code}.parquet")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        todo.append(code)

    print(f"待处理 {len(todo)} 只（总 {len(codes)} 只）", flush=True)
    if not todo:
        print("无待处理，跳过爬取。", flush=True)
    else:
        failures = []
        t0 = time.time()
        done = empty = 0
        last_milestone = 0
        print(f"[进度] {render_bar(0, len(todo))}  开始...", flush=True)
        if args.workers <= 1:
            for i, code in enumerate(todo, 1):
                c, rows, err = process(code, (args.sleep, args.sleep * 1.8))
                if err:
                    failures.append((c, err))
                elif rows == 0:
                    empty += 1
                else:
                    done += 1
                processed = done + empty + len(failures)
                milestone = int(processed / len(todo) * 10)
                if milestone > last_milestone:
                    last_milestone = milestone
                    el = time.time() - t0
                    print(f"[进度] {render_bar(processed, len(todo))} "
                          f"成功={done} 空={empty} 失败={len(failures)} 耗时{el:.0f}s", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(process, c, (args.sleep, args.sleep * 1.8)): c
                        for c in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    c, rows, err = fut.result()
                    if err:
                        failures.append((c, err))
                    elif rows == 0:
                        empty += 1
                    else:
                        done += 1
                    processed = done + empty + len(failures)
                    milestone = int(processed / len(todo) * 10)
                    if milestone > last_milestone:
                        last_milestone = milestone
                        el = time.time() - t0
                        print(f"[进度] {render_bar(processed, len(todo))} "
                              f"成功={done} 空={empty} 失败={len(failures)} 耗时{el:.0f}s", flush=True)
        print(f"爬取完成。成功={done} 空={empty} 失败={len(failures)} 耗时{time.time()-t0:.0f}s", flush=True)
        if failures:
            print(f"失败清单(前10条): {failures[:10]}", flush=True)

    if not args.no_merge:
        print("开始合并 ...", flush=True)
        merge()


if __name__ == "__main__":
    main()
