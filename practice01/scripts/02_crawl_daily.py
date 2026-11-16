# -*- coding: utf-8 -*-
"""爬取沪深A股日线(后复权, 2015-01-01 ~ 2024-12-31)，逐只存 parquet，支持断点续传/重试。
复权口径：后复权(hfq)。下载时剔除"非正价(≤0)尾巴"行(退市整理期东方财富hfq因子bug, 含归零)，
被剔除的(代码->行数)写入 data/daily/_neg_dropped.json 备查。
用法:
  python 02_crawl_daily.py --limit 40                # 全量中前40只(续传模式)
  python 02_crawl_daily.py --only 000001,300104 ...  # 指定代码(逗号分隔)
  python 02_crawl_daily.py --workers 3               # 全量, 3线程(默认)
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

UNIVERSE = "data/universe/universe.csv"
DAILY_DIR = "data/daily"
FAIL_FILE = os.path.join(DAILY_DIR, "_failures.json")
NEG_FILE = os.path.join(DAILY_DIR, "_neg_dropped.json")

RENAME = {
    "日期": "date", "股票代码": "code", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
}
START, END = "20150101", "20241231"


def fetch_one(code: str) -> pd.DataFrame:
    """拉取单只股票后复权日线，返回标准化DataFrame；空则返回空df。"""
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=START, end_date=END, adjust="hfq")
            if df is None or len(df) == 0:
                return pd.DataFrame()
            df = df.rename(columns=RENAME)
            keep = ["date", "code", "open", "close", "high", "low", "volume", "amount"]
            df = df[[c for c in keep if c in df.columns]]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1) + random.random())
            else:
                raise
    return pd.DataFrame()


def process(code: str, sleep_range=(0.2, 0.5)):
    """处理单只：拉取->剔负价行->写parquet。返回 (code, rows, n_neg, error或None)。"""
    path = os.path.join(DAILY_DIR, f"{code}.parquet")
    try:
        df = fetch_one(code)
        if len(df) == 0:
            return (code, 0, 0, None)  # 空：窗口内无数据(如2025后上市)
        # 剔除非正价尾巴(任何OHLC<=0的行, 含复权因子归零导致的0价)
        neg_mask = (df[["open", "close", "high", "low"]] <= 0).any(axis=1)
        n_neg = int(neg_mask.sum())
        df = df[~neg_mask].reset_index(drop=True)
        if len(df) == 0:
            return (code, 0, n_neg, None)
        df.to_parquet(path, index=False)
        time.sleep(random.uniform(*sleep_range))
        return (code, len(df), n_neg, None)
    except Exception as e:
        return (code, -1, 0, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前N只(续传时仍跳过已完成的)")
    ap.add_argument("--only", type=str, default="", help="逗号分隔的指定代码")
    ap.add_argument("--workers", type=int, default=1, help="线程数")
    ap.add_argument("--sleep", type=float, default=0.3, help="每只间平均sleep秒")
    args = ap.parse_args()

    os.makedirs(DAILY_DIR, exist_ok=True)
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    codes = list(uni["code"])

    if args.only:
        codes = [c.strip().zfill(6) for c in args.only.split(",") if c.strip()]
    elif args.limit > 0:
        codes = codes[:args.limit]

    # 已有 parquet 且非空的视为完成；失败/空股无文件，会自动被重新尝试
    todo = []
    for c in codes:
        p = os.path.join(DAILY_DIR, f"{c}.parquet")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        todo.append(c)

    print(f"待处理 {len(todo)} 只 / 总 {len(codes)} 只", flush=True)
    if not todo:
        print("无待处理，退出。")
        return

    failures = {}
    neg_dropped = {}
    t0 = time.time()
    done_cnt = 0
    empty_cnt = 0
    total_neg = 0

    def finish(code, rows, n_neg, err):
        nonlocal done_cnt, empty_cnt, total_neg
        if err:
            failures[code] = err
        elif rows == 0:
            empty_cnt += 1
        else:
            done_cnt += 1
        if n_neg > 0:
            neg_dropped[code] = n_neg
            total_neg += n_neg

    if args.workers <= 1:
        for i, c in enumerate(todo, 1):
            res = process(c, sleep_range=(args.sleep, args.sleep * 1.8))
            finish(*res)
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] ok={done_cnt} empty={empty_cnt} fail={len(failures)} "
                      f"剔负价行={total_neg} 耗时{el:.0f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, c, (args.sleep, args.sleep * 1.8)): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                res = fut.result()
                finish(*res)
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"[{i}/{len(todo)}] ok={done_cnt} empty={empty_cnt} fail={len(failures)} "
                          f"剔负价行={total_neg} 耗时{el:.0f}s", flush=True)

    if failures:
        with open(FAIL_FILE, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
    if neg_dropped:
        with open(NEG_FILE, "w", encoding="utf-8") as f:
            json.dump(neg_dropped, f, ensure_ascii=False, indent=2)
    print(f"\n完成。成功={done_cnt} 空={empty_cnt} 失败={len(failures)} "
          f"剔负价行总{total_neg}行(分布见{os.path.basename(NEG_FILE)}) 总耗时{time.time()-t0:.0f}s")
    if failures:
        print("失败清单:", failures)


if __name__ == "__main__":
    main()
