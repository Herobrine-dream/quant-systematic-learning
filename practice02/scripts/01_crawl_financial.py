# -*- coding: utf-8 -*-
"""爬取沪深A股年报财务三表(2010-2025)，逐只存 parquet，支持断点续传/重试/多线程。

接口：东方财富(akshare)
  - 利润表   stock_profit_sheet_by_yearly_em
  - 资产负债表 stock_balance_sheet_by_yearly_em
  - 现金流量表 stock_cash_flow_sheet_by_yearly_em
存储：data/financial/{profit,balance,cashflow}/{code}.parquet
用法：
  python 01_crawl_financial.py --limit 20          # 前20只(测试)
  python 01_crawl_financial.py --only 600519,000001  # 指定代码
  python 01_crawl_financial.py --workers 3         # 全量(默认)
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice02/scripts
ROOT = os.path.dirname(BASE_DIR)                                # practice02
UNIVERSE = os.path.join(os.path.dirname(ROOT), "practice01", "data",
                        "universe", "universe_final.csv")
FIN_DIR = os.path.join(ROOT, "data", "financial")
FAIL_FILE = os.path.join(FIN_DIR, "_failures.json")

START, END = "2010-01-01", "2025-12-31"        # 只保留这些年报

# 三张表 -> 对应接口
STATEMENTS = {
    "profit": ak.stock_profit_sheet_by_yearly_em,
    "balance": ak.stock_balance_sheet_by_yearly_em,
    "cashflow": ak.stock_cash_flow_sheet_by_yearly_em,
}


def to_symbol(code: str) -> str:
    """6位代码 -> 交易所前缀格式：SH600519 / SZ000001。"""
    return ("SH" if code.startswith("6") else "SZ") + code


def fetch_statement(kind: str, code: str) -> pd.DataFrame:
    """拉取单只股票单张年报表，过滤到 2010-2025 年。"""
    fn = STATEMENTS[kind]
    symbol = to_symbol(code)
    for attempt in range(3):
        try:
            df = fn(symbol=symbol)
            if df is None or len(df) == 0:
                return pd.DataFrame()
            # REPORT_DATE 过滤到目标区间
            rd = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
            df = df[(rd >= START) & (rd <= END)]
            df = df.sort_values("REPORT_DATE").reset_index(drop=True)
            return df
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1) + random.random())
            else:
                raise
    return pd.DataFrame()


def process(kind: str, code: str, sleep_range=(0.1, 0.3)):
    """处理单只单表：拉取->写parquet。返回 (kind, code, rows, error或None)。"""
    d = os.path.join(FIN_DIR, kind)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{code}.parquet")
    try:
        df = fetch_statement(kind, code)
        if len(df) == 0:
            return (kind, code, 0, None)
        df.to_parquet(path, index=False)
        time.sleep(random.uniform(*sleep_range))
        return (kind, code, len(df), None)
    except Exception as e:
        return (kind, code, -1, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前N只")
    ap.add_argument("--only", type=str, default="", help="逗号分隔的指定代码")
    ap.add_argument("--workers", type=int, default=3, help="线程数")
    ap.add_argument("--sleep", type=float, default=0.2, help="每任务平均sleep秒")
    args = ap.parse_args()

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    codes = list(uni.loc[uni["status"] == "current", "code"])   # 现存股
    if args.only:
        codes = [c.strip().zfill(6) for c in args.only.split(",") if c.strip()]
    elif args.limit > 0:
        codes = codes[:args.limit]

    # 任务列表 = 代码 × 三表；已存在且非空的跳过(断点续传)
    todo = []
    for code in codes:
        for kind in STATEMENTS:
            p = os.path.join(FIN_DIR, kind, f"{code}.parquet")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
            todo.append((kind, code))

    print(f"待处理任务 {len(todo)} 个（{len(codes)} 只 × 3 表）", flush=True)
    if not todo:
        print("无待处理，退出。")
        return

    failures = {}
    t0 = time.time()
    done = empty = 0
    done_codes = set()

    if args.workers <= 1:
        for i, (kind, code) in enumerate(todo, 1):
            k, c, rows, err = process(kind, code, (args.sleep, args.sleep * 1.8))
            if err:
                failures[f"{k}:{c}"] = err
            elif rows == 0:
                empty += 1
            else:
                done += 1
            done_codes.add(c)
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] ok={done} empty={empty} fail={len(failures)} "
                      f"耗时{el:.0f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, k, c, (args.sleep, args.sleep * 1.8)): (k, c)
                    for k, c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                k, c, rows, err = fut.result()
                if err:
                    failures[f"{k}:{c}"] = err
                elif rows == 0:
                    empty += 1
                else:
                    done += 1
                done_codes.add(c)
                if i % 200 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"[{i}/{len(todo)}] ok={done} empty={empty} fail={len(failures)} "
                          f"耗时{el:.0f}s", flush=True)

    if failures:
        with open(FAIL_FILE, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
    print(f"\n完成。成功={done} 空={empty} 失败={len(failures)} 总耗时{time.time()-t0:.0f}s")
    if failures:
        print(f"失败清单({len(failures)}条): {list(failures.items())[:5]}")


if __name__ == "__main__":
    main()
