# -*- coding: utf-8 -*-
"""爬取练习04所需的财务指标数据（5221 只现存股），支持断点续传/重试/多线程。

用途：练习04「财务因子与 PIT 生死线」——构造 ROE / EP / 毛利率 / 营收同比增速，
      对比「按报告期 REPORT_DATE vs 按公告日 NOTICE_DATE」两种对齐的 IC 差异。

接口：akshare(东方财富)  stock_financial_analysis_indicator_em
  - 速度快（单只约 1.6s，无分页），直接返回各财务指标
  - 符号格式为「后缀」：600519.SH / 000001.SZ（区别于三表的 SH600519 前缀）
存储：practice04/data/indicator/{code}.parquet
用法：
  python 01_crawl_indicator.py --limit 20            # 前20只(测试)
  python 01_crawl_indicator.py --only 600519,000001  # 指定代码
  python 01_crawl_indicator.py --workers 3           # 全量(默认)
进度：每 1% 汇报一次（文字进度条）。
"""
import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice04/scripts
ROOT = os.path.dirname(BASE_DIR)                                # practice04
UNIVERSE = os.path.join(os.path.dirname(ROOT), "practice01", "data",
                        "universe", "universe_final.csv")
OUT_DIR = os.path.join(ROOT, "data", "indicator")

START, END = "2010-01-01", "2025-12-31"    # 报告期区间（与年报同区间）

# 保留的字段：双日期 + 练习04所需因子（ROE/毛利率/EPS/营收/净利同比）
COLS = ["REPORT_DATE", "NOTICE_DATE", "REPORT_DATE_NAME",
        "PARENTNETPROFIT", "ROEJQ", "XSMLL", "EPSJB",
        "TOTALOPERATEREVE", "TOTALOPERATEREVETZ", "PARENTNETPROFITTZ"]


def to_symbol(code: str) -> str:
    """6位代码 -> 后缀格式：600519.SH / 000001.SZ（财务指标接口专用）。"""
    return code + (".SH" if code.startswith("6") else ".SZ")


def fetch(code: str) -> pd.DataFrame:
    """拉取单只的财务指标，过滤到 2010-2025 报告期，只留需要的列。"""
    for attempt in range(3):
        try:
            df = ak.stock_financial_analysis_indicator_em(symbol=to_symbol(code))
            if df is None or len(df) == 0:
                return pd.DataFrame()
            rd = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
            df = df[(rd >= START) & (rd <= END)]
            # 容忍缺列（如银行无某些字段）：缺失列补 NaN，保证 schema 一致
            df = df.reindex(columns=COLS)
            df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
            df["NOTICE_DATE"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce")
            df["code"] = code
            df = df.dropna(subset=["NOTICE_DATE"]).sort_values("REPORT_DATE")
            return df.reset_index(drop=True)
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1) + random.random())
            else:
                raise
    return pd.DataFrame()


def process(code: str, sleep_range=(0.1, 0.2)):
    """处理单只：拉取->写 parquet。返回 (code, rows, error或None)。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{code}.parquet")
    try:
        df = fetch(code)
        if len(df) == 0:
            return (code, 0, None)
        df.to_parquet(path, index=False)
        time.sleep(random.uniform(*sleep_range))
        return (code, len(df), None)
    except Exception as e:
        return (code, -1, f"{type(e).__name__}: {e}")


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
    ap.add_argument("--sleep", type=float, default=0.1, help="每任务平均sleep秒")
    args = ap.parse_args()

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    codes = list(uni.loc[uni["status"] == "current", "code"])   # 现存股
    if args.only:
        codes = [c.strip().zfill(6) for c in args.only.split(",") if c.strip()]
    elif args.limit > 0:
        codes = codes[:args.limit]

    todo = []
    for code in codes:
        p = os.path.join(OUT_DIR, f"{code}.parquet")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        todo.append(code)

    print(f"待处理 {len(todo)} 只（总 {len(codes)} 只）", flush=True)
    if not todo:
        print("无待处理，退出。", flush=True)
        return

    failures = []
    t0 = time.time()
    done = empty = 0
    last_milestone = 0
    print(f"[进度] {render_bar(0, len(todo))}  开始...", flush=True)

    if args.workers <= 1:
        for code in todo:
            c, rows, err = process(code, (args.sleep, args.sleep * 1.8))
            if err:
                failures.append((c, err))
            elif rows == 0:
                empty += 1
            else:
                done += 1
            processed = done + empty + len(failures)
            milestone = int(processed / len(todo) * 100)
            if milestone > last_milestone:
                last_milestone = milestone
                print(f"[进度] {render_bar(processed, len(todo))} "
                      f"成功={done} 空={empty} 失败={len(failures)} 耗时{time.time()-t0:.0f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, c, (args.sleep, args.sleep * 1.8)): c for c in todo}
            for fut in as_completed(futs):
                c, rows, err = fut.result()
                if err:
                    failures.append((c, err))
                elif rows == 0:
                    empty += 1
                else:
                    done += 1
                processed = done + empty + len(failures)
                milestone = int(processed / len(todo) * 100)
                if milestone > last_milestone:
                    last_milestone = milestone
                    print(f"[进度] {render_bar(processed, len(todo))} "
                          f"成功={done} 空={empty} 失败={len(failures)} 耗时{time.time()-t0:.0f}s", flush=True)

    print(f"完成。成功={done} 空={empty} 失败={len(failures)} 耗时{time.time()-t0:.0f}s", flush=True)
    if failures:
        print(f"失败清单(前10条): {failures[:10]}", flush=True)


if __name__ == "__main__":
    main()
