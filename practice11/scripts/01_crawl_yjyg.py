# -*- coding: utf-8 -*-
"""
练习 11 · 业绩预告爬取（东财 stock_yjyg_em，按报告期，断点续传）
================================================================
接口按【报告期】组织：stock_yjyg_em(date="20231231") 返回该报告期全部预告，
内含【公告日期】列（事件日）。故遍历 2015~2024 的 40 个季度报告期。

输出：data/yjyg/{报告期}.csv（每期一个文件，断点续传）
"""
import os
import time

import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # practice11/
DATA = os.path.join(BASE, "data")
OUT_DIR = os.path.join(DATA, "yjyg")
START_YEAR, END_YEAR = 2015, 2024
SLEEP = 0.5


def quarter_dates(y0, y1):
    ds = []
    for y in range(y0, y1 + 1):
        for m, d in [("03", "31"), ("06", "30"), ("09", "30"), ("12", "31")]:
            ds.append(f"{y}{m}{d}")
    return ds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    periods = quarter_dates(START_YEAR, END_YEAR)
    ok, skip, fail = 0, 0, 0
    for i, p in enumerate(periods):
        out = os.path.join(OUT_DIR, f"{p}.csv")
        if os.path.exists(out) and os.path.getsize(out) > 20:   # 断点续传
            skip += 1
            continue
        try:
            df = ak.stock_yjyg_em(date=p)
            if df is not None and len(df) > 0:
                df.to_csv(out, index=False, encoding="utf-8-sig")
                ok += 1
                print(f"[{i+1}/{len(periods)}] {p} OK {len(df)}", flush=True)
            else:
                print(f"[{i+1}/{len(periods)}] {p} 空", flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[{i+1}/{len(periods)}] {p} ERR {type(e).__name__}: {e}", flush=True)
        time.sleep(SLEEP)
    print(f"\n完成：抓取 {ok}，跳过 {skip}，失败 {fail}", flush=True)


if __name__ == "__main__":
    main()
