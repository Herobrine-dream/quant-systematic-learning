# -*- coding: utf-8 -*-
"""
练习 10 · 可转债数据爬取（断点续传 + 限速 + 重试）
================================================================
1. bond_zh_cov           -> data/cb_universe.csv   全量转债元数据（代码/正股/转股价/发行规模/上市时间/评级）
2. bond_zh_hs_cov_daily -> data/cb_daily/{code}.csv  逐只日线（新浪，历史 2018+）
3. bond_cb_adj_logs_jsl -> data/cb_adj/{code}.csv    逐只转股价【下修】记录（集思录）

用法：PyCharm 选 Anaconda 解释器，直接运行；中断后重跑同一命令即断点续传。
"""
import os
import time

import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # practice10/
DATA = os.path.join(BASE, "data")
DAILY_DIR = os.path.join(DATA, "cb_daily")
ADJ_DIR = os.path.join(DATA, "cb_adj")

UNIVERSE = os.path.join(DATA, "cb_universe.csv")
END_DATE = "2024-12-31"          # 正股价数据只到 2024-12-31，回测窗口与之对齐
SLEEP = 1.2                      # 秒/请求，防封 IP
TRIES = 3


def symbol_of(code):
    """可转债代码 -> 新浪带市场前缀代码。11xxxx=沪(sh)，12xxxx=深(sz)。"""
    code = str(code).zfill(6)
    return ("sh" if code.startswith("11") else "sz") + code


def with_retry(fn, *args, tries=TRIES, **kw):
    last = None
    for a in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (a + 1))
    raise last


def crawl_universe():
    os.makedirs(DATA, exist_ok=True)
    print("[1] 抓取全量转债元数据 bond_zh_cov ...", flush=True)
    df = with_retry(ak.bond_zh_cov)
    df.to_csv(UNIVERSE, index=False, encoding="utf-8-sig")
    print(f"    原始 {len(df)} 行 -> {UNIVERSE}", flush=True)
    return df


def _existing_nontrivial(d):
    """已存在且非空的代码集合（用于断点续传跳过）。"""
    out = set()
    if os.path.isdir(d):
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if fn.endswith(".csv") and os.path.getsize(p) > 20:
                out.add(os.path.splitext(fn)[0])
    return out


def crawl_daily(codes):
    os.makedirs(DAILY_DIR, exist_ok=True)
    done = _existing_nontrivial(DAILY_DIR)
    todo = [c for c in codes if c not in done]
    print(f"[2] 抓取转债日线（新浪）：待抓 {len(todo)} 只，已存在 {len(done)} 只", flush=True)
    for i, code in enumerate(todo):
        try:
            df = with_retry(ak.bond_zh_hs_cov_daily, symbol=symbol_of(code))
            if df is not None and len(df) > 0:
                df.to_csv(os.path.join(DAILY_DIR, f"{code}.csv"), index=False, encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            print(f"    daily {code} ERR {type(e).__name__}: {e}", flush=True)
        time.sleep(SLEEP)
        if (i + 1) % 20 == 0:
            print(f"    daily {i + 1}/{len(todo)}", flush=True)
    print("    daily 完成", flush=True)


def crawl_adj(codes):
    os.makedirs(ADJ_DIR, exist_ok=True)
    done = _existing_nontrivial(ADJ_DIR)
    todo = [c for c in codes if c not in done]
    print(f"[3] 抓取转股价下修记录（集思录）：待抓 {len(todo)} 只，已存在 {len(done)} 只", flush=True)
    for i, code in enumerate(todo):
        try:
            df = with_retry(ak.bond_cb_adj_logs_jsl, symbol=str(code).zfill(6))
            path = os.path.join(ADJ_DIR, f"{code}.csv")
            # 无下修记录返回空表：写非空占位（含表头），避免断点续传重复抓取
            if df is None or len(df) == 0:
                with open(path, "w", encoding="utf-8-sig") as f:
                    f.write("转债名称,股东大会日,下修前转股价,下修后转股价,新转股价生效日期,下修底价\n")
            else:
                df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            print(f"    adj {code} ERR {type(e).__name__}: {e}", flush=True)
        time.sleep(SLEEP)
        if (i + 1) % 20 == 0:
            print(f"    adj {i + 1}/{len(todo)}", flush=True)
    print("    adj 完成", flush=True)


def main():
    uni = crawl_universe()
    uni = uni.copy()
    uni["债券代码"] = uni["债券代码"].astype(str).str.zfill(6)
    uni["上市时间"] = pd.to_datetime(uni["上市时间"], errors="coerce")
    uni = uni[uni["上市时间"].notna() & (uni["上市时间"] <= END_DATE)]
    codes = sorted(uni["债券代码"].unique().tolist())
    print(f"    回测窗口内（上市 <= {END_DATE}）的转债：{len(codes)} 只", flush=True)

    # 保存过滤后的股票池（供回测用）
    uni.to_csv(os.path.join(DATA, "cb_universe_filtered.csv"), index=False, encoding="utf-8-sig")

    crawl_daily(codes)
    crawl_adj(codes)
    print("\n[完成] 数据已落地：data/cb_universe*.csv + data/cb_daily/ + data/cb_adj/", flush=True)


if __name__ == "__main__":
    main()
