# -*- coding: utf-8 -*-
"""
练习 08 · 数据爬取：中证500 基准日线 + 历史成分（当期 PIT）
================================================================

要解决的问题（本题第一道陷阱 = 前视偏差）：
  中证500 成分每年 6/12 月定期调整，不能用"现在的成分"回测历史。
  必须用每个调仓日【当时的真实成分名单】。

数据源（已实测可用）：
  - 基准日线：akshare `stock_zh_index_hist_csindex(symbol="000905")`（中证官网）
  - 历史成分：baostock `query_zz500_stocks(date=月末交易日)`，按日期返回当期成分快照
    （返回的 updateDate 是"该日期生效的最近一次调样"，天然 PIT 正确）。

产出（practice08/data/）：
  - benchmark_close.csv   date, close（中证500 收盘点位）
  - cons_hist.csv         date, code（每个月末调仓日的 500 只成分，6 位代码）

用法（PyCharm 选 Anaconda 解释器）：python scripts/01_crawl_index.py
"""
import os

import pandas as pd
import akshare as ak
import baostock as bs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # practice08/
DATA = os.path.join(BASE_DIR, "data")


def crawl_benchmark():
    """中证500 收盘点位（中证官网，东财接口偶发断连，故用中证官网）。"""
    df = ak.stock_zh_index_hist_csindex(symbol="000905",
                                        start_date="20150101", end_date="20241231")
    out = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "close"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(os.path.join(DATA, "benchmark_close.csv"), index=False, encoding="utf-8-sig")
    return out


def month_end_dates(bench):
    """每个月的最后一个交易日（调仓日）。"""
    g = bench.groupby(bench["date"].dt.to_period("M"))["date"].max()
    return pd.DatetimeIndex(g.values)


def crawl_constituents(dates):
    """逐调仓日取当期成分快照。baostock 代码形如 sh.600004 -> 转成 600004。"""
    lg = bs.login()
    assert lg.error_code == "0", f"baostock 登录失败: {lg.error_msg}"
    rows = []
    for i, d in enumerate(dates):
        ds = d.strftime("%Y-%m-%d")
        rs = bs.query_zz500_stocks(date=ds)
        if rs.error_code != "0":
            print(f"  warn {ds}: {rs.error_code} {rs.error_msg}", flush=True)
            continue
        for c in rs.get_data()["code"]:
            rows.append({"date": ds, "code": c.split(".")[-1]})   # "sh.600004" -> "600004"
        if (i + 1) % 12 == 0 or i == len(dates) - 1:
            print(f"  {i + 1}/{len(dates)}  {ds}  ok", flush=True)
    bs.logout()
    cons = pd.DataFrame(rows)
    cons.to_csv(os.path.join(DATA, "cons_hist.csv"), index=False, encoding="utf-8-sig")
    return cons


def main():
    os.makedirs(DATA, exist_ok=True)
    print("=" * 60)
    print("[1] 中证500 基准日线 ...", flush=True)
    bench = crawl_benchmark()
    print(f"    {bench.shape[0]} 行，{bench['date'].min().date()} ~ {bench['date'].max().date()}")

    dates = month_end_dates(bench)
    print(f"[2] 历史成分，{len(dates)} 个调仓日 ...", flush=True)
    cons = crawl_constituents(dates)

    n = cons.groupby("date").size()
    print("=" * 60)
    print(f"完成：{len(cons)} 行，每调仓日成分数 min={n.min()} max={n.max()}")
    print("产出：data/benchmark_close.csv, data/cons_hist.csv")


if __name__ == "__main__":
    main()
