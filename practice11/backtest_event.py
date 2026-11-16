# -*- coding: utf-8 -*-
"""
练习 11 · 事件驱动：业绩预告超预期（事件研究法）
================================================================

流程（对应题目 5 步）：
  1. 取业绩预告（data/yjyg/*.csv，含预告类型/变动幅度/公告日期）
  2. 定义事件：预告类型=「预增」且 变动幅度 > 同行业×同报告期 的预增幅度中位数
  3. 事件后 1/5/20/60 日超额收益 AR = 个股日收益 − 中证500日收益
  4. 画 CAR 曲线（平均 AR 逐日累加）+ 分年度

关键口径：
  - 事件日 t0 = 公告日期之后第一个交易日（A股预告通常盘后公告，次日才可交易）
  - 去重：预测指标只留「归属于上市公司股东的净利润」（去掉扣非）；同公司同报告期留最早公告
  - 基准：中证500（data/benchmark_zz500.csv，2015-2024）
  - 个股收益用后复权（hfq）日线

用法：先跑 scripts/01_crawl_yjyg.py，再跑本脚本。
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice11/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
YJYG_DIR = os.path.join(BASE_DIR, "data", "yjyg")
BENCHMARK = os.path.join(BASE_DIR, "data", "benchmark_zz500.csv")
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")
INDUSTRY = os.path.join(ROOT, "practice03", "data", "industry_map.csv")
RESULTS = os.path.join(BASE_DIR, "results")

WINDOWS = [1, 5, 20, 60]          # 事件后观察窗口（交易日）
MAX_HORIZON = 60                  # 最长窗口

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 加载 =================
def load_yjyg():
    files = sorted(glob.glob(os.path.join(YJYG_DIR, "*.csv")))
    parts = []
    for f in files:
        period = os.path.splitext(os.path.basename(f))[0]      # 报告期，如 20151231
        df = pd.read_csv(f, dtype={"股票代码": str})
        df["报告期"] = period
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    raw["股票代码"] = raw["股票代码"].str.zfill(6)
    raw["公告日期"] = pd.to_datetime(raw["公告日期"], errors="coerce")
    raw["业绩变动幅度"] = pd.to_numeric(raw["业绩变动幅度"], errors="coerce")
    # 去重1：只留「归属于上市公司股东的净利润」口径（去掉扣非）
    raw = raw[raw["预测指标"].astype(str).str.contains("归属于上市公司股东的净利润", na=False)]
    # 去重2：同公司同报告期多次预告，留最早公告
    raw = raw.sort_values("公告日期").drop_duplicates(subset=["股票代码", "报告期"], keep="first")
    return raw


def load_close():
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close"])
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    close.columns.name = "code"; close.index.name = "date"
    return close


def load_benchmark():
    s = pd.read_csv(BENCHMARK, parse_dates=["date"]).set_index("date")["close"].sort_index()
    return s


def load_industry():
    ind = pd.read_csv(INDUSTRY, dtype={"code": str})
    return pd.Series(ind["industry"].values, index=ind["code"])


# ================= 绘图 =================
def plot_car(car_curve, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(car_curve.index, car_curve.values, color="#d62728", lw=1.6)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_title("业绩预告超预期 事件后平均累计超额收益（CAR，相对中证500）")
    ax.set_xlabel("事件后交易日"); ax.set_ylabel("平均 CAR")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_yearly(yearly, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly.index.astype(str), yearly["CAR20"].values, color="#1f77b4")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_title("分年度 CAR(20日)（事件后 20 日平均累计超额）")
    ax.set_xlabel("年份"); ax.set_ylabel("平均 CAR(20)")
    for i, (y, r) in enumerate(yearly.iterrows()):
        ax.text(i, r["CAR20"], f"{r['CAR20']:+.2%}\n(n={int(r['样本数'])})",
                ha="center", va="bottom" if r["CAR20"] > 0 else "top", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("练习 11 · 事件驱动：业绩预告超预期")
    print("=" * 72)

    print("\n[1] 加载数据 ...", flush=True)
    yj = load_yjyg()
    print(f"    预告原始（去重后）：{len(yj)} 条", flush=True)

    industry = load_industry()
    yj["行业"] = yj["股票代码"].map(industry)
    yj = yj.dropna(subset=["行业", "公告日期", "业绩变动幅度"])

    close = load_close()
    bench = load_benchmark()
    calendar = close.index
    bench = bench.reindex(calendar).ffill()
    stock_ret = close.pct_change()
    bench_ret = bench.pct_change()
    # date -> 位置 映射
    pos_of = {d: i for i, d in enumerate(calendar)}

    print("[2] 定义事件：预增 且 幅度 > 同行业×同报告期中位数 ...", flush=True)
    yj = yj[yj["预告类型"].astype(str).str.strip() == "预增"].copy()
    med = yj.groupby(["行业", "报告期"])["业绩变动幅度"].median().rename("中位数")
    yj = yj.join(med, on=["行业", "报告期"])
    events = yj[yj["业绩变动幅度"] > yj["中位数"]].copy()
    print(f"    预增样本 {len(yj)} 条；超预期事件 {len(events)} 条", flush=True)

    print("[3] 算事件后 AR / CAR ...", flush=True)
    rows, event_years = [], []
    for _, e in events.iterrows():
        code, ad = e["股票代码"], e["公告日期"]
        if code not in close.columns:
            continue
        # t0 = 公告日期之后第一个交易日（盘后公告，次日才可交易）
        i0 = pos_of.get(ad, None)
        if i0 is None:
            # 公告日非交易日：取之后第一个交易日
            idx = calendar.searchsorted(ad, side="right")
            i0 = int(idx) if idx < len(calendar) else None
        else:
            i0 = i0 + 1  # 公告日当天收盘后发布，从次日算
        if i0 is None or i0 + MAX_HORIZON >= len(calendar):
            continue
        sr = stock_ret[code].to_numpy()
        br = bench_ret.to_numpy()
        ar = sr[i0:i0 + MAX_HORIZON] - br[i0:i0 + MAX_HORIZON]
        if np.isnan(ar).sum() > MAX_HORIZON * 0.3:   # 停牌太多则跳过
            continue
        rows.append(ar)
        event_years.append(ad.year)
    if not rows:
        print("    无有效事件，退出", flush=True); return
    ar_mat = pd.DataFrame(np.vstack(rows), columns=range(1, MAX_HORIZON + 1))
    ar_mat = ar_mat.fillna(0.0)
    car = ar_mat.cumsum(axis=1)
    car_curve = car.mean(axis=0)
    event_years = pd.Series(event_years, name="年份")
    print(f"    有效事件 {len(car)} 个", flush=True)

    # ---- 各窗口 CAR ----
    print("\n" + "=" * 72)
    print("[事件后累计超额收益 CAR]")
    print("=" * 72)
    car_table = {}
    for w in WINDOWS:
        v = car_curve[w]                       # 列名=第 w 天（1起）
        car_table[w] = v
        print(f"  CAR({w:>2}日) = {v:+.4%}")
    # t 检验（CAR 均值是否显著不为 0）
    for w in WINDOWS:
        c = car[w]
        t = c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))
        print(f"    CAR({w:>2}日) t = {t:+.2f}   (n={len(c)})")

    # ---- 分年度 ----
    print("\n[分年度 CAR(20日) 稳定性]")
    yearly = []
    for y, idx in event_years.groupby(event_years.values).groups.items():
        sub = car.loc[idx, 20]
        yearly.append({"年份": y, "CAR20": sub.mean(), "样本数": len(sub)})
    yearly = pd.DataFrame(yearly).sort_values("年份").set_index("年份")
    for y, r in yearly.iterrows():
        print(f"  {y}  CAR20={r['CAR20']:+.4%}  n={int(r['样本数'])}")

    # ---- 保存 + 绘图 ----
    print("\n[4] 保存结果 ...", flush=True)
    car_curve.to_csv(os.path.join(RESULTS, "CAR曲线.csv"), encoding="utf-8-sig", header=["CAR"])
    pd.DataFrame({"窗口(日)": WINDOWS, "CAR": [car_table[w] for w in WINDOWS]}).to_csv(
        os.path.join(RESULTS, "CAR汇总.csv"), encoding="utf-8-sig", index=False)
    yearly.to_csv(os.path.join(RESULTS, "分年度CAR.csv"), encoding="utf-8-sig")
    plot_car(car_curve, os.path.join(RESULTS, "CAR曲线.png"))
    plot_yearly(yearly, os.path.join(RESULTS, "分年度CAR.png"))

    print("\n[验收提示] 看 results/CAR曲线.png + 分年度CAR.png + CAR汇总.csv + 分年度CAR.csv")
    plt.show()


if __name__ == "__main__":
    main()
