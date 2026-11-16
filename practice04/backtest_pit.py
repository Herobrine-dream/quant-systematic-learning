# -*- coding: utf-8 -*-
"""
练习 04 · 财务因子与 PIT 生死线
========================================================

要回答的问题：同一个 ROE 因子，用「报告期」对齐 vs 用「公告日」对齐，IC 差多少？
（这个差值就是"未来函数"造成的虚高，教材称其"会很难看，但很有教育意义"。）

对应教材《第五章 练习 04》：
  1. ROE 因子，分别按 报告期(REPORT_DATE) / 公告日(NOTICE_DATE) 对齐，对比 IC；
  2. 再构造 EP(1/PE)、毛利率、营收同比增速（都用公告日对齐）；
  3. 计算这四个因子的相关性矩阵；
  4. 数据字典见《报告期与公告日详解.md》。

数据：
  - 财务指标 practice04/data/indicator/*.parquet（ROEJQ / EPSJB / XSMLL / 营收同比）
  - 后复权日线 daily_all.parquet（未来收益）
  - 不复权日线 raw_close_all.parquet（算 EP 的市值/价格）
  - 统一用「年报」口径（年报披露滞后最大，PIT 差异最明显；详见详解文档）

用法（PyCharm 选 Anaconda 解释器）：
    python backtest_pit.py
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 参数 =================
REBAL = 20              # 调仓间隔（交易日）
MAIN_HORIZON = 20       # 未来收益窗口
MIN_STOCKS = 30         # 某期有效股票少于该数则跳过
MAD_N = 3.0             # 去极值

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice04/
ROOT = os.path.dirname(BASE_DIR)
IND_DIR = os.path.join(BASE_DIR, "data", "indicator")
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")
RAW_CLOSE = os.path.join(ROOT, "practice03", "data", "raw_close_all.parquet")
UNIVERSE = os.path.join(ROOT, "practice01", "data_dynamic", "universe.csv")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 数据加载 =================
def load_indicator():
    """读全部财务指标文件 -> 长表（只留年报）。"""
    files = sorted(glob.glob(os.path.join(IND_DIR, "*.parquet")))
    cols = ["REPORT_DATE", "NOTICE_DATE", "REPORT_DATE_NAME",
            "ROEJQ", "EPSJB", "XSMLL", "TOTALOPERATEREVETZ"]
    parts = []
    for f in files:
        code = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_parquet(f, columns=cols)
        df["code"] = code
        parts.append(df)
    ind = pd.concat(parts, ignore_index=True)
    ind["REPORT_DATE"] = pd.to_datetime(ind["REPORT_DATE"], errors="coerce")
    ind["NOTICE_DATE"] = pd.to_datetime(ind["NOTICE_DATE"], errors="coerce")
    # 只用年报（披露滞后最大、PIT 差异最明显；避免混入单季/中报的 ROE 尺度不一致）
    ind = ind[ind["REPORT_DATE_NAME"].astype(str).str.contains("年报", na=False)]
    return ind.dropna(subset=["NOTICE_DATE"])


def load_close():
    """后复权收盘价宽表（未来收益用）。"""
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close"])
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    close.columns.name = "code"
    close.index.name = "date"
    return close


def load_raw_close(dates):
    """不复权收盘价宽表（EP 的估值水平用）。"""
    panel = pd.read_parquet(RAW_CLOSE, columns=["date", "code", "close"])
    raw = panel.pivot(index="date", columns="code", values="close").reindex(dates)
    raw.columns.name = "code"
    raw.index.name = "date"
    return raw


def build_panel(long_df, value_col, date_col, dates):
    """长表 -> (交易日 × 股票) 宽表，按 date_col 前向填充（PIT）。"""
    panel = long_df.pivot_table(index=date_col, columns="code",
                                values=value_col, aggfunc="last")
    panel = panel.reindex(panel.index.union(dates)).sort_index().ffill()
    panel = panel.reindex(dates)
    panel.columns.name = "code"
    panel.index.name = "date"
    return panel


def build_mask(close, uni):
    """动态可交易掩码：上市日 <= t < 退市日。"""
    uni = uni.copy()
    uni["list_date"] = pd.to_datetime(uni["list_date"], errors="coerce")
    uni["delist_date"] = pd.to_datetime(uni["delist_date"], errors="coerce")
    meta = uni.set_index("code").reindex(close.columns)
    ld = meta["list_date"].to_numpy()
    dd = meta["delist_date"].to_numpy().copy()
    dd[np.isnat(dd)] = np.datetime64("2099-12-31", "ns")
    dates = close.index.to_numpy()
    return pd.DataFrame((dates[:, None] >= ld[None, :]) & (dates[:, None] < dd[None, :]),
                        index=close.index, columns=close.columns)


# ================= 因子处理 =================
def winsorize_mad(s, n=MAD_N):
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    k = n * 1.4826 * mad
    return s.clip(lower=med - k, upper=med + k)


def to_rank(s):
    return s.rank(pct=True)


# ================= 检验 =================
def compute_ic(factor_panel, close, mask, rebal_positions):
    """算因子 RankIC 序列（去极值 + rank 后，与未来 20 日收益的 Spearman）。"""
    ic_list, dates = [], []
    for i in rebal_positions:
        f = factor_panel.iloc[i].where(mask.iloc[i])
        if f.notna().sum() < MIN_STOCKS:
            continue
        f_rank = to_rank(winsorize_mad(f))
        fwd = close.iloc[i + MAIN_HORIZON] / close.iloc[i] - 1
        vm = f_rank.notna() & fwd.notna()
        if vm.sum() < MIN_STOCKS:
            continue
        ic = f_rank[vm].corr(fwd[vm], method="spearman")
        ic_list.append(ic)
        dates.append(close.index[i])
    return pd.Series(ic_list, index=pd.DatetimeIndex(dates), name="IC")


def ic_stats(ic):
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"IC均值": np.nan, "ICIR": np.nan, "IC_t值": np.nan, "IC>0占比": np.nan}
    mean, std = ic.mean(), ic.std()
    return {"IC均值": mean, "ICIR": mean / std if std > 0 else np.nan,
            "IC_t值": (mean / std) * np.sqrt(n) if std > 0 else np.nan,
            "IC>0占比": float((ic > 0).mean())}


def factor_corr_matrix(panels, close, mask, rebal_positions):
    """各因子两两的秩相关（时序平均）—— 练习4第4步。"""
    names = list(panels.keys())
    n = len(names)
    acc = np.zeros((n, n))
    cnt = 0
    for i in rebal_positions:
        act = mask.iloc[i]
        ranks = {}
        for name in names:
            f = panels[name].iloc[i].where(act)
            ranks[name] = to_rank(winsorize_mad(f))
        corr = pd.DataFrame(ranks).corr(method="spearman")
        acc += corr.to_numpy()
        cnt += 1
    return pd.DataFrame(acc / cnt if cnt else acc, index=names, columns=names)


def fmt(x):
    return f"{x:.4f}" if pd.notna(x) else "  nan"


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 70)
    print("练习 04 · 财务因子与 PIT 生死线")
    print("=" * 70)

    print("\n[加载数据] ...", flush=True)
    close = load_close()
    raw_close = load_raw_close(close.index)
    ind = load_indicator()
    print(f"   后复权 {close.shape}；不复权 {raw_close.shape}；年报指标 {len(ind)} 条", flush=True)

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    mask = build_mask(close, uni)

    # ---- 构造因子面板 ----
    print("[构造因子] ...", flush=True)
    # ROE：两种对齐（报告期 vs 公告日）——练习4核心
    roe_report = build_panel(ind, "ROEJQ", "REPORT_DATE", close.index)   # 偷看未来
    roe_notice = build_panel(ind, "ROEJQ", "NOTICE_DATE", close.index)   # 正确 PIT
    # EP / 毛利率 / 营收同比（都按公告日对齐）
    eps_notice = build_panel(ind, "EPSJB", "NOTICE_DATE", close.index)
    ep_notice = eps_notice / raw_close                                  # EP = EPS/股价
    xsmll_notice = build_panel(ind, "XSMLL", "NOTICE_DATE", close.index)
    totrev_notice = build_panel(ind, "TOTALOPERATEREVETZ", "NOTICE_DATE", close.index)

    rebal_positions = list(range(0, len(close) - MAIN_HORIZON, REBAL))
    print(f"   调仓期数：{len(rebal_positions)}", flush=True)

    # ---- IC ----
    print("[计算 IC] ...", flush=True)
    ic_roe_report = compute_ic(roe_report, close, mask, rebal_positions)
    ic_roe_notice = compute_ic(roe_notice, close, mask, rebal_positions)
    ic_ep = compute_ic(ep_notice, close, mask, rebal_positions)
    ic_xsmll = compute_ic(xsmll_notice, close, mask, rebal_positions)
    ic_totrev = compute_ic(totrev_notice, close, mask, rebal_positions)

    # ---- 第1步：ROE 两种对齐对比（核心） ----
    print("\n" + "=" * 70)
    print("[第1步·核心] ROE 因子：报告期 vs 公告日 对齐的 IC 对比")
    print("=" * 70)
    cmp = pd.DataFrame({
        "对齐方式": ["按报告期(偷看未来)", "按公告日(正确PIT)"],
        "IC均值": [ic_stats(ic_roe_report)["IC均值"], ic_stats(ic_roe_notice)["IC均值"]],
        "ICIR": [ic_stats(ic_roe_report)["ICIR"], ic_stats(ic_roe_notice)["ICIR"]],
        "IC_t值": [ic_stats(ic_roe_report)["IC_t值"], ic_stats(ic_roe_notice)["IC_t值"]],
        "IC>0占比": [ic_stats(ic_roe_report)["IC>0占比"], ic_stats(ic_roe_notice)["IC>0占比"]],
    }).set_index("对齐方式")
    print(cmp.map(fmt).to_string())
    diff = cmp.loc["按报告期(偷看未来)", "IC均值"] - cmp.loc["按公告日(正确PIT)", "IC均值"]
    print(f"\n  IC 差值（报告期 - 公告日）= {diff:+.4f}")
    print("  若为正：说明按报告期对齐高估了预测力（未来函数）。")

    # ---- 第2步：其余三因子（公告日对齐） ----
    print("\n" + "=" * 70)
    print("[第2步] 其余三因子（均按公告日对齐）的 IC")
    print("=" * 70)
    others = pd.DataFrame({
        "因子": ["EP(1/PE)", "毛利率", "营收同比增速"],
        "IC均值": [ic_stats(ic_ep)["IC均值"], ic_stats(ic_xsmll)["IC均值"],
                  ic_stats(ic_totrev)["IC均值"]],
        "ICIR": [ic_stats(ic_ep)["ICIR"], ic_stats(ic_xsmll)["ICIR"],
                 ic_stats(ic_totrev)["ICIR"]],
        "IC_t值": [ic_stats(ic_ep)["IC_t值"], ic_stats(ic_xsmll)["IC_t值"],
                   ic_stats(ic_totrev)["IC_t值"]],
    }).set_index("因子")
    print(others.map(fmt).to_string())

    # ---- 第3步：四因子相关性矩阵 ----
    print("\n" + "=" * 70)
    print("[第3步] 四因子相关性矩阵（秩相关，时序平均；均公告日对齐）")
    print("=" * 70)
    panels = {"ROE": roe_notice, "EP": ep_notice, "毛利率": xsmll_notice, "营收同比": totrev_notice}
    corr = factor_corr_matrix(panels, close, mask, rebal_positions)
    print(corr.map(fmt).to_string())

    # ---- 保存 ----
    print("\n[保存] ...", flush=True)
    cmp.to_csv(os.path.join(RESULTS, "PIT对比_ROE对齐方式.csv"), encoding="utf-8-sig")
    others.to_csv(os.path.join(RESULTS, "PIT对比_其余三因子IC.csv"), encoding="utf-8-sig")
    corr.to_csv(os.path.join(RESULTS, "PIT对比_四因子相关性.csv"), encoding="utf-8-sig")
    pd.DataFrame({"报告期": ic_roe_report, "公告日": ic_roe_notice}).to_csv(
        os.path.join(RESULTS, "PIT对比_ROE_IC时序.csv"), encoding="utf-8-sig")

    # ---- 图1：ROE 报告期 vs 公告日 IC 时序 ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ic_roe_report.index, ic_roe_report.values, lw=1.0, color="#d62728",
            label="按报告期(偷看未来)")
    ax.plot(ic_roe_notice.index, ic_roe_notice.values, lw=1.0, color="#1f77b4",
            label="按公告日(正确PIT)")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"ROE 因子：报告期 vs 公告日 对齐的 RankIC 时序（{MAIN_HORIZON}日）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("RankIC")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "PIT对比_1_ROE_IC时序.png"), dpi=120)

    # ---- 图2：分年度 IC 对比 ----
    annual = pd.DataFrame({
        "按报告期": ic_roe_report.groupby(ic_roe_report.index.year).mean(),
        "按公告日": ic_roe_notice.groupby(ic_roe_notice.index.year).mean(),
    })
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(annual))
    w = 0.38
    ax.bar(x - w / 2, annual["按报告期"].values, w, color="#d62728", label="按报告期(偷看未来)")
    ax.bar(x + w / 2, annual["按公告日"].values, w, color="#1f77b4", label="按公告日(正确PIT)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in annual.index], fontsize=8)
    ax.set_title("ROE 因子：分年度 IC 对比（报告期 vs 公告日）")
    ax.set_ylabel("RankIC 均值"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "PIT对比_2_分年度IC.png"), dpi=120)

    print("已保存到 results/。")
    print("\n[验收提示] 教材练习04：看两种对齐的 IC 差值（越正越说明未来函数高估）；")
    print("  再结合四因子相关性矩阵，完成数据字典（见《报告期与公告日详解.md》）。")
    plt.show()


if __name__ == "__main__":
    main()
