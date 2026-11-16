# -*- coding: utf-8 -*-
"""
练习 03 · ROE 因子中性化对比（四版本 + ln市值参照）
========================================================

要回答的问题：ROE 因子的（不）预测力，到底来自「盈利」本身，还是来自它顺带
暴露的行业 / 市值？

对应教材《第五章 练习 03 · 中性化对比实验》：
  - 四个版本：原始 / 仅去极值 / +行业中性 / +行业+市值中性
  - 每个版本算 RankIC、ICIR、分层单调性、最大回撤
  - 额外把 ln(市值) 本身当因子，看它的 IC 作参照

五个因子版本（每个调仓日横截面各做一次）：
  A 原始           = ROE
  B 仅去极值        = MAD 3× winsorize(ROE)
  C 行业中性        = B 减「行业均值」
  D 行业+市值中性   = B 对「行业哑变量 + ln(市值)」回归取残差
  E ln市值参照      = ln(市值) 本身当因子

数据（复用练习02财报 + 新增练习03两套）：
  - 利润表 PARENT_NETPROFIT + 资产负债表 TOTAL_PARENT_EQUITY -> ROE（PIT 用 NOTICE_DATE）
  - 资产负债表 SHARE_CAPITAL（总股本，元≈股数）
  - 行业分类 industry_map.csv（申万一级，静态）
  - 不复权收盘价 raw_close_all.parquet -> 市值 = 不复权价 × 总股本
  - 后复权收盘价 daily_all.parquet -> 未来收益（收益率不受复权影响，hfq 即可）

输出：控制台打印 10 指标对比表 + 分年度 IC + 诊断结论；
      practice03/results/ 保存 4 张 CSV + 6 张 PNG。

用法（PyCharm 选 Anaconda 解释器）：
    python backtest_roe_neutral.py
注意：需先完成「不复权日线」爬取（scripts/02_crawl_raw_close.py）。
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 可调参数 =================
REBAL = 20                 # 调仓间隔（交易日）
N_GROUPS = 10              # 分层组数
MAD_N = 3.0                # 去极值：MAD 3 倍
HORIZONS = [1, 5, 10, 20, 60]   # IC 衰减窗口
MAIN_HORIZON = 20          # 主回测窗口 = 调仓周期
MIN_STOCKS = 30            # 某期有效股票少于该数则跳过

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice03/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")   # 后复权日线
UNIVERSE = os.path.join(ROOT, "practice01", "data_dynamic", "universe.csv")
FIN_DIR = os.path.join(ROOT, "practice02", "data", "financial")            # 财报三表
INDUSTRY = os.path.join(BASE_DIR, "data", "industry_map.csv")              # 申万行业
RAW_CLOSE = os.path.join(BASE_DIR, "data", "raw_close_all.parquet")        # 不复权日线
RESULTS = os.path.join(BASE_DIR, "results")

VERSIONS = ["A 原始", "B 仅去极值", "C 行业中性", "D 行业+市值中性", "E ln市值"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]   # 每版本一个颜色

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 第 1 步：数据加载 =================
def load_close():
    """后复权收盘价宽表（用于算未来收益率）。"""
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close"])
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    close.columns.name = "code"
    close.index.name = "date"
    return close


def load_raw_close(dates):
    """不复权收盘价宽表（用于算市值水平）。"""
    panel = pd.read_parquet(RAW_CLOSE, columns=["date", "code", "close"])
    raw = panel.pivot(index="date", columns="code", values="close")
    raw = raw.reindex(dates).sort_index()          # 对齐到后复权交易日历
    raw.columns.name = "code"
    raw.index.name = "date"
    return raw


def load_financials():
    """读全部利润表+资产负债表，算出每份年报的 ROE 与总股本（长表）。"""
    import glob
    profit_dir = os.path.join(FIN_DIR, "profit")
    balance_dir = os.path.join(FIN_DIR, "balance")
    files = sorted(glob.glob(os.path.join(profit_dir, "*.parquet")))

    parts = []
    for idx, pf in enumerate(files, 1):
        code = os.path.splitext(os.path.basename(pf))[0]
        bf = os.path.join(balance_dir, f"{code}.parquet")
        if not os.path.exists(bf):
            continue
        p = pd.read_parquet(pf, columns=["REPORT_DATE", "NOTICE_DATE", "PARENT_NETPROFIT"])
        b = pd.read_parquet(bf, columns=["REPORT_DATE", "TOTAL_PARENT_EQUITY", "SHARE_CAPITAL"])
        p["code"] = code
        b["code"] = code
        p["REPORT_DATE"] = pd.to_datetime(p["REPORT_DATE"])
        b["REPORT_DATE"] = pd.to_datetime(b["REPORT_DATE"])
        parts.append(p.merge(b, on=["code", "REPORT_DATE"], how="inner"))
        if idx % 1000 == 0:
            print(f"   已读财报 {idx}/{len(files)} 只 ...", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df["NOTICE_DATE"] = pd.to_datetime(df["NOTICE_DATE"], errors="coerce")
    eq = df["TOTAL_PARENT_EQUITY"]
    df["ROE"] = df["PARENT_NETPROFIT"] / eq.where(eq > 0)   # 权益<=0 无意义
    df = df.dropna(subset=["NOTICE_DATE", "ROE"])
    return df[["code", "REPORT_DATE", "NOTICE_DATE", "ROE", "SHARE_CAPITAL"]]


def build_panel(long_df, value_col, dates):
    """把长表按「公告日」前向填充成 (交易日 × 股票) 宽表 —— PIT 对齐。"""
    panel = long_df.pivot_table(index="NOTICE_DATE", columns="code",
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


# ================= 第 4 步：去极值 / 转 rank / 中性化 =================
def winsorize_mad(s, n=MAD_N):
    """MAD 3 倍去极值（NaN 自动忽略并保留）。"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    k = n * 1.4826 * mad
    return s.clip(lower=med - k, upper=med + k)


def to_rank(s):
    """转 rank（百分位 [0,1]，NaN 保留）。"""
    return s.rank(pct=True)


def neutralize(y, industry, ln_mkt=None):
    """横截面中性化：y ~ 截距 + 行业哑变量(drop_first) + [ln市值]，返回残差。

    y / industry / ln_mkt 为等长 Series（index=code）。只对三者都非空的行做 OLS，
    其余返回 NaN。残差 = 因子里「行业 / 市值解释不了」的那部分。
    """
    keep = y.notna() & industry.notna()
    if ln_mkt is not None:
        keep = keep & ln_mkt.notna()
    out = pd.Series(np.nan, index=y.index)
    if keep.sum() < 2:
        return out

    Y = y[keep].to_numpy(dtype=float)
    dum = pd.get_dummies(industry[keep], drop_first=True).to_numpy(dtype=float)
    cols = [np.ones((len(Y), 1))]
    if dum.shape[1] > 0:
        cols.append(dum)
    if ln_mkt is not None:
        cols.append(ln_mkt[keep].to_numpy(dtype=float).reshape(-1, 1))
    X = np.hstack(cols)

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    out[keep] = Y - X @ beta
    return out


# ================= 第 6/7 步：逐期检验 =================
def run_tests(close, roe_panel, shares_panel, raw_close, industry_series, mask,
              rebal_positions):
    """对五个版本逐期算 RankIC 与分层收益。"""
    # 市值 = 不复权价 × 总股本；ln(市值)（市值<=0 置 NaN）
    mktcap = raw_close * shares_panel
    ln_mkt_panel = np.log(mktcap.where(mktcap > 0))

    ic_lists = {v: {h: [] for h in HORIZONS} for v in VERSIONS}
    ic_dates = {v: {h: [] for h in HORIZONS} for v in VERSIONS}
    dec_rows = {v: [] for v in VERSIONS}
    dec_dates = {v: [] for v in VERSIONS}
    cov_sum = {v: 0.0 for v in VERSIONS}
    n_periods = 0

    for i in rebal_positions:
        date = close.index[i]
        roe = roe_panel.iloc[i]
        lnm = ln_mkt_panel.iloc[i]
        act = mask.iloc[i]
        ind = industry_series.reindex(roe.index)

        valid_A = act & roe.notna()
        n_tradable = int(act.sum())
        if valid_A.sum() < MIN_STOCKS:
            continue

        # 去极值（统一用「有 ROE 的可交易股」口径）
        roe_all = roe.where(valid_A)
        B_w = winsorize_mad(roe_all)

        # ---- 五个版本 ----
        factor = {}
        factor["A 原始"] = to_rank(roe_all)
        factor["B 仅去极值"] = to_rank(B_w)
        # C：行业中性 = 去极值后减「行业均值」（行业哑变量回归的残差等价于此）
        factor["C 行业中性"] = to_rank(B_w - B_w.groupby(ind).transform("mean"))
        # D：行业 + 市值中性
        factor["D 行业+市值中性"] = to_rank(neutralize(B_w, ind, lnm))
        # E：ln(市值) 本身当因子（同样去极值）
        factor["E ln市值"] = to_rank(winsorize_mad(lnm.where(act & lnm.notna())))

        # ---- 未来收益 ----
        close_i = close.iloc[i]
        fwd = {h: close.iloc[i + h] / close_i - 1 for h in HORIZONS}

        # ---- 逐版本：IC + 分层 ----
        for v in VERSIONS:
            fv = factor[v]
            cov_sum[v] += fv.notna().sum() / n_tradable if n_tradable else np.nan

            for h in HORIZONS:
                mh = fv.notna() & fwd[h].notna()
                if mh.sum() < MIN_STOCKS:
                    continue
                ic = fv[mh].corr(fwd[h][mh], method="spearman")
                ic_lists[v][h].append(ic)
                ic_dates[v][h].append(date)

            fwd_main = fwd[MAIN_HORIZON]
            mm = fv.notna() & fwd_main.notna()
            if mm.sum() < MIN_STOCKS:
                continue
            fv2 = fv[mm]
            fw = fwd_main[mm]
            q = pd.cut(fv2, bins=np.linspace(0, 1, N_GROUPS + 1),
                       labels=False, include_lowest=True)
            grp = fw.groupby(q).mean().reindex(range(N_GROUPS))
            dec_rows[v].append(grp.to_numpy())
            dec_dates[v].append(date)
        n_periods += 1

    ic_series = {v: {h: pd.Series(ic_lists[v][h], index=pd.DatetimeIndex(ic_dates[v][h]))
                     for h in HORIZONS} for v in VERSIONS}
    decile_rets = {}
    for v in VERSIONS:
        if dec_rows[v]:
            decile_rets[v] = pd.DataFrame(np.array(dec_rows[v]),
                                          index=pd.DatetimeIndex(dec_dates[v]),
                                          columns=range(N_GROUPS))
        else:
            decile_rets[v] = pd.DataFrame(columns=range(N_GROUPS))
    coverage = {v: (cov_sum[v] / n_periods if n_periods else np.nan) for v in VERSIONS}
    return ic_series, decile_rets, coverage


# ================= 指标统计 =================
def ic_stats(ic):
    """IC 序列汇总：均值/标准差/ICIR/t值/胜率/期数。"""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"IC均值": np.nan, "IC标准差": np.nan, "ICIR": np.nan,
                "IC_t值": np.nan, "IC>0占比": np.nan, "期数": 0}
    mean, std = ic.mean(), ic.std()
    return {"IC均值": mean, "IC标准差": std,
            "ICIR": mean / std if std > 0 else np.nan,
            "IC_t值": (mean / std) * np.sqrt(n) if std > 0 else np.nan,
            "IC>0占比": float((ic > 0).mean()), "期数": n}


def nav_metrics(rets, ppy):
    """期收益序列 -> 年化收益 / 夏普 / 最大回撤。"""
    rets = rets.dropna()
    if len(rets) == 0:
        return {"多空年化": np.nan, "多空夏普": np.nan, "多空最大回撤": np.nan}
    total = (1 + rets).prod() - 1
    ann = (1 + total) ** (ppy / len(rets)) - 1
    vol = rets.std() * np.sqrt(ppy)
    sharpe = ann / vol if vol > 0 else np.nan
    nav = (1 + rets).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    return {"多空年化": ann, "多空夏普": sharpe, "多空最大回撤": mdd}


# ================= 第 8 步：绘图 =================
def plot_ic_ts(ic_series, out):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for v, c in zip(VERSIONS, COLORS):
        ax.plot(ic_series[v][MAIN_HORIZON].index, ic_series[v][MAIN_HORIZON].values,
                lw=1.0, color=c, label=v)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"五版本 RankIC 时序对比（{MAIN_HORIZON}日）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("RankIC")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_ic_bar(comparison, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(VERSIONS))
    vals = comparison["IC均值"].values
    tvals = comparison["IC_t值"].values
    ax.bar(x, vals, color=COLORS, alpha=0.85)
    for xi, v, t in zip(x, vals, tvals):
        sig = "**" if abs(t) > 2 else ("*" if abs(t) > 1.65 else "")
        ax.text(xi, v + (0.002 if v >= 0 else -0.006),
                f"{v:.4f}{sig}\nt={t:.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(VERSIONS, fontsize=9)
    ax.set_title(f"五版本 RankIC 均值对比（{MAIN_HORIZON}日；* |t|>1.65，** |t|>2）")
    ax.set_ylabel("RankIC 均值")
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_annual(annual_df, out):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(annual_df))
    width = 0.8 / len(VERSIONS)
    for k, v in enumerate(VERSIONS):
        ax.bar(x + (k - len(VERSIONS) / 2 + 0.5) * width, annual_df[v].values,
               width, color=COLORS[k], label=v)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in annual_df.index], fontsize=8)
    ax.set_title("分年度 RankIC 均值对比")
    ax.set_ylabel("RankIC 均值"); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_decay(ic_series, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for v, c in zip(VERSIONS, COLORS):
        means = [ic_series[v][h].mean() for h in HORIZONS]
        ax.plot(HORIZONS, means, marker="o", color=c, label=v)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(HORIZONS)
    ax.set_title("五版本 IC 衰减曲线")
    ax.set_xlabel("未来收益窗口（交易日）"); ax.set_ylabel("RankIC 均值")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_ls_nav(decile_rets, out):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for v, c in zip(VERSIONS, COLORS):
        if decile_rets[v].empty:
            continue
        spread = decile_rets[v][9] - decile_rets[v][0]
        nav = (1 + spread).cumprod()
        ax.plot(nav.index, nav.values, lw=1.2, color=c, label=v)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.set_title("五版本多空（第10组−第1组）累计净值")
    ax.set_xlabel("调仓日"); ax.set_ylabel("累计净值(1为起点)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_lnmkt_panel(ic_series, decile_rets, out):
    """E ln市值 因子的完整检验四合一。"""
    ic = ic_series["E ln市值"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # 左上：IC 时序
    ax = axes[0, 0]
    ax.plot(ic[MAIN_HORIZON].index, ic[MAIN_HORIZON].values, lw=0.8, color=COLORS[4])
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"ln市值 IC 时序（{MAIN_HORIZON}日）")
    # 右上：IC 直方图
    ax = axes[0, 1]
    ax.hist(ic[MAIN_HORIZON].dropna(), bins=25, color=COLORS[4], alpha=0.8, edgecolor="white")
    ax.axvline(ic[MAIN_HORIZON].mean(), color="red", lw=2, ls="--")
    ax.set_title("ln市值 IC 分布")
    # 左下：分层净值
    ax = axes[1, 0]
    if not decile_rets["E ln市值"].empty:
        nav = (1 + decile_rets["E ln市值"]).cumprod()
        cmap = plt.cm.viridis
        for g in range(N_GROUPS):
            ax.plot(nav.index, nav[g].values, lw=0.8, color=cmap(g / (N_GROUPS - 1)))
    ax.set_title("ln市值 分层累计净值（10组）")
    # 右下：衰减
    ax = axes[1, 1]
    ax.plot(HORIZONS, [ic[h].mean() for h in HORIZONS], marker="o", color=COLORS[4])
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(HORIZONS)
    ax.set_title("ln市值 IC 衰减")
    fig.tight_layout(); fig.savefig(out, dpi=120)


def fmt(x):
    return f"{x:.4f}" if pd.notna(x) else "  nan"


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 70)
    print("练习 03 · ROE 因子中性化对比（四版本 + ln市值）")
    print("=" * 70)

    print("\n[第1步] 加载日线（后复权/不复权）...", flush=True)
    close = load_close()
    raw_close = load_raw_close(close.index)
    print(f"   后复权宽表：{close.shape}；不复权宽表：{raw_close.shape}", flush=True)

    print("[第2/3步] 加载财报 -> ROE + 总股本（PIT）...", flush=True)
    fin = load_financials()
    roe_panel = build_panel(fin, "ROE", close.index)
    shares_panel = build_panel(fin, "SHARE_CAPITAL", close.index)
    print(f"   ROE 宽表 {roe_panel.shape}；总股本宽表 {shares_panel.shape}", flush=True)

    ind_map = pd.read_csv(INDUSTRY, dtype={"code": str})
    industry_series = pd.Series(ind_map["industry"].values, index=ind_map["code"])
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    mask = build_mask(close, uni)

    rebal_positions = list(range(0, len(close) - max(HORIZONS), REBAL))
    print(f"   调仓期数：{len(rebal_positions)}（每 {REBAL} 个交易日一次）", flush=True)

    print("[第4~7步] 五版本 -> 去极值/中性化 -> RankIC -> 分层 ...", flush=True)
    ic_series, decile_rets, coverage = run_tests(
        close, roe_panel, shares_panel, raw_close, industry_series, mask, rebal_positions)

    # ---- 核心对比表 ----
    rows = []
    for v in VERSIONS:
        st = ic_stats(ic_series[v][MAIN_HORIZON])
        dr = decile_rets[v]
        mean_dec = dr.mean()
        mono = mean_dec.reset_index(drop=True).corr(pd.Series(range(N_GROUPS)),
                                                    method="spearman")
        nm = nav_metrics(dr[9] - dr[0], 252 / REBAL)
        rows.append({"版本": v,
                     "IC均值": st["IC均值"], "IC标准差": st["IC标准差"],
                     "ICIR": st["ICIR"], "IC_t值": st["IC_t值"], "IC胜率": st["IC>0占比"],
                     "分层单调性": mono,
                     "多空年化": nm["多空年化"], "多空夏普": nm["多空夏普"],
                     "多空最大回撤": nm["多空最大回撤"], "覆盖率": coverage[v]})
    comp = pd.DataFrame(rows).set_index("版本")

    print("\n" + "=" * 70)
    print("[核心对比表] 五版本 × 10 指标（主窗口 20 日）")
    print("=" * 70)
    print(comp.map(fmt).to_string())

    # ---- 分年度 IC ----
    annual = {v: ic_series[v][MAIN_HORIZON].groupby(
        ic_series[v][MAIN_HORIZON].index.year).mean() for v in VERSIONS}
    annual_df = pd.DataFrame(annual)      # index=年份, columns=版本
    print("\n" + "=" * 70)
    print("[分年度 IC] 各版本 RankIC 均值")
    print("=" * 70)
    print(annual_df.map(fmt).to_string())

    # ---- 诊断结论 ----
    icA = comp.loc["A 原始", "IC均值"]
    icB = comp.loc["B 仅去极值", "IC均值"]
    icC = comp.loc["C 行业中性", "IC均值"]
    icD = comp.loc["D 行业+市值中性", "IC均值"]
    icE = comp.loc["E ln市值", "IC均值"]
    print("\n" + "=" * 70)
    print("[诊断结论]")
    print("=" * 70)
    print(f"  去极值影响  A -> B : IC {icA:+.4f} -> {icB:+.4f}  变化 {icB - icA:+.4f}")
    print(f"  行业暴露影响 B -> C : IC {icB:+.4f} -> {icC:+.4f}  变化 {icC - icB:+.4f}")
    print(f"  市值暴露影响 C -> D : IC {icC:+.4f} -> {icD:+.4f}  变化 {icD - icC:+.4f}")
    print(f"  ln(市值)自身 IC = {icE:+.4f}（作为市值暴露强度的参照）")
    if abs(icD) < 0.5 * abs(icA) and abs(icA) > 0.005:
        print("  → 判定：中性化后 IC 掉一半以上，原因子主要是市值/行业暴露。")
    elif abs(icD) > abs(icA) and abs(icD) > 0.005:
        print("  → 判定：中性化后 IC 不降反升，ROE 的盈利信号之前被暴露掩盖了。")
    else:
        print("  → 判定：ROE 在原始口径下本就没有显著信号，中性化未改变这一结论。")

    # ---- 保存 ----
    print("\n[第8步] 保存图表到 results/ ...", flush=True)
    plot_ic_ts(ic_series, os.path.join(RESULTS, "ROE中性化_1_IC时序对比.png"))
    plot_ic_bar(comp, os.path.join(RESULTS, "ROE中性化_2_IC均值对比.png"))
    plot_annual(annual_df, os.path.join(RESULTS, "ROE中性化_3_分年度IC对比.png"))
    plot_decay(ic_series, os.path.join(RESULTS, "ROE中性化_4_IC衰减对比.png"))
    plot_ls_nav(decile_rets, os.path.join(RESULTS, "ROE中性化_5_多空净值对比.png"))
    plot_lnmkt_panel(ic_series, decile_rets, os.path.join(RESULTS, "ROE中性化_6_ln市值完整检验.png"))

    comp.to_csv(os.path.join(RESULTS, "ROE中性化_核心对比表.csv"), encoding="utf-8-sig")
    annual_df.to_csv(os.path.join(RESULTS, "ROE中性化_分年度IC.csv"), encoding="utf-8-sig")
    pd.DataFrame({v: ic_series[v][MAIN_HORIZON] for v in VERSIONS}).to_csv(
        os.path.join(RESULTS, "ROE中性化_IC时序.csv"), encoding="utf-8-sig")
    for v in VERSIONS:
        decile_rets[v].to_csv(os.path.join(RESULTS, f"ROE中性化_分层收益_{v.replace(' ', '')}.csv"),
                              encoding="utf-8-sig")

    print("已保存 6 张图 + 若干 CSV。")
    print("\n[验收提示] 看核心对比表：")
    print("  - 若 C/D 的 IC 相比 B 明显变化，说明行业/市值暴露贡献大；")
    print("  - 对照 E(ln市值) 的 IC，判断市值暴露有多强；")
    print("  - 若 D 版本 IC 仍为负/为零，说明 ROE 在这个样本里没有独立 Alpha。")
    plt.show()


if __name__ == "__main__":
    main()
