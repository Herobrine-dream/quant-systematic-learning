# -*- coding: utf-8 -*-
"""
练习 05 · 多因子合成并跑赢等权基准
========================================================

要回答的问题：把 5 个逻辑不同的因子合成一个组合，三种加权方式
（等权 / IC 加权 / ICIR 加权），复杂方法到底有没有跑赢等权？

流程（对应教材《第五章 练习 05》）：
  1. 选 5 个因子：反转(20d) / 动量(120d) / 换手(volume/总股本) / ROE / EP；
  2. 每个调仓日横截面「行业 + ln市值」中性化 -> MAD 去极值 -> rank；
  3. 两两秩相关，剔除 |相关性| > 0.7 的因子；
  4. 三种合成：等权 / IC 加权 / ICIR 加权（权重只用过去的 IC，无未来函数）；
  5. 分层回测（买合成分前 20%，含 0.1% 单边成本），对比组合 ICIR。

数据（全部复用练习 01~04 已爬好的文件，零新增爬取）：
  - daily_all.parquet          后复权 close/volume（反转/动量/换手/未来收益）
  - raw_close_all.parquet      不复权 close（EP、市值）
  - indicator/*.parquet        ROEJQ / EPSJB（ROE、EP，公告日 PIT）
  - financial/balance/*.parquet SHARE_CAPITAL（总股本，换手率分母 + 市值）
  - industry_map.csv           申万一级行业（中性化哑变量）
  - universe.csv               动态可交易掩码（上市日 ~ 退市日）

用法（PyCharm 选 Anaconda 解释器）：
    python backtest_composite.py
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ================= 可调参数 =================
REBAL = 20                 # 调仓间隔（交易日）
MAIN_HORIZON = 20          # 未来收益窗口（= 调仓周期）
REV_WIN = 20               # 反转回看窗口
MOM_WIN = 120              # 动量回看窗口（避开与 20 日反转过高相关）
TURN_WIN = 20              # 换手率用 20 日均成交量平滑
TOP_FRAC = 0.2             # 买合成分前 20%
COST = 0.001               # 单边交易成本 0.1%
CORR_THRESH = 0.7          # 两两相关性超过该值剔除其一
WARMUP = 12                # IC/ICIR 加权需要的预热期数（约 1 年）
MAD_N = 3.0                # 去极值：MAD 3 倍
MIN_STOCKS = 30            # 某期有效股票少于该数则跳过

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice05/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")
RAW_CLOSE = os.path.join(ROOT, "practice03", "data", "raw_close_all.parquet")
UNIVERSE = os.path.join(ROOT, "practice01", "data_dynamic", "universe.csv")
IND_DIR = os.path.join(ROOT, "practice04", "data", "indicator")
BAL_DIR = os.path.join(ROOT, "practice02", "data", "financial", "balance")
INDUSTRY = os.path.join(ROOT, "practice03", "data", "industry_map.csv")
RESULTS = os.path.join(BASE_DIR, "results")

METHODS = ["等权", "IC加权", "ICIR加权"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 数据加载 =================
def load_close_volume():
    """后复权 close + volume 宽表（未来收益 / 反转 / 动量 / 换手分子）。"""
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close", "volume"])
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    volume = panel.pivot(index="date", columns="code", values="volume").sort_index()
    for df in (close, volume):
        df.columns.name = "code"
        df.index.name = "date"
    return close, volume


def load_raw_close(dates):
    """不复权收盘价宽表（EP 分母 / 市值）。"""
    panel = pd.read_parquet(RAW_CLOSE, columns=["date", "code", "close"])
    raw = panel.pivot(index="date", columns="code", values="close").reindex(dates).sort_index()
    raw.columns.name = "code"
    raw.index.name = "date"
    return raw


def load_indicator():
    """财务指标长表（只留年报），ROEJQ + EPSJB。"""
    files = sorted(glob.glob(os.path.join(IND_DIR, "*.parquet")))
    cols = ["REPORT_DATE", "NOTICE_DATE", "REPORT_DATE_NAME", "ROEJQ", "EPSJB"]
    parts = []
    for f in files:
        code = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_parquet(f, columns=cols)
        df["code"] = code
        parts.append(df)
    ind = pd.concat(parts, ignore_index=True)
    ind["REPORT_DATE"] = pd.to_datetime(ind["REPORT_DATE"], errors="coerce")
    ind["NOTICE_DATE"] = pd.to_datetime(ind["NOTICE_DATE"], errors="coerce")
    ind = ind[ind["REPORT_DATE_NAME"].astype(str).str.contains("年报", na=False)]
    return ind.dropna(subset=["NOTICE_DATE"])


def load_shares():
    """总股本长表（换手率分母 + 市值），只留年报，公告日 PIT。"""
    files = sorted(glob.glob(os.path.join(BAL_DIR, "*.parquet")))
    cols = ["NOTICE_DATE", "REPORT_DATE_NAME", "SHARE_CAPITAL"]
    parts = []
    for f in files:
        code = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_parquet(f, columns=cols)
        df["code"] = code
        parts.append(df)
    sh = pd.concat(parts, ignore_index=True)
    sh["NOTICE_DATE"] = pd.to_datetime(sh["NOTICE_DATE"], errors="coerce")
    sh = sh[sh["REPORT_DATE_NAME"].astype(str).str.contains("年报", na=False)]
    sh = sh.dropna(subset=["NOTICE_DATE", "SHARE_CAPITAL"])
    return sh[["code", "NOTICE_DATE", "SHARE_CAPITAL"]]


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


def neutralize(y, industry, ln_mkt):
    """横截面中性化：y ~ 截距 + 行业哑变量 + ln市值，返回残差。"""
    keep = y.notna() & industry.notna() & ln_mkt.notna()
    out = pd.Series(np.nan, index=y.index)
    if keep.sum() < 2:
        return out
    Y = y[keep].to_numpy(dtype=float)
    dum = pd.get_dummies(industry[keep], drop_first=True).to_numpy(dtype=float)
    cols = [np.ones((len(Y), 1))]
    if dum.shape[1] > 0:
        cols.append(dum)
    cols.append(ln_mkt[keep].to_numpy(dtype=float).reshape(-1, 1))
    X = np.hstack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    out[keep] = Y - X @ beta
    return out


# ================= 指标统计 =================
def ic_stats(ic):
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


def annualize(rets, ppy=252 / REBAL):
    rets = rets.dropna()
    if len(rets) == 0:
        return np.nan
    total = (1 + rets).prod() - 1
    return (1 + total) ** (ppy / len(rets)) - 1


def max_drawdown(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return np.nan
    nav = (1 + rets).cumprod()
    return (nav / nav.cummax() - 1).min()


def drop_correlated(corr, ic_abs, names, thresh=CORR_THRESH):
    """贪心剔除 |相关性|>thresh 的因子（每次去掉信号更弱的一方）。"""
    keep = list(names)
    dropped = []
    while True:
        sub = corr.loc[keep, keep].abs()
        np.fill_diagonal(sub.values, 0.0)
        if sub.size == 0 or sub.max().max() <= thresh:
            break
        i, j = np.unravel_index(np.argmax(sub.values), sub.shape)
        a, b = sub.index[i], sub.columns[j]
        drop = a if ic_abs.get(a, 0.0) <= ic_abs.get(b, 0.0) else b
        dropped.append((drop, a, b, corr.loc[a, b]))
        keep.remove(drop)
    return keep, dropped


def fmt(x):
    return f"{x:.4f}" if pd.notna(x) else "  nan"


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("练习 05 · 多因子合成（等权 / IC加权 / ICIR加权）")
    print("=" * 72)

    # ---- 加载 ----
    print("\n[1] 加载数据 ...", flush=True)
    close, volume = load_close_volume()
    raw_close = load_raw_close(close.index).reindex(columns=close.columns)
    ind = load_indicator()
    shares = load_shares()
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    mask = build_mask(close, uni)
    ind_map = pd.read_csv(INDUSTRY, dtype={"code": str})
    industry = pd.Series(ind_map["industry"].values, index=ind_map["code"])
    print(f"    后复权 {close.shape}；不复权 {raw_close.shape}；指标 {len(ind)}；股本 {len(shares)}",
          flush=True)

    # ---- 构造因子面板 ----
    print("[2] 构造 5 个因子面板 ...", flush=True)
    shares_panel = build_panel(shares, "SHARE_CAPITAL", "NOTICE_DATE", close.index)
    shares_panel = shares_panel.reindex(columns=close.columns)
    roe_panel = build_panel(ind, "ROEJQ", "NOTICE_DATE", close.index).reindex(columns=close.columns)
    eps_panel = build_panel(ind, "EPSJB", "NOTICE_DATE", close.index).reindex(columns=close.columns)

    rev = -(close / close.shift(REV_WIN) - 1)                    # 反转：跌多 -> 买
    mom = close / close.shift(MOM_WIN) - 1                       # 动量：涨多 -> 买
    turn = -(volume.rolling(TURN_WIN).mean() / shares_panel)     # 换手：高换手 -> 卖
    ep = eps_panel / raw_close                                   # EP = 每股收益/价
    factor_panels = {"反转": rev, "动量": mom, "换手": turn, "ROE": roe_panel, "EP": ep}
    names = list(factor_panels.keys())

    mktcap = raw_close * shares_panel
    ln_mkt = np.log(mktcap.where(mktcap > 0))

    rebal = list(range(MOM_WIN, len(close) - MAIN_HORIZON, REBAL))
    print(f"    调仓期数：{len(rebal)}（每 {REBAL} 交易日一次，从第 {MOM_WIN} 日起）", flush=True)

    # ---- 第一遍：中性化 + rank，算相关性 + 单因子 IC ----
    print("[3] 中性化(行业+ln市值) + rank，算相关性矩阵 ...", flush=True)
    rank_lists = {n: [] for n in names}
    ic_lists = {n: [] for n in names}
    ic_dates = []
    corr_acc = np.zeros((len(names), len(names)))

    for idx_i, i in enumerate(rebal):
        date = close.index[i]
        act = mask.iloc[i]
        lnm = ln_mkt.iloc[i]
        fwd = close.iloc[i + MAIN_HORIZON] / close.iloc[i] - 1

        ranks = {}
        for n in names:
            f = factor_panels[n].iloc[i]
            keep = act & f.notna() & lnm.notna()
            fv = f[keep]
            out = pd.Series(np.nan, index=close.columns)
            if fv.notna().sum() >= MIN_STOCKS:
                resid = neutralize(winsorize_mad(fv), industry.reindex(fv.index),
                                   lnm[keep])
                out.loc[fv.index] = to_rank(resid).reindex(fv.index)
            ranks[n] = out

        rdf = pd.DataFrame(ranks)
        corr_acc += rdf.corr(method="spearman").to_numpy()
        for n in names:
            r = ranks[n]
            vm = r.notna() & fwd.notna()
            ic_lists[n].append(r[vm].corr(fwd[vm], method="spearman")
                               if vm.sum() >= MIN_STOCKS else np.nan)
        ic_dates.append(date)
        for n in names:
            rank_lists[n].append(ranks[n])

        if (idx_i + 1) % 20 == 0:
            print(f"    ...{idx_i + 1}/{len(rebal)}", flush=True)

    corr = pd.DataFrame(corr_acc / len(rebal), index=names, columns=names)
    rank_panels = {n: pd.DataFrame(rank_lists[n], index=pd.DatetimeIndex(ic_dates))
                   for n in names}
    ic_df = pd.DataFrame(ic_lists, index=pd.DatetimeIndex(ic_dates))

    # ---- 单因子 IC 表 ----
    single_rows = []
    for n in names:
        st = ic_stats(ic_df[n])
        single_rows.append({"因子": n, **st})
    single_ic = pd.DataFrame(single_rows).set_index("因子")

    print("\n[单因子 IC]（中性化 + rank 后，20 日 RankIC）")
    print(single_ic.map(fmt).to_string())
    print("\n[因子相关性矩阵]（中性化 + rank 后，时序平均）")
    print(corr.map(fmt).to_string())

    # ---- 剔除高相关因子 ----
    ic_abs = {n: abs(single_ic.loc[n, "IC均值"]) if pd.notna(single_ic.loc[n, "IC均值"])
              else 0.0 for n in names}
    survivors, dropped = drop_correlated(corr, ic_abs, names)
    print("\n[相关性筛选] 阈值 |r| >", CORR_THRESH)
    for d, a, b, r in dropped:
        print(f"    剔除「{d}」：|r({a},{b})| = {abs(r):.3f} > {CORR_THRESH}，且信号更弱")
    print(f"    保留因子：{survivors}")

    # ---- 第二遍：三种合成 + 分层回测 ----
    print("\n[4] 三种合成回测（含 0.1% 单边成本）...", flush=True)

    # 各方法状态
    nav = {m: [] for m in METHODS}
    ic_comp = {m: [] for m in METHODS}
    excess = {m: [] for m in METHODS}
    turn_hist = {m: [] for m in METHODS}
    bench_rets = []
    prev_hold = {m: None for m in METHODS}
    # 权重历史（只用过去 IC）：running sum / sumsq / count
    sum_ic = {n: 0.0 for n in survivors}
    sumsq_ic = {n: 0.0 for n in survivors}
    cnt_ic = {n: 0 for n in survivors}

    def weights_for(method):
        w = {}
        for n in survivors:
            if method == "等权" or cnt_ic[n] < WARMUP:
                w[n] = 1.0 / len(survivors)
                continue
            mean = sum_ic[n] / cnt_ic[n]
            if method == "IC加权":
                w[n] = max(0.0, mean)
            else:  # ICIR 加权
                var = sumsq_ic[n] / cnt_ic[n] - mean ** 2
                std = np.sqrt(var) if var > 0 else 0.0
                w[n] = mean / std if std > 0 else 0.0
        tot = sum(w.values())
        if tot <= 0:
            return {n: 1.0 / len(survivors) for n in survivors}
        return {n: v / tot for n, v in w.items()}

    for k, i in enumerate(rebal):
        date = close.index[i]
        act = mask.iloc[i]
        fwd = close.iloc[i + MAIN_HORIZON] / close.iloc[i] - 1
        bench = fwd[act & fwd.notna()].mean(skipna=True)
        bench_rets.append(bench)

        for method in METHODS:
            w = weights_for(method)
            score = sum(w[n] * rank_panels[n].iloc[k] for n in survivors)
            valid = score.notna() & act & fwd.notna()
            if valid.sum() < MIN_STOCKS:
                nav[method].append(np.nan)
                ic_comp[method].append(np.nan)
                excess[method].append(np.nan)
                turn_hist[method].append(np.nan)
                prev_hold[method] = None
                continue
            kk = max(1, int(valid.sum() * TOP_FRAC))
            new = score[valid].nlargest(kk).index
            gross = fwd[new].mean(skipna=True)
            if prev_hold[method] is None:
                to = 1.0
            else:
                to = 1.0 - len(new.intersection(prev_hold[method])) / kk
            net = gross - COST * to
            nav[method].append(net)
            turn_hist[method].append(to)
            ic_comp[method].append(score[valid].corr(fwd[valid], method="spearman"))
            excess[method].append(net - bench)
            prev_hold[method] = new

        # 更新权重历史（用本期的单因子 IC，供下一期使用）
        for n in survivors:
            icv = ic_df[n].iloc[k]
            if pd.notna(icv):
                sum_ic[n] += icv
                sumsq_ic[n] += icv ** 2
                cnt_ic[n] += 1

    # ---- 汇总 ----
    nav_s = {m: pd.Series(nav[m], index=pd.DatetimeIndex(ic_dates)) for m in METHODS}
    excess_s = {m: pd.Series(excess[m], index=pd.DatetimeIndex(ic_dates)) for m in METHODS}
    turn_s = {m: pd.Series(turn_hist[m], index=pd.DatetimeIndex(ic_dates)) for m in METHODS}
    ic_s = {m: pd.Series(ic_comp[m], index=pd.DatetimeIndex(ic_dates)) for m in METHODS}
    bench_s = pd.Series(bench_rets, index=pd.DatetimeIndex(ic_dates))

    bench_ann = annualize(bench_s)
    rows = []
    for method in METHODS:
        st = ic_stats(ic_s[method])
        rows.append({
            "方法": method,
            "组合IC均值": st["IC均值"], "组合ICIR": st["ICIR"], "组合IC_t": st["IC_t值"],
            "组合年化(净)": annualize(nav_s[method]),
            "超额年化": annualize(nav_s[method]) - bench_ann,
            "平均换手": turn_s[method].mean(),
            "最大回撤": max_drawdown(nav_s[method]),
            "有效期数": st["期数"],
        })
    comp = pd.DataFrame(rows).set_index("方法")

    print("\n" + "=" * 72)
    print("[核心对比表] 三种合成方法（20 日调仓，含 0.1% 单边成本）")
    print("=" * 72)
    print(comp.map(fmt).to_string())
    print(f"\n  基准：全A等权 年化 {bench_ann:.4f}")

    # ---- 结论 ----
    eq_icir = comp.loc["等权", "组合ICIR"]
    best = comp["组合ICIR"].idxmax()
    print("\n[结论] 复杂方法有没有跑赢等权？")
    for m in ["IC加权", "ICIR加权"]:
        d = comp.loc[m, "组合ICIR"] - eq_icir
        print(f"    {m} 组合ICIR - 等权 = {d:+.4f}")
    print(f"    → 组合 ICIR 最高的是「{best}」；"
          f"年化超额最高的是「{comp['超额年化'].idxmax()}」。")

    # ---- 保存 CSV ----
    print("\n[5] 保存结果 ...", flush=True)
    comp.to_csv(os.path.join(RESULTS, "合成_方法对比表.csv"), encoding="utf-8-sig")
    single_ic.to_csv(os.path.join(RESULTS, "合成_单因子IC.csv"), encoding="utf-8-sig")
    corr.to_csv(os.path.join(RESULTS, "合成_因子相关性.csv"), encoding="utf-8-sig")
    pd.DataFrame({m: ic_s[m] for m in METHODS}).to_csv(
        os.path.join(RESULTS, "合成_组合IC时序.csv"), encoding="utf-8-sig")
    pd.DataFrame({**{m: nav_s[m] for m in METHODS}, "全A等权": bench_s}).to_csv(
        os.path.join(RESULTS, "合成_收益时序.csv"), encoding="utf-8-sig")

    # ---- 绘图 ----
    plot_corr(corr, os.path.join(RESULTS, "合成_1_因子相关性热力图.png"))
    plot_ic_ts(ic_s, os.path.join(RESULTS, "合成_2_组合IC时序.png"))
    plot_nav(nav_s, bench_s, os.path.join(RESULTS, "合成_3_累计净值对比.png"))
    plot_annual_ic(ic_s, os.path.join(RESULTS, "合成_4_分年度组合IC.png"))

    print("已保存 4 张图 + 5 个 CSV 到 results/。")
    print("\n[验收提示] 看《合成_方法对比表.csv》：组合 ICIR / 超额年化 / 换手 / 最大回撤；")
    print("  核心结论回答：复杂加权（IC / ICIR）是否在组合 ICIR 上跑赢等权。")
    plt.show()


# ================= 绘图 =================
def plot_corr(corr, out):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, fontsize=10)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=10)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(corr.iloc[i, j]) < 0.6 else "white", fontsize=9)
    ax.set_title("中性化 + rank 后因子两两秩相关")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_ic_ts(ic_s, out):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for m, c in zip(METHODS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ax.plot(ic_s[m].index, ic_s[m].values, lw=1.0, color=c, label=m)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"三种合成方法组合 RankIC 时序（{MAIN_HORIZON}日）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("组合 RankIC"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_nav(nav_s, bench_s, out):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m, c in zip(METHODS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ax.plot(nav_s[m].index, (1 + nav_s[m]).cumprod().values, lw=1.2, color=c, label=m)
    ax.plot(bench_s.index, (1 + bench_s).cumprod().values, lw=1.0, color="gray",
            ls="--", label="全A等权基准")
    ax.axhline(1, color="gray", lw=0.6, ls=":")
    ax.set_title("三种合成组合累计净值（净，扣 0.1% 单边成本）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("累计净值(1为起点)"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_annual_ic(ic_s, out):
    annual = pd.DataFrame({m: ic_s[m].groupby(ic_s[m].index.year).mean() for m in METHODS})
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(annual)); w = 0.28
    for k, m in enumerate(METHODS):
        ax.bar(x + (k - 1) * w, annual[m].values, w, label=m)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in annual.index], fontsize=8)
    ax.set_title("三种合成方法分年度组合 IC 均值")
    ax.set_ylabel("组合 RankIC"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main()
