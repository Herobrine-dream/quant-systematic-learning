# -*- coding: utf-8 -*-
"""
练习 08 · 中证500 指数增强（月度调仓）
================================================================

流程（对应题目 5 步）：
  1. 股票池 = 中证500【当期】成分（data/cons_hist.csv，PIT，非当前成分）
  2. 合成因子打分 = 练习05 的 5 因子（反转/动量/换手/ROE/EP）行业+市值中性 rank，
     再用滚动过去 IC 加权（PIT，练习06 同款逻辑）
  3. 组合 = 成分内按打分取前 100 只等权；行业偏离 ≤3% 硬约束（贪心迭代）；
     市值中性由因子层面中性化实现、组合层面验证暴露
  4. 月度调仓，回测 2016 至今，基准 = 中证500；单边成本 0.1%
  5. 指标：年化超额 / 信息比率 IR / 跟踪误差 TE / 超额最大回撤 / 月度超额胜率

关键口径（沿用项目约定）：
  - 收益用【后复权价】；市值用【不复权价 × 总股本】（§五.1）
  - 基准权重用 总市值 代理（中证500 约等于自由流通市值加权，此处近似）
  - 收益口径：月末收盘 -> 下月末收盘（月度 close-to-close）

用法（PyCharm 选 Anaconda 解释器）：先跑 scripts/01_crawl_index.py，再跑本脚本
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ================= 可调参数 =================
TOP_N = 100                 # 持仓数
COST = 0.0010               # 单边成本
IND_DEV_MAX = 0.03          # 行业偏离上限（3%）
REV_WIN, MOM_WIN, TURN_WIN = 20, 120, 20
MAD_N = 3.0
MIN_STOCKS = 30
WARMUP = 12                 # IC 加权预热期（月）
START = "2016-01-01"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice08/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")
RAW_CLOSE = os.path.join(ROOT, "practice03", "data", "raw_close_all.parquet")
UNIVERSE = os.path.join(ROOT, "practice01", "data_dynamic", "universe.csv")
IND_DIR = os.path.join(ROOT, "practice04", "data", "indicator")
BAL_DIR = os.path.join(ROOT, "practice02", "data", "financial", "balance")
INDUSTRY = os.path.join(ROOT, "practice03", "data", "industry_map.csv")
BENCHMARK = os.path.join(BASE_DIR, "data", "benchmark_close.csv")
CONS_HIST = os.path.join(BASE_DIR, "data", "cons_hist.csv")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 数据加载（沿用练习06） =================
def load_hfq():
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close", "open", "volume", "amount"])
    out = {}
    for col in ["close", "open", "volume", "amount"]:
        out[col] = panel.pivot(index="date", columns="code", values=col).sort_index()
        out[col].columns.name = "code"; out[col].index.name = "date"
    return out


def load_raw(dates):
    panel = pd.read_parquet(RAW_CLOSE, columns=["date", "code", "close"])
    close = panel.pivot(index="date", columns="code", values="close").sort_index().reindex(dates)
    close.columns.name = "code"; close.index.name = "date"
    return close


def load_indicator():
    files = sorted(glob.glob(os.path.join(IND_DIR, "*.parquet")))
    cols = ["REPORT_DATE", "NOTICE_DATE", "REPORT_DATE_NAME", "ROEJQ", "EPSJB"]
    parts = []
    for f in files:
        code = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_parquet(f, columns=cols); df["code"] = code; parts.append(df)
    ind = pd.concat(parts, ignore_index=True)
    ind["REPORT_DATE"] = pd.to_datetime(ind["REPORT_DATE"], errors="coerce")
    ind["NOTICE_DATE"] = pd.to_datetime(ind["NOTICE_DATE"], errors="coerce")
    ind = ind[ind["REPORT_DATE_NAME"].astype(str).str.contains("年报", na=False)]
    return ind.dropna(subset=["NOTICE_DATE"])


def load_shares():
    files = sorted(glob.glob(os.path.join(BAL_DIR, "*.parquet")))
    cols = ["NOTICE_DATE", "REPORT_DATE_NAME", "SHARE_CAPITAL"]
    parts = []
    for f in files:
        code = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_parquet(f, columns=cols); df["code"] = code; parts.append(df)
    sh = pd.concat(parts, ignore_index=True)
    sh["NOTICE_DATE"] = pd.to_datetime(sh["NOTICE_DATE"], errors="coerce")
    sh = sh[sh["REPORT_DATE_NAME"].astype(str).str.contains("年报", na=False)]
    return sh.dropna(subset=["NOTICE_DATE", "SHARE_CAPITAL"])[["code", "NOTICE_DATE", "SHARE_CAPITAL"]]


def build_panel(long_df, value_col, date_col, dates):
    panel = long_df.pivot_table(index=date_col, columns="code", values=value_col, aggfunc="last")
    panel = panel.reindex(panel.index.union(dates)).sort_index().ffill()
    panel = panel.reindex(dates)
    panel.columns.name = "code"; panel.index.name = "date"
    return panel


def winsorize_mad(s, n=MAD_N):
    med = s.median(); mad = (s - med).abs().median()
    if mad == 0:
        return s
    k = n * 1.4826 * mad
    return s.clip(lower=med - k, upper=med + k)


def to_rank(s):
    return s.rank(pct=True)


def neutralize(y, industry, ln_mkt):
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


# ================= 指标 =================
def perf_summary(e):
    """e = 月度超额收益序列。返回 (年化超额, IR, TE, 超额最大回撤, 月度胜率)。"""
    e = e.dropna()
    if len(e) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    T = len(e)
    ann = (1 + ((1 + e).prod() - 1)) ** (12 / T) - 1
    te = e.std(ddof=1) * np.sqrt(12)
    ir = ann / te if te > 0 else np.nan
    nav = (1 + e).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    win = (e > 0).mean()
    return ann, ir, te, mdd, win


def annualize(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return np.nan
    return (1 + ((1 + rets).prod() - 1)) ** (12 / len(rets)) - 1


# ================= 行业偏离约束（贪心迭代） =================
def enforce_industry(w, ind_of, bench_ind, ind_dev_max=IND_DEV_MAX, iters=8):
    w = dict(w)
    for _ in range(iters):
        port_ind = {}
        for c, wt in w.items():
            k = ind_of.get(c, "其他")
            port_ind[k] = port_ind.get(k, 0.0) + wt
        over = {k: port_ind[k] - bench_ind.get(k, 0.0)
                for k in port_ind if port_ind[k] - bench_ind.get(k, 0.0) > ind_dev_max}
        if not over:
            break
        for k in over:
            target = bench_ind.get(k, 0.0) + ind_dev_max
            scale = target / port_ind[k] if port_ind[k] > 0 else 0.0
            for c, wt in list(w.items()):
                if ind_of.get(c, "其他") == k:
                    w[c] = wt * scale
        freed = 1.0 - sum(w.values())
        if freed > 1e-9:
            non_over = [c for c in w if ind_of.get(c, "其他") not in over]
            if non_over:
                add = freed / len(non_over)
                for c in non_over:
                    w[c] += add
    return w


# ================= 绘图 =================
def plot_excess_nav(e, out):
    nav = (1 + e).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(nav.index, nav.values, color="#d62728", lw=1.4, label="超额净值")
    ax.axhline(1, color="gray", lw=0.6, ls=":")
    ax.set_title("中证500指增 超额净值曲线（组合 / 基准）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("超额净值(1为起点)"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_exposures(dev, size_gap, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    dev = dev.sort_values()
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in dev.values]
    ax1.barh(dev.index.astype(str), dev.values, color=colors)
    ax1.axvline(IND_DEV_MAX, color="gray", ls="--", lw=1, label="+3%")
    ax1.axvline(-IND_DEV_MAX, color="gray", ls="--", lw=1)
    ax1.set_title("行业平均偏离（组合 − 基准，虚线 ±3%）"); ax1.legend(fontsize=8)
    ax2.plot(size_gap.index, size_gap.values, color="#2ca02c", lw=1.2)
    ax2.axhline(0, color="gray", lw=0.6, ls=":")
    ax2.set_title("市值暴露差（组合 − 基准，ln市值加权，≈0 即中性）")
    ax2.set_xlabel("调仓日")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("练习 08 · 中证500 指数增强（月度调仓，2016 至今）")
    print("=" * 72)

    # ---- 加载 ----
    print("\n[1] 加载数据 ...", flush=True)
    hfq = load_hfq()
    close, open_, volume, amount = hfq["close"], hfq["open"], hfq["volume"], hfq["amount"]
    raw_close = load_raw(close.index).reindex(columns=close.columns)
    ind = load_indicator(); shares = load_shares()
    ind_map = pd.read_csv(INDUSTRY, dtype={"code": str})
    industry = pd.Series(ind_map["industry"].values, index=ind_map["code"])
    ind_of = industry.to_dict()

    bench = pd.read_csv(BENCHMARK)
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")["close"].sort_index()
    bench = bench.reindex(close.index).ffill()          # 对齐 A 股日历

    cons = pd.read_csv(CONS_HIST, dtype={"code": str})
    cons["date"] = pd.to_datetime(cons["date"])
    cons_by_date = {d.strftime("%Y-%m-%d"): set(g["code"]) for d, g in cons.groupby("date")}

    # 调仓日 = 月末交易日（用 close 自己的日历，位置直接可得）
    se = pd.Series(np.arange(len(close)), index=close.index)
    pos = se.groupby(close.index.to_period("M")).last().to_numpy(dtype=int)
    rebal_dates = close.index[pos]
    keep = rebal_dates >= START
    pos, rebal_dates = pos[keep], rebal_dates[keep]
    print(f"    调仓日 {len(rebal_dates)} 个（{rebal_dates[0].date()} ~ {rebal_dates[-1].date()}）", flush=True)

    # ---- 因子面板 ----
    print("[2] 构造 5 因子面板 ...", flush=True)
    shares_panel = build_panel(shares, "SHARE_CAPITAL", "NOTICE_DATE", close.index).reindex(columns=close.columns)
    roe_panel = build_panel(ind, "ROEJQ", "NOTICE_DATE", close.index).reindex(columns=close.columns)
    eps_panel = build_panel(ind, "EPSJB", "NOTICE_DATE", close.index).reindex(columns=close.columns)

    rev = -(close / close.shift(REV_WIN) - 1)
    mom = close / close.shift(MOM_WIN) - 1
    turn = -(volume.rolling(TURN_WIN).mean() / shares_panel)
    ep = eps_panel / raw_close
    factor_panels = {"反转": rev, "动量": mom, "换手": turn, "ROE": roe_panel, "EP": ep}
    names = list(factor_panels.keys())

    mktcap = raw_close * shares_panel
    ln_mkt = np.log(mktcap.where(mktcap > 0))

    # ---- 主循环 ----
    print("[3] 月度调仓回测 ...", flush=True)
    sum_ic = {n: 0.0 for n in names}
    cnt_ic = {n: 0 for n in names}
    rp_list, rb_list, dates_list = [], [], []
    size_gap_list, port_ind_rows, bench_ind_rows = [], [], []
    prev_holdings = set()

    for t in range(len(pos) - 1):
        i, j = pos[t], pos[t + 1]
        date = close.index[i]
        cset = cons_by_date.get(date.strftime("%Y-%m-%d"), set()) & set(close.columns)
        if len(cset) < MIN_STOCKS:
            continue

        # --- 各因子中性化 + rank（全市场，用于 IC 与打分）---
        act = close.iloc[i].notna()
        lnm = ln_mkt.iloc[i]
        ranks = {}
        for n in names:
            f = factor_panels[n].iloc[i]
            keep = act & f.notna() & lnm.notna()
            fv = f[keep]
            out = pd.Series(np.nan, index=close.columns)
            if fv.notna().sum() >= MIN_STOCKS:
                resid = neutralize(winsorize_mad(fv), industry.reindex(fv.index), lnm[keep])
                out.loc[fv.index] = to_rank(resid).reindex(fv.index)
            ranks[n] = out

        # --- 前瞻月度收益 + 单因子 IC（全市场口径，滚动更新）---
        fwd = close.iloc[j] / close.iloc[i] - 1.0
        for n in names:
            r = ranks[n]; vm = r.notna() & fwd.notna()
            if vm.sum() >= MIN_STOCKS:
                ic = r[vm].corr(fwd[vm], method="spearman")
                if pd.notna(ic):
                    sum_ic[n] += ic; cnt_ic[n] += 1

        # --- IC 加权打分（用滚动过去 IC，PIT）---
        w = {n: 1.0 / len(names) if cnt_ic[n] < WARMUP else max(0.0, sum_ic[n] / cnt_ic[n])
             for n in names}
        tot = sum(w.values())
        w = {n: 1.0 / len(names) for n in names} if tot <= 0 else {n: v / tot for n, v in w.items()}
        score = sum(w[n] * ranks[n] for n in names)

        # --- 选股：成分内按打分取前 100 ---
        sc = score.reindex(list(cset)).dropna()
        if len(sc) < MIN_STOCKS:
            continue
        top = set(sc.nlargest(TOP_N).index)
        hold_w = {c: 1.0 / TOP_N for c in top}

        # --- 基准权重（总市值代理）+ 行业约束 ---
        mc = mktcap.iloc[i].reindex(list(cset)).dropna()
        b_w = mc / mc.sum()
        bench_ind = {}
        for c, bw in b_w.items():
            k = ind_of.get(c, "其他")
            bench_ind[k] = bench_ind.get(k, 0.0) + bw
        hold_w = enforce_industry(hold_w, ind_of, bench_ind)

        # --- 收益（月度 close-to-close，扣成本）---
        held = list(hold_w.keys())
        px0 = close.iloc[i][held]; px1 = close.iloc[j][held]
        ret = float((px1 / px0 - 1.0).mul(pd.Series(hold_w)).sum(skipna=True))
        turnover = 1.0 - sum(hold_w[c] for c in (prev_holdings & set(held)))
        rp_list.append(ret - COST * turnover)
        rb_list.append(float(bench.iloc[j] / bench.iloc[i] - 1))
        dates_list.append(date)

        # --- 暴露记录 ---
        valid_c = [c for c in held if pd.notna(lnm.get(c))]
        valid_b = [c for c in b_w.index if pd.notna(lnm.get(c))]
        if valid_c and valid_b:
            wp = sum(hold_w[c] for c in valid_c); wb = sum(b_w[c] for c in valid_b)
            size_p = sum((hold_w[c] / wp) * lnm[c] for c in valid_c)
            size_b = sum((b_w[c] / wb) * lnm[c] for c in valid_b)
            size_gap_list.append(size_p - size_b)
        else:
            size_gap_list.append(np.nan)
        port_ind = {}
        for c, wt in hold_w.items():
            k = ind_of.get(c, "其他"); port_ind[k] = port_ind.get(k, 0.0) + wt
        port_ind_rows.append(port_ind)
        bench_ind_rows.append(bench_ind)
        prev_holdings = set(held)

        if (t + 1) % 24 == 0:
            print(f"    ...{t + 1}/{len(pos) - 1}  {date.date()}", flush=True)

    idx = pd.DatetimeIndex(dates_list)
    rp = pd.Series(rp_list, index=idx, name="组合")
    rb = pd.Series(rb_list, index=idx, name="基准")
    e = rp - rb
    port_ind_hist = pd.DataFrame(port_ind_rows, index=idx).fillna(0.0)
    bench_ind_hist = pd.DataFrame(bench_ind_rows, index=idx).fillna(0.0)
    size_gap = pd.Series(size_gap_list, index=idx, name="市值暴露差").dropna()

    # ---- 指标 ----
    ann_ex, ir, te, mdd, win = perf_summary(e)
    print("\n" + "=" * 72)
    print("[指标表]（全部基于超额收益，不是绝对收益）")
    print("=" * 72)
    print(f"  年化超额       = {ann_ex:.4f}")
    print(f"  信息比率 IR    = {ir:.4f}")
    print(f"  跟踪误差 TE    = {te:.4f}")
    print(f"  超额最大回撤   = {mdd:.4f}")
    print(f"  月度超额胜率   = {win:.4f}")
    print(f"  组合年化(参考) = {annualize(rp):.4f}")
    print(f"  基准年化(参考) = {annualize(rb):.4f}")

    # ---- 保存 ----
    print("\n[4] 保存结果 ...", flush=True)
    perf = pd.DataFrame({
        "指标": ["年化超额", "信息比率IR", "跟踪误差TE", "超额最大回撤", "月度超额胜率",
                 "组合年化", "基准年化"],
        "值": [ann_ex, ir, te, mdd, win, annualize(rp), annualize(rb)],
    })
    perf.to_csv(os.path.join(RESULTS, "指标表.csv"), encoding="utf-8-sig", index=False)
    pd.DataFrame({"组合月收益": rp, "基准月收益": rb, "超额月收益": e}).to_csv(
        os.path.join(RESULTS, "月度超额序列.csv"), encoding="utf-8-sig")
    port_ind_hist.to_csv(os.path.join(RESULTS, "行业暴露时序.csv"), encoding="utf-8-sig")
    size_gap.to_csv(os.path.join(RESULTS, "市值暴露时序.csv"), encoding="utf-8-sig")

    dev = port_ind_hist.mean() - bench_ind_hist.mean()      # 行业平均偏离（组合 − 基准）
    plot_excess_nav(e, os.path.join(RESULTS, "超额净值曲线.png"))
    plot_exposures(dev, size_gap, os.path.join(RESULTS, "行业与市值暴露.png"))

    print("\n[验收提示] 看 results/超额净值曲线.png + 指标表.csv + 行业与市值暴露.png")
    print("  标题写的是『超额』，不是『收益』。")
    plt.show()


if __name__ == "__main__":
    main()
