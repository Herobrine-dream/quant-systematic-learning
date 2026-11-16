# -*- coding: utf-8 -*-
"""
练习 06 · 真实回测改造（成本、涨跌停、停牌、T+1）
========================================================

要回答的问题：练习 05 最好的组合（IC 加权）那个"好看"的回测，
加上现实交易摩擦后，年化到底还剩多少？每项摩擦各吃掉多少？

改造项（逐项叠加，每步记录年化掉了多少）：
  V0 基线           = 练习05 IC加权（当日收盘价成交，0.1% 单边成本）
  V1 次日开盘成交    = 成交价从 close[t] 改成 open[t+1]，持有到 open[t+H+1]
  V2 +涨停不买跌停不卖 = 开盘涨停买不进 / 开盘跌停卖不出
  V3 +停牌不可交易    = 成交日无行情（停牌）则买卖都跳过
  V4 +成交量约束      = 单笔买入金额 ≤ 当日成交额 5%（组合规模 1 亿）
  成本扫描           = 在 V4 上 0 → 0.05% → 0.1% → 0.2% → 0.3%

关键口径：
  - 收益用【后复权价】算；涨跌停判断用【不复权价】的前收 + 开盘价
    （涨跌停是交易所按不复权价四舍五入到分，复权价会判错，见 hfq 记忆）。
  - 涨跌停幅度从 code 前缀 + name 推断：688/300/301 → 20%，含 ST → 5%，其余 10%。
  - 停牌股在日线里【缺行】（volume==0 行为 0），pivot 后 open/close 为 NaN 即停牌。

数据：全部复用练习 01~05，零新增爬取。
用法（PyCharm 选 Anaconda 解释器）：python backtest_realistic.py
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ================= 可调参数 =================
REBAL = 20                 # 调仓间隔（交易日）
MAIN_HORIZON = 20          # 持有期（= 调仓周期）
REV_WIN = 20               # 反转回看窗口
MOM_WIN = 120              # 动量回看窗口
TURN_WIN = 20              # 换手率平滑窗口
TOP_FRAC = 0.2             # 买合成分前 20%
MAD_N = 3.0                # 去极值：MAD 3 倍
MIN_STOCKS = 30            # 某期有效股票少于该数则跳过
WARMUP = 12                # IC 加权预热期数
CAPITAL = 1e8              # 组合规模 1 亿（成交量约束）
CAP_FRAC = 0.05            # 单笔 ≤ 当日成交额 5%
COST_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003]   # 成本扫描

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice06/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")
RAW_CLOSE = os.path.join(ROOT, "practice03", "data", "raw_close_all.parquet")
UNIVERSE = os.path.join(ROOT, "practice01", "data_dynamic", "universe.csv")
IND_DIR = os.path.join(ROOT, "practice04", "data", "indicator")
BAL_DIR = os.path.join(ROOT, "practice02", "data", "financial", "balance")
INDUSTRY = os.path.join(ROOT, "practice03", "data", "industry_map.csv")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 数据加载 =================
def load_hfq():
    """后复权 close / open / volume / amount 宽表。"""
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close", "open", "volume", "amount"])
    out = {}
    for col in ["close", "open", "volume", "amount"]:
        out[col] = panel.pivot(index="date", columns="code", values=col).sort_index()
        out[col].columns.name = "code"
        out[col].index.name = "date"
    return out


def load_raw(dates):
    """不复权 close / open 宽表（涨跌停判断 + EP/市值）。"""
    panel = pd.read_parquet(RAW_CLOSE, columns=["date", "code", "close", "open"])
    out = {}
    for col in ["close", "open"]:
        out[col] = panel.pivot(index="date", columns="code", values=col).reindex(dates).sort_index()
        out[col].columns.name = "code"
        out[col].index.name = "date"
    return out


def load_indicator():
    """财务指标长表（年报 only）：ROEJQ / EPSJB。"""
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
    """总股本长表（年报 only，公告日 PIT）。"""
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
    return sh.dropna(subset=["NOTICE_DATE", "SHARE_CAPITAL"])[["code", "NOTICE_DATE", "SHARE_CAPITAL"]]


def build_panel(long_df, value_col, date_col, dates):
    panel = long_df.pivot_table(index=date_col, columns="code", values=value_col, aggfunc="last")
    panel = panel.reindex(panel.index.union(dates)).sort_index().ffill()
    panel = panel.reindex(dates)
    panel.columns.name = "code"
    panel.index.name = "date"
    return panel


def build_mask(close, uni):
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


def build_limit_rate(uni):
    """涨跌停幅度：688/300/301 → 20%，含 ST → 5%，其余 10%。"""
    code = uni["code"]
    rate = pd.Series(0.10, index=code.values)
    rate.loc[code[code.str.startswith(("688", "300", "301"))].values] = 0.20
    rate.loc[code[uni["name"].str.contains("ST", na=False)].values] = 0.05
    return rate


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


def fmt(x):
    return f"{x:.4f}" if pd.notna(x) else "  nan"


# ================= 持仓滚动回测 =================
def run_version(rank_panels, ic_df, survivors, close, open_, raw_close, raw_open, amount,
                mask, limit_rate, rebal, use_open, limit_on, suspend_on, capacity_on, cost):
    """通用持仓滚动回测，返回 (净收益序列, 单边换手序列, 基准收益序列)。"""
    H = MAIN_HORIZON
    n_surv = len(survivors)
    sum_ic = {n: 0.0 for n in survivors}
    sumsq_ic = {n: 0.0 for n in survivors}
    cnt_ic = {n: 0 for n in survivors}

    holdings = set()
    rets, turns, benches, dates = [], [], [], []

    for k, i in enumerate(rebal):
        date = close.index[i]
        act = mask.iloc[i]

        # ---- IC 加权权重（只用过去 IC）----
        w = {}
        for n in survivors:
            if cnt_ic[n] < WARMUP:
                w[n] = 1.0 / n_surv
            else:
                w[n] = max(0.0, sum_ic[n] / cnt_ic[n])
        tot = sum(w.values())
        w = {n: 1.0 / n_surv for n in survivors} if tot <= 0 else {n: v / tot for n, v in w.items()}

        # ---- 目标持仓 ----
        score = sum(w[n] * rank_panels[n].iloc[k] for n in survivors)
        valid = score.notna() & act
        if valid.sum() < MIN_STOCKS:
            for n in survivors:  # 即使跳过，IC 历史仍要更新（因子属性）
                icv = ic_df[n].iloc[k]
                if pd.notna(icv):
                    sum_ic[n] += icv; sumsq_ic[n] += icv ** 2; cnt_ic[n] += 1
            continue
        kk = max(1, int(valid.sum() * TOP_FRAC))
        target = set(score[valid].nlargest(kk).index)

        prev = holdings
        buy = list(target - prev)
        sell = list(prev - target)

        # ---- 成交日 / 成交价 ----
        if use_open:
            ex = i + 1
            buy_px = open_.iloc[ex]
            next_px = open_.iloc[i + H + 1]
            prev_close_raw = raw_close.iloc[i]          # 前收 = t 日不复权收盘
        else:
            ex = i
            buy_px = close.iloc[ex]
            next_px = close.iloc[i + H]
            prev_close_raw = raw_close.iloc[i - 1]      # 前收 = t-1 日不复权收盘

        open_ex = raw_open.iloc[ex]
        amt_ex = amount.iloc[ex]
        hfq_open_ex = open_.iloc[ex]
        lr = limit_rate.reindex(buy_px.index).fillna(0.10)

        def limit_up(c):
            pc = prev_close_raw.get(c)
            if pd.isna(pc) or pc <= 0:
                return False
            o = open_ex.get(c)
            return pd.notna(o) and o >= round(pc * (1 + lr.get(c, 0.10)), 2) - 0.001

        def limit_down(c):
            pc = prev_close_raw.get(c)
            if pd.isna(pc) or pc <= 0:
                return False
            o = open_ex.get(c)
            return pd.notna(o) and o <= round(pc * (1 - lr.get(c, 0.10)), 2) + 0.001

        def suspended(c):
            return pd.isna(hfq_open_ex.get(c))

        # ---- 执行买卖（加约束）----
        buy_exec, sell_exec = [], []
        for c in buy:
            ok = True
            if limit_on and limit_up(c):
                ok = False       # 涨停买不进
            if suspend_on and suspended(c):
                ok = False       # 停牌买不进
            if capacity_on:
                a = amt_ex.get(c)
                if pd.isna(a) or a * CAP_FRAC < CAPITAL / kk:
                    ok = False   # 当日成交额 5% 不够容纳单笔买入
            if ok:
                buy_exec.append(c)
        for c in sell:
            ok = True
            if limit_on and limit_down(c):
                ok = False       # 跌停卖不出
            if suspend_on and suspended(c):
                ok = False       # 停牌卖不出
            if ok:
                sell_exec.append(c)

        holdings = (prev - set(sell_exec)) | set(buy_exec)
        hlist = list(holdings)

        if not hlist:
            rets.append(np.nan); turns.append(np.nan); benches.append(np.nan); dates.append(date)
        else:
            ret = (next_px[hlist] / buy_px[hlist] - 1).mean(skipna=True)
            turn = (len(sell_exec) if prev else len(buy_exec)) / max(1, len(hlist))
            net = ret - cost * turn
            rets.append(net)
            turns.append(turn)
            bench = (next_px[act] / buy_px[act] - 1).mean(skipna=True)
            benches.append(bench)
            dates.append(date)

        # ---- 更新 IC 历史（供下一期权重）----
        for n in survivors:
            icv = ic_df[n].iloc[k]
            if pd.notna(icv):
                sum_ic[n] += icv; sumsq_ic[n] += icv ** 2; cnt_ic[n] += 1

    idx = pd.DatetimeIndex(dates)
    return (pd.Series(rets, index=idx, name="net"),
            pd.Series(turns, index=idx, name="turn"),
            pd.Series(benches, index=idx, name="bench"))


# ================= 绘图 =================
def plot_nav(series, labels, out):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for s, lab, c in zip(series, labels, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "gray"]):
        ax.plot(s.index, (1 + s).cumprod().values, lw=1.2, color=c, label=lab)
    ax.axhline(1, color="gray", lw=0.6, ls=":")
    ax.set_title("改造前后组合累计净值（净，逐项叠加）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("累计净值(1为起点)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=120)


def plot_cost_curve(cost_ann, bench_ann, breakeven, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    costs_pct = [c * 100 for c in cost_ann.index]
    ax.plot(costs_pct, cost_ann.values, marker="o", color="#d62728", lw=1.5, label="组合年化(净)")
    ax.axhline(bench_ann, color="gray", ls="--", lw=1.0, label=f"全A等权基准年化 {bench_ann:.4f}")
    if breakeven is not None:
        be_pct = breakeven * 100
        ax.axvline(be_pct, color="#2ca02c", ls=":", lw=1.5, label=f"盈亏平衡成本 {be_pct:.2f}%")
        ax.set_xlim(0, max(costs_pct[-1], be_pct) * 1.05)
    ax.set_xlabel("单边成本 (%)")
    ax.set_ylabel("年化收益")
    ax.set_title("收益-成本曲线（V4 全约束；虚线为外推）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120)


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("练习 06 · 真实回测改造（IC 加权组合）")
    print("=" * 72)

    # ---- 加载 ----
    print("\n[1] 加载数据 ...", flush=True)
    hfq = load_hfq()
    close, open_, volume, amount = hfq["close"], hfq["open"], hfq["volume"], hfq["amount"]
    raw = load_raw(close.index)
    raw_close, raw_open = raw["close"].reindex(columns=close.columns), raw["open"].reindex(columns=close.columns)
    ind = load_indicator()
    shares = load_shares()
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    mask = build_mask(close, uni)
    limit_rate = build_limit_rate(uni)
    ind_map = pd.read_csv(INDUSTRY, dtype={"code": str})
    industry = pd.Series(ind_map["industry"].values, index=ind_map["code"])
    print(f"    后复权 {close.shape}；不复权 {raw_close.shape}", flush=True)

    # ---- 因子面板 ----
    print("[2] 构造 5 个因子面板 ...", flush=True)
    shares_panel = build_panel(shares, "SHARE_CAPITAL", "NOTICE_DATE", close.index).reindex(columns=close.columns)
    roe_panel = build_panel(ind, "ROEJQ", "NOTICE_DATE", close.index).reindex(columns=close.columns)
    eps_panel = build_panel(ind, "EPSJB", "NOTICE_DATE", close.index).reindex(columns=close.columns)

    rev = -(close / close.shift(REV_WIN) - 1)
    mom = close / close.shift(MOM_WIN) - 1
    turn = -(volume.rolling(TURN_WIN).mean() / shares_panel)
    ep = eps_panel / raw_close
    factor_panels = {"反转": rev, "动量": mom, "换手": turn, "ROE": roe_panel, "EP": ep}
    names = list(factor_panels.keys())
    del volume  # 释放内存（换手因子已算完）

    mktcap = raw_close * shares_panel
    ln_mkt = np.log(mktcap.where(mktcap > 0))

    rebal = list(range(MOM_WIN, len(close) - MAIN_HORIZON - 1, REBAL))
    print(f"    调仓期数：{len(rebal)}", flush=True)

    # ---- 第一遍：中性化 rank + 单因子 IC ----
    print("[3] 中性化(行业+ln市值) + rank + 单因子 IC ...", flush=True)
    rank_lists = {n: [] for n in names}
    ic_lists = {n: [] for n in names}
    ic_dates = []
    for idx_i, i in enumerate(rebal):
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
                resid = neutralize(winsorize_mad(fv), industry.reindex(fv.index), lnm[keep])
                out.loc[fv.index] = to_rank(resid).reindex(fv.index)
            ranks[n] = out
        for n in names:
            r = ranks[n]
            vm = r.notna() & fwd.notna()
            ic_lists[n].append(r[vm].corr(fwd[vm], method="spearman")
                               if vm.sum() >= MIN_STOCKS else np.nan)
        ic_dates.append(close.index[i])
        for n in names:
            rank_lists[n].append(ranks[n])
        if (idx_i + 1) % 30 == 0:
            print(f"    ...{idx_i + 1}/{len(rebal)}", flush=True)

    rank_panels = {n: pd.DataFrame(rank_lists[n], index=pd.DatetimeIndex(ic_dates)) for n in names}
    ic_df = pd.DataFrame(ic_lists, index=pd.DatetimeIndex(ic_dates))

    # 练习05结论：IC 加权最佳；动量(负IC)/ROE(≈0) 会被截为 0，此处直接用全部 5 因子交给 IC 加权
    survivors = names

    # ---- 第二遍：逐版本回测 ----
    print("[4] 逐版本持仓滚动回测 ...", flush=True)
    versions = [
        ("V0 基线(收盘成交)", dict(use_open=False, limit_on=False, suspend_on=False, capacity_on=False, cost=0.001)),
        ("V1 次日开盘成交", dict(use_open=True, limit_on=False, suspend_on=False, capacity_on=False, cost=0.001)),
        ("V2 +涨停不买跌停不卖", dict(use_open=True, limit_on=True, suspend_on=False, capacity_on=False, cost=0.001)),
        ("V3 +停牌不可交易", dict(use_open=True, limit_on=True, suspend_on=True, capacity_on=False, cost=0.001)),
        ("V4 +成交量约束(1亿)", dict(use_open=True, limit_on=True, suspend_on=True, capacity_on=True, cost=0.001)),
    ]
    results = {}
    for label, kw in versions:
        net, turn_s, bench = run_version(rank_panels, ic_df, survivors,
                                         close, open_, raw_close, raw_open, amount,
                                         mask, limit_rate, rebal, **kw)
        results[label] = (net, turn_s, bench)
        print(f"    {label}: 年化(净) {annualize(net):.4f}, 平均换手 {turn_s.mean():.3f}",
              flush=True)

    # ---- 汇总对比表 ----
    rows = []
    prev_ann = None
    for label, _ in versions:
        net, turn_s, bench = results[label]
        ann = annualize(net)
        bench_ann = annualize(bench)
        rows.append({
            "版本": label,
            "年化(净)": ann,
            "超额年化": ann - bench_ann,
            "平均换手": turn_s.mean(),
            "最大回撤": max_drawdown(net),
            "相对上一步年化变化": (ann - prev_ann) if prev_ann is not None else np.nan,
        })
        prev_ann = ann
    comp = pd.DataFrame(rows).set_index("版本")

    print("\n" + "=" * 72)
    print("[改造前后对比表] 每一步让年化掉了多少")
    print("=" * 72)
    print(comp.map(fmt).to_string())

    # ---- 成本扫描 + 盈亏平衡 ----
    print("\n[5] 成本扫描（V4 全约束）...", flush=True)
    kw4 = dict(use_open=True, limit_on=True, suspend_on=True, capacity_on=True)
    cost_ann = {}
    cost_bench = None
    for c in COST_GRID:
        net, turn_s, bench = run_version(rank_panels, ic_df, survivors,
                                         close, open_, raw_close, raw_open, amount,
                                         mask, limit_rate, rebal, cost=c, **kw4)
        cost_ann[c] = annualize(net)
        if cost_bench is None:
            cost_bench = annualize(bench)
        print(f"    成本 {c*100:.2f}% -> 年化(净) {cost_ann[c]:.4f}", flush=True)

    cost_s = pd.Series(cost_ann)
    # 盈亏平衡：先线性插值找交点；扫描范围内无交点则用最后两点线性外推
    breakeven = None
    ann_arr = cost_s.values
    cost_arr = cost_s.index.values
    for a in range(len(cost_arr) - 1):
        y0, y1 = ann_arr[a], ann_arr[a + 1]
        if (y0 - cost_bench) * (y1 - cost_bench) <= 0 and y0 != y1:
            x0, x1 = cost_arr[a], cost_arr[a + 1]
            breakeven = x0 + (cost_bench - y0) * (x1 - x0) / (y1 - y0)
            break
    if breakeven is None and len(cost_arr) >= 2:
        y0, y1 = ann_arr[-2], ann_arr[-1]
        x0, x1 = cost_arr[-2], cost_arr[-1]
        if y0 != y1:
            slope = (y1 - y0) / (x1 - x0)
            breakeven = x1 + (cost_bench - y1) / slope
    if breakeven is not None and breakeven > cost_arr[-1]:
        breakeven_note = f"{breakeven*100:.2f}%（外推，超出扫描上限）"
    elif breakeven is not None:
        breakeven_note = f"{breakeven*100:.2f}%"
    else:
        breakeven_note = "无（未跑赢基准）"

    print("\n" + "=" * 72)
    print("[盈亏平衡成本]")
    print("=" * 72)
    print(f"    全A等权基准年化 = {cost_bench:.4f}")
    print(f"    盈亏平衡成本 = {breakeven_note}")
    print(f"    （成本高过这个值，策略超额收益归零 —— 这就是策略的安全边际）")

    # ---- 保存 ----
    print("\n[6] 保存结果 ...", flush=True)
    comp.to_csv(os.path.join(RESULTS, "改造前后对比表.csv"), encoding="utf-8-sig")
    pd.DataFrame({"成本": [f"{c*100:.2f}%" for c in cost_s.index],
                  "年化(净)": cost_s.values,
                  "基准年化": cost_bench}).to_csv(
        os.path.join(RESULTS, "成本扫描.csv"), encoding="utf-8-sig", index=False)

    nav_series = [results[l][0] for l, _ in versions]
    plot_nav(nav_series, [l for l, _ in versions], os.path.join(RESULTS, "改造_1_净值对比.png"))
    plot_cost_curve(cost_s, cost_bench, breakeven, os.path.join(RESULTS, "改造_2_收益成本曲线.png"))

    print("已保存 2 张图 + 2 个 CSV 到 results/。")
    print("\n[验收提示] 看《改造前后对比表.csv》：每一步让年化掉了多少；")
    print("  再看《成本扫描.csv》+ 收益-成本曲线：盈亏平衡成本 = 策略安全边际。")
    plt.show()


if __name__ == "__main__":
    main()
