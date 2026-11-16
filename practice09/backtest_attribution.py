# -*- coding: utf-8 -*-
"""
练习 09 · 归因（把收益拆开）
================================================================

三步（对应题目）：
  1. Brinson 归因：行业配置贡献 vs 个股选择贡献（+ 交互贡献）
  2. 风格归因：5 个风格代理（市值/估值/动量/波动/换手）做时序回归，看 Alpha 和 R²
  3. 判断：R²>0.8 说明什么

归因对象 = 练习08 的中证500 指增组合（月度调仓、成分内前100等权、行业偏离≤3%）。
本脚本【自包含】：重跑练习08 的调仓逻辑并保存持仓，再做两种归因。

关键口径（沿用项目约定 §五/§七）：
  - 收益用【后复权价】；市值/估值用【不复权价】（§五.1）
  - Brinson 用【毛收益】拆配置/选择/交互（纯投资决策），成本单独一行对账到净收益
  - 风格回归的 Y = 净超额收益（组合-基准，月度）；X = 5 个多空因子月收益
  - 风格因子：全 A 股、前/后 1/4 分位、等权、多空收益差；动量窗口 20d
  - 停牌股（close=NaN）按 0 收益处理（与练习08 一致）

用法（PyCharm 选 Anaconda 解释器）：直接运行本脚本即可。
"""
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ================= 可调参数 =================
TOP_N = 100                 # 持仓数（练习08 同款）
COST = 0.0010               # 单边成本
IND_DEV_MAX = 0.03          # 行业偏离上限
REV_WIN, TURN_WIN = 20, 20  # 反转/换手窗口
MOM_WIN = 20                # 风格因子「动量」窗口（用户偏好：120d → 20d）
QUANTILE = 0.25             # 风格因子分位（用户偏好：1/4）
MAD_N = 3.0
MIN_STOCKS = 30
WARMUP = 12                 # IC 加权预热期（月）
START = "2016-01-01"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice09/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")
RAW_CLOSE = os.path.join(ROOT, "practice03", "data", "raw_close_all.parquet")
IND_DIR = os.path.join(ROOT, "practice04", "data", "indicator")
BAL_DIR = os.path.join(ROOT, "practice02", "data", "financial", "balance")
INDUSTRY = os.path.join(ROOT, "practice03", "data", "industry_map.csv")
BENCHMARK = os.path.join(BASE_DIR, "..", "practice08", "data", "benchmark_close.csv")
CONS_HIST = os.path.join(BASE_DIR, "..", "practice08", "data", "cons_hist.csv")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 数据加载（沿用练习08） =================
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
    files = sorted(os.listdir(IND_DIR))
    cols = ["REPORT_DATE", "NOTICE_DATE", "REPORT_DATE_NAME", "ROEJQ", "EPSJB"]
    parts = []
    for fn in files:
        if not fn.endswith(".parquet"):
            continue
        code = os.path.splitext(fn)[0]
        df = pd.read_parquet(os.path.join(IND_DIR, fn), columns=cols); df["code"] = code
        parts.append(df)
    ind = pd.concat(parts, ignore_index=True)
    ind["REPORT_DATE"] = pd.to_datetime(ind["REPORT_DATE"], errors="coerce")
    ind["NOTICE_DATE"] = pd.to_datetime(ind["NOTICE_DATE"], errors="coerce")
    ind = ind[ind["REPORT_DATE_NAME"].astype(str).str.contains("年报", na=False)]
    return ind.dropna(subset=["NOTICE_DATE"])


def load_shares():
    files = sorted(os.listdir(BAL_DIR))
    cols = ["NOTICE_DATE", "REPORT_DATE_NAME", "SHARE_CAPITAL"]
    parts = []
    for fn in files:
        if not fn.endswith(".parquet"):
            continue
        code = os.path.splitext(fn)[0]
        df = pd.read_parquet(os.path.join(BAL_DIR, fn), columns=cols); df["code"] = code
        parts.append(df)
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


def annualize(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return np.nan
    return (1 + ((1 + rets).prod() - 1)) ** (12 / len(rets)) - 1


def ols_with_tstats(y, X):
    """OLS with intercept. y:(n,), X:(n,k). 返回 (beta, se, t, r2)。"""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    n, k = X1.shape
    dof = n - k
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    r2 = 1.0 - (resid @ resid) / float((y - y.mean()) @ (y - y.mean()))
    return beta, se, t, r2


# ================= 风格因子定义 =================
# 每个风格因子：value 面板 + 多头方向（long_high=True 表示做多「取值高」的一侧）
FACTOR_SPECS = {
    "市值SMB": dict(long_high=False),   # 做多小市值（ln市值低）
    "估值HML": dict(long_high=True),    # 做多高EP（便宜）
    "动量20d": dict(long_high=True),    # 做多高20d收益（赢家）——A股20d实为反转，见结果
    "波动":    dict(long_high=False),   # 做多低波动
    "换手":    dict(long_high=False),   # 做多低换手
}


# ================= 绘图 =================
def plot_brinson(alloc_tot, sel_tot, out):
    inds = sorted(set(alloc_tot) | set(sel_tot))
    rows = []
    for k in inds:
        rows.append({"行业": k, "配置贡献": alloc_tot.get(k, 0.0), "选择贡献": sel_tot.get(k, 0.0)})
    df = pd.DataFrame(rows).set_index("行业")
    df["合计"] = df["配置贡献"] + df["选择贡献"]
    df = df.sort_values("合计")
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(df))
    h = 0.38
    ax.barh(y + h / 2, df["配置贡献"].values, height=h, color="#1f77b4", label="配置贡献")
    ax.barh(y - h / 2, df["选择贡献"].values, height=h, color="#d62728", label="选择贡献")
    ax.set_yticks(y); ax.set_yticklabels(df.index.astype(str))
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_title("Brinson 归因：分行业 配置贡献 vs 选择贡献（累计，毛收益口径）")
    ax.set_xlabel("累计贡献"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_factors(factor_df, out):
    nav = (1 + factor_df.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in nav.columns:
        ax.plot(nav.index, nav.values if False else nav[c].values, lw=1.2, label=c)
    ax.axhline(1, color="gray", lw=0.6, ls=":")
    ax.set_title("5 个风格多空因子累计净值（1 为起点）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("累计净值"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("练习 09 · 归因（Brinson + 风格回归）")
    print("=" * 72)

    # ---- 加载 ----
    print("\n[1] 加载数据 ...", flush=True)
    hfq = load_hfq()
    close, volume = hfq["close"], hfq["volume"]
    raw_close = load_raw(close.index).reindex(columns=close.columns)
    ind = load_indicator(); shares = load_shares()
    ind_map = pd.read_csv(INDUSTRY, dtype={"code": str})
    industry = pd.Series(ind_map["industry"].values, index=ind_map["code"])
    ind_of = industry.to_dict()

    bench = pd.read_csv(BENCHMARK)
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")["close"].sort_index()
    bench = bench.reindex(close.index).ffill()

    cons = pd.read_csv(CONS_HIST, dtype={"code": str})
    cons["date"] = pd.to_datetime(cons["date"])
    cons_by_date = {d.strftime("%Y-%m-%d"): set(g["code"]) for d, g in cons.groupby("date")}

    se = pd.Series(np.arange(len(close)), index=close.index)
    pos = se.groupby(close.index.to_period("M")).last().to_numpy(dtype=int)
    rebal_dates = close.index[pos]
    keep = rebal_dates >= START
    pos, rebal_dates = pos[keep], rebal_dates[keep]
    print(f"    调仓日 {len(rebal_dates)} 个（{rebal_dates[0].date()} ~ {rebal_dates[-1].date()}）", flush=True)

    # ---- 因子面板 ----
    print("[2] 构造因子面板 ...", flush=True)
    shares_panel = build_panel(shares, "SHARE_CAPITAL", "NOTICE_DATE", close.index).reindex(columns=close.columns)
    roe_panel = build_panel(ind, "ROEJQ", "NOTICE_DATE", close.index).reindex(columns=close.columns)
    eps_panel = build_panel(ind, "EPSJB", "NOTICE_DATE", close.index).reindex(columns=close.columns)

    rev = -(close / close.shift(REV_WIN) - 1)          # 20d 反转（组合打分用）
    mom120 = close / close.shift(120) - 1              # 120d 动量（组合打分用）
    turn_neg = -(volume.rolling(TURN_WIN).mean() / shares_panel)   # 负向换手（组合打分用）
    ep = eps_panel / raw_close
    factor_panels = {"反转": rev, "动量": mom120, "换手": turn_neg, "ROE": roe_panel, "EP": ep}
    names = list(factor_panels.keys())

    mktcap = raw_close * shares_panel
    ln_mkt = np.log(mktcap.where(mktcap > 0))

    # 风格因子的原始取值面板（不取负、不中性化，只排序）
    mom20 = close / close.shift(MOM_WIN) - 1.0                    # 20d 过去收益
    turn_raw = volume.rolling(TURN_WIN).mean() / shares_panel     # 换手率（原始）
    vol_panel = close.pct_change().rolling(20).std()              # 20d 日收益波动
    style_vals = {
        "市值SMB": ln_mkt,
        "估值HML": ep,
        "动量20d": mom20,
        "波动":    vol_panel,
        "换手":    turn_raw,
    }
    style_names = list(FACTOR_SPECS.keys())

    # ---- 主循环：重跑练习08 调仓 + 存持仓 + Brinson 累加 ----
    print("[3] 月度调仓回测 + Brinson 归因累加 ...", flush=True)
    sum_ic = {n: 0.0 for n in names}
    cnt_ic = {n: 0 for n in names}
    rp_list, rb_list, dates_list = [], [], []
    holdings_rows = []          # (date, code, weight, side)
    e_gross_list = []           # 毛超额（用于 Brinson 对账）
    cost_list = []
    alloc_tot, sel_tot, inter_tot = {}, {}, {}
    alloc_sum = sel_sum = inter_sum = 0.0
    gross_excess_sum = 0.0
    recon_active_sum = 0.0
    rb_recon_sum = 0.0
    rb_index_sum = 0.0
    prev_holdings = set()

    for t in range(len(pos) - 1):
        i, j = pos[t], pos[t + 1]
        date = close.index[i]
        cset = cons_by_date.get(date.strftime("%Y-%m-%d"), set()) & set(close.columns)
        if len(cset) < MIN_STOCKS:
            continue

        act = close.iloc[i].notna()
        lnm = ln_mkt.iloc[i]
        ranks = {}
        for n in names:
            f = factor_panels[n].iloc[i]
            keep_v = act & f.notna() & lnm.notna()
            fv = f[keep_v]
            out = pd.Series(np.nan, index=close.columns)
            if fv.notna().sum() >= MIN_STOCKS:
                resid = neutralize(winsorize_mad(fv), industry.reindex(fv.index), lnm[keep_v])
                out.loc[fv.index] = to_rank(resid).reindex(fv.index)
            ranks[n] = out

        fwd = close.iloc[j] / close.iloc[i] - 1.0
        for n in names:
            r = ranks[n]; vm = r.notna() & fwd.notna()
            if vm.sum() >= MIN_STOCKS:
                ic = r[vm].corr(fwd[vm], method="spearman")
                if pd.notna(ic):
                    sum_ic[n] += ic; cnt_ic[n] += 1

        w = {n: 1.0 / len(names) if cnt_ic[n] < WARMUP else max(0.0, sum_ic[n] / cnt_ic[n])
             for n in names}
        tot = sum(w.values())
        w = {n: 1.0 / len(names) for n in names} if tot <= 0 else {n: v / tot for n, v in w.items()}
        score = sum(w[n] * ranks[n] for n in names)

        sc = score.reindex(list(cset)).dropna()
        if len(sc) < MIN_STOCKS:
            continue
        top = set(sc.nlargest(TOP_N).index)
        hold_w = {c: 1.0 / TOP_N for c in top}

        mc = mktcap.iloc[i].reindex(list(cset)).dropna()
        b_w = mc / mc.sum()
        bench_ind = {}
        for c, bw in b_w.items():
            k = ind_of.get(c, "其他")
            bench_ind[k] = bench_ind.get(k, 0.0) + bw
        hold_w = enforce_industry(hold_w, ind_of, bench_ind)

        # ---- 收益 ----
        held = list(hold_w.keys())
        px0 = close.iloc[i][held]; px1 = close.iloc[j][held]
        gross_ret = float((px1 / px0 - 1.0).mul(pd.Series(hold_w)).sum(skipna=True))
        turnover = 1.0 - sum(hold_w[c] for c in (prev_holdings & set(held)))
        cost = COST * turnover
        rp = gross_ret - cost
        rb = float(bench.iloc[j] / bench.iloc[i] - 1)

        rp_list.append(rp); rb_list.append(rb); dates_list.append(date)
        e_gross_list.append(gross_ret - rb)
        cost_list.append(cost)
        for c, wt in hold_w.items():
            holdings_rows.append((date, c, wt, "组合"))
        for c, wt in b_w.items():
            holdings_rows.append((date, c, wt, "基准"))
        prev_holdings = set(held)

        # ---- Brinson（毛收益口径，逐期累加）----
        fwd_fill = fwd.fillna(0.0)
        wp_i, wb_i = {}, {}
        for c, wt in hold_w.items():
            k = ind_of.get(c, "其他"); wp_i[k] = wp_i.get(k, 0.0) + wt
        for c, wt in b_w.items():
            k = ind_of.get(c, "其他"); wb_i[k] = wb_i.get(k, 0.0) + wt
        rb_num, rb_den = {}, {}
        for c, wt in b_w.items():
            k = ind_of.get(c, "其他"); r = fwd_fill.get(c, 0.0)
            rb_num[k] = rb_num.get(k, 0.0) + wt * r; rb_den[k] = rb_den.get(k, 0.0) + wt
        r_bi = {k: rb_num[k] / rb_den[k] if rb_den[k] > 0 else 0.0 for k in rb_den}
        rp_num, rp_den = {}, {}
        for c, wt in hold_w.items():
            k = ind_of.get(c, "其他"); r = fwd_fill.get(c, 0.0)
            rp_num[k] = rp_num.get(k, 0.0) + wt * r; rp_den[k] = rp_den.get(k, 0.0) + wt
        r_pi = {k: rp_num[k] / rp_den[k] if rp_den[k] > 0 else 0.0 for k in rp_den}

        # 重构的组合/基准毛收益（与上面行业收益同口径：NaN->0）
        R_p_hat = sum(wt * float(fwd_fill.get(c, 0.0)) for c, wt in hold_w.items())
        R_b_hat = sum(wt * float(fwd_fill.get(c, 0.0)) for c, wt in b_w.items())

        period_alloc = period_sel = period_inter = 0.0
        for k in set(wp_i) | set(wb_i):
            wpi = wp_i.get(k, 0.0); wbi = wb_i.get(k, 0.0)
            rbi = r_bi.get(k, 0.0); rpi = r_pi.get(k, 0.0)
            alloc = (wpi - wbi) * (rbi - R_b_hat)
            sel = wbi * (rpi - rbi)
            inter = (wpi - wbi) * (rpi - rbi)
            alloc_tot[k] = alloc_tot.get(k, 0.0) + alloc
            sel_tot[k] = sel_tot.get(k, 0.0) + sel
            inter_tot[k] = inter_tot.get(k, 0.0) + inter
            period_alloc += alloc; period_sel += sel; period_inter += inter
        alloc_sum += period_alloc; sel_sum += period_sel; inter_sum += period_inter
        recon_active_sum += (R_p_hat - R_b_hat)      # 重构的主动收益（Brinson 对账目标）
        gross_excess_sum += (gross_ret - rb)         # 实际毛超额（vs 中证500指数）
        rb_recon_sum += R_b_hat                      # 重构基准收益累计
        rb_index_sum += rb                           # 实际指数收益累计

        if (t + 1) % 24 == 0:
            print(f"    ...{t + 1}/{len(pos) - 1}  {date.date()}", flush=True)

    idx = pd.DatetimeIndex(dates_list)
    rp = pd.Series(rp_list, index=idx, name="组合")
    rb = pd.Series(rb_list, index=idx, name="基准")
    e_net = rp - rb                                              # 净超额
    e_gross = pd.Series(e_gross_list, index=idx, name="毛超额")
    cost_s = pd.Series(cost_list, index=idx, name="成本")

    # ---- Brinson 汇总 ----
    print("\n[4] Brinson 归因汇总 ...", flush=True)
    check = alloc_sum + sel_sum + inter_sum - recon_active_sum
    print(f"    三项合计 = {alloc_sum:.4%} + {sel_sum:.4%} + {inter_sum:.4%} = "
          f"{alloc_sum + sel_sum + inter_sum:.4%}")
    print(f"    重构主动收益(算术累计) = {recon_active_sum:.4%}   对账误差 = {check:.2e}")
    print(f"    实际毛超额(vs指数, 算术累计) = {gross_excess_sum:.4%}")
    print(f"    基准重构差(指数 - 重构基准) = {rb_index_sum - rb_recon_sum:.4%}  "
          f"(自由流通vs总市值 + 价格指数不含分红)")
    br_rows = []
    for k in sorted(set(alloc_tot) | set(sel_tot) | set(inter_tot)):
        br_rows.append({
            "行业": k,
            "配置贡献": alloc_tot.get(k, 0.0),
            "选择贡献": sel_tot.get(k, 0.0),
            "交互贡献": inter_tot.get(k, 0.0),
            "合计": alloc_tot.get(k, 0.0) + sel_tot.get(k, 0.0) + inter_tot.get(k, 0.0),
        })
    br_df = pd.DataFrame(br_rows).sort_values("合计", ascending=False)
    br_df.loc["总计"] = br_df[["配置贡献", "选择贡献", "交互贡献", "合计"]].sum(numeric_only=True)
    br_df.to_csv(os.path.join(RESULTS, "Brinson归因表.csv"), encoding="utf-8-sig")

    # ---- 构造 5 个风格因子（全 A 股，前/后 1/4 分位，等权）----
    print("[5] 构造 5 个风格因子（全A，1/4 分位）...", flush=True)
    factor_ret = {n: [] for n in style_names}
    factor_dates = []
    for t in range(len(pos) - 1):
        i, j = pos[t], pos[t + 1]
        date = close.index[i]
        fwd_all = close.iloc[j] / close.iloc[i] - 1.0
        factor_dates.append(date)
        for n in style_names:
            v = style_vals[n].iloc[i]
            keep_v = v.notna() & fwd_all.notna()
            if keep_v.sum() < 60:
                factor_ret[n].append(np.nan); continue
            vv = v[keep_v]; rr = fwd_all[keep_v]
            lo, hi = vv.quantile(QUANTILE), vv.quantile(1 - QUANTILE)
            long_high = FACTOR_SPECS[n]["long_high"]
            if long_high:
                long_mask, short_mask = vv >= hi, vv <= lo
            else:
                long_mask, short_mask = vv <= lo, vv >= hi
            factor_ret[n].append(rr[long_mask].mean() - rr[short_mask].mean())
    factor_df = pd.DataFrame(factor_ret, index=pd.DatetimeIndex(factor_dates))
    factor_df.to_csv(os.path.join(RESULTS, "风格因子收益.csv"), encoding="utf-8-sig")

    # ---- 风格回归 ----
    print("[6] 风格回归（Y=净超额，X=5 因子）...", flush=True)
    reg = pd.concat([e_net.rename("超额"), factor_df], axis=1, join="inner").dropna()
    y = reg["超额"].to_numpy(dtype=float)
    X = reg[style_names].to_numpy(dtype=float)
    beta, se, tstats, r2 = ols_with_tstats(y, X)
    alpha_m, betas = beta[0], beta[1:]
    alpha_ann = (1 + alpha_m) ** 12 - 1

    reg_rows = []
    for n, b, tt in zip(style_names, betas, tstats[1:]):
        fmean = float(factor_df[n].mean())
        reg_rows.append({
            "因子": n, "beta": b, "t值": tt,
            "因子月均收益": fmean, "月均贡献": b * fmean,
            "年化贡献(线性)": b * fmean * 12,
        })
    reg_rows.append({
        "因子": "Alpha(截距)", "beta": alpha_m, "t值": tstats[0],
        "因子月均收益": np.nan, "月均贡献": alpha_m, "年化贡献(线性)": alpha_ann,
    })
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(os.path.join(RESULTS, "风格归因表.csv"), encoding="utf-8-sig", index=False)

    print("\n" + "=" * 72)
    print("[风格归因表]")
    print("=" * 72)
    for _, r in reg_df.iterrows():
        print(f"  {r['因子']:<12} beta={r['beta']:+.4f}  t={r['t值']:+.2f}  "
              f"月均贡献={r['月均贡献']:+.4%}")
    print(f"  R^2 = {r2:.4f}")
    print(f"  年化 Alpha = {alpha_ann:+.4%}")
    print(f"  判断：R^2 {'>' if r2 > 0.8 else '<='} 0.8 -> "
          f"{'风格 beta 主导，Alpha 含量低' if r2 > 0.8 else '风格只解释了一部分，需看 Alpha t 值'}")

    # ---- 保存持仓 & 汇总 ----
    print("\n[7] 保存结果 ...", flush=True)
    hd = pd.DataFrame(holdings_rows, columns=["date", "code", "weight", "side"])
    hd.to_csv(os.path.join(RESULTS, "持仓明细.csv"), encoding="utf-8-sig", index=False)
    plot_brinson(alloc_tot, sel_tot, os.path.join(RESULTS, "归因_Brinson.png"))
    plot_factors(factor_df, os.path.join(RESULTS, "归因_风格因子.png"))

    print("\n[验收提示] 看 results/Brinson归因表.csv + 风格归因表.csv + 归因_*.png")
    plt.show()


if __name__ == "__main__":
    main()
