# -*- coding: utf-8 -*-
"""
练习 10 · 可转债双低轮动回测
================================================================

流程（对应题目 5 步）：
  1. 双低值 = 转债价 + 转股溢价率×100（溢价率取百分数）
  2. 筛选：剔除已触发强赎(15/30日≥130%)、发行规模<1亿、上市不满20日、评级低于AA-
  3. 每周调仓，选双低值最低 20 只等权
  4. 回测 2018-2024（正股价数据截止 2024-12-31），单边成本 0.1%
  5. 分年度：每年都是正收益吗

关键口径（本项目约定 + 本练习新增）：
  - 转股价值 = 100 / 转股价 × 正股价（不复权）
  - 转股价用【下修记录重构的阶梯函数】；分红除权的机械调整忽略（小量，见结果分析）
  - 剩余规模用【发行规模】代理（无历史剩余规模接口）
  - 强赎用【已触发 15/30 日 ≥ 转股价×130%】代理"已公告强赎"

用法：先跑 scripts/01_crawl_cb.py，再跑本脚本。
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ================= 可调参数 =================
TOP_N = 20                  # 持仓数
COST = 0.0010               # 单边成本（佣金+冲击）
MIN_SIZE = 1.0              # 发行规模下限（亿元，代理剩余规模）
LIST_MIN_DAYS = 20          # 上市满 20 日
RATING_OK = {"AAA", "AA+", "AA", "AA-"}   # 评级下限 AA-
REDEEM_WIN, REDEEM_CNT, REDEEM_RATIO = 30, 15, 1.30   # 强赎条款 15/30 日 ≥ 130%
START = "2018-01-01"
END = "2024-12-31"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice10/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DATA = os.path.join(BASE_DIR, "data")
DAILY_DIR = os.path.join(DATA, "cb_daily")
ADJ_DIR = os.path.join(DATA, "cb_adj")
UNIVERSE = os.path.join(DATA, "cb_universe_filtered.csv")
RAW_CLOSE = os.path.join(ROOT, "practice03", "data", "raw_close_all.parquet")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 数据加载 =================
def load_universe():
    uni = pd.read_csv(UNIVERSE, dtype={"债券代码": str, "正股代码": str})
    uni["债券代码"] = uni["债券代码"].str.zfill(6)
    uni["正股代码"] = uni["正股代码"].str.zfill(6)
    uni["上市时间"] = pd.to_datetime(uni["上市时间"], errors="coerce")
    uni["发行规模"] = pd.to_numeric(uni["发行规模"], errors="coerce")
    uni["转股价"] = pd.to_numeric(uni["转股价"], errors="coerce")
    return uni


def load_cb_close():
    """读取所有转债日线，返回 date×code 的收盘价面板。"""
    files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.csv")))
    parts = []
    for f in files:
        code = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df["code"] = code
        parts.append(df[["date", "close", "code"]])
    long_ = pd.concat(parts, ignore_index=True)
    long_["date"] = pd.to_datetime(long_["date"])
    panel = long_.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    panel.columns.name = "code"; panel.index.name = "date"
    return panel


def load_raw_close():
    long_ = pd.read_parquet(RAW_CLOSE)
    panel = long_.pivot(index="date", columns="code", values="close").sort_index()
    panel.columns.name = "code"; panel.index.name = "date"
    return panel


def load_adj(code):
    p = os.path.join(ADJ_DIR, f"{code}.csv")
    if not os.path.exists(p) or os.path.getsize(p) < 5:
        return None
    try:
        df = pd.read_csv(p)
    except Exception:  # noqa: BLE001  空文件/坏文件
        return None
    if df is None or len(df) == 0:
        return None
    return df


# ================= 转股价阶梯函数 =================
def build_step(adj_df, current_cp):
    """下修记录 -> 转股价阶梯断点 [(生效日, 转股价), ...]。无下修用当前转股价常数。"""
    if adj_df is None or len(adj_df) == 0:
        return [(pd.Timestamp("2000-01-01"), float(current_cp))]
    adj = adj_df.dropna(subset=["新转股价生效日期"]).sort_values("新转股价生效日期")
    if len(adj) == 0:
        return [(pd.Timestamp("2000-01-01"), float(current_cp))]
    adj = adj.copy()
    adj["新转股价生效日期"] = pd.to_datetime(adj["新转股价生效日期"], errors="coerce")
    adj = adj.dropna(subset=["新转股价生效日期"])
    bp = [(pd.Timestamp("2000-01-01"), float(adj.iloc[0]["下修前转股价"]))]   # 首次下修前
    for _, r in adj.iterrows():
        bp.append((r["新转股价生效日期"], float(r["下修后转股价"])))
    return bp


def step_series(breakpoints, index):
    """把阶梯断点对齐到给定日历，前向填充成时间序列。"""
    s = pd.Series([v for _, v in breakpoints], index=pd.DatetimeIndex([d for d, _ in breakpoints]))
    s = s.reindex(s.index.union(index)).sort_index().ffill().reindex(index)
    return s


# ================= 基准（中证转债指数，可选） =================
def load_benchmark():
    cache = os.path.join(DATA, "cb_index.csv")
    if os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["date"]).set_index("date")["close"]
        return s
    try:
        import akshare as ak
        df = ak.index_zh_a_hist(symbol="000832", period="daily",
                                start_date=START.replace("-", ""), end_date=END.replace("-", ""))
        date_col = "日期" if "日期" in df.columns else df.columns[0]
        close_col = "收盘" if "收盘" in df.columns else df.columns[-2]
        df = df.rename(columns={date_col: "date", close_col: "close"})
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].sort_index()
        s.to_csv(cache, encoding="utf-8-sig")
        return s
    except Exception as e:  # noqa: BLE001
        print(f"    [警告] 中证转债指数获取失败：{e}", flush=True)
        return None


# ================= 指标 =================
def annualize(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return np.nan
    return (1 + ((1 + rets).prod() - 1)) ** (52 / len(rets)) - 1


def max_drawdown(nav):
    return (nav / nav.cummax() - 1).min()


# ================= 绘图 =================
def plot_nav(nav, bench, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(nav.index, nav.values, color="#d62728", lw=1.4, label="双低策略")
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        b = b / b.iloc[0] * nav.iloc[0]
        ax.plot(b.index, b.values, color="#7f7f7f", lw=1.0, label="中证转债指数")
    ax.axhline(nav.iloc[0], color="gray", lw=0.6, ls=":")
    ax.set_title("可转债双低轮动 净值曲线（2018–2024，周频）")
    ax.set_xlabel("日期"); ax.set_ylabel("净值"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("练习 10 · 可转债双低轮动（周频，2018–2024）")
    print("=" * 72)

    print("\n[1] 加载数据 ...", flush=True)
    uni = load_universe()
    cb_close = load_cb_close()
    raw_close = load_raw_close()
    calendar = raw_close.index
    cb_close = cb_close.reindex(calendar)
    cb_close_ff = cb_close.ffill()                      # 用于收益估值（退市按最后价）
    print(f"    转债日线：{cb_close.shape[1]} 只；正股价日历：{len(calendar)} 天", flush=True)

    print("[2] 重构转股价阶梯 + 逐只算双低/强赎触发 ...", flush=True)
    cols, trig_cols = {}, {}
    uni_meta = {}
    for _, r in uni.iterrows():
        code = r["债券代码"]
        stock = r["正股代码"]
        if code not in cb_close.columns or stock not in raw_close.columns:
            continue
        adj = load_adj(code)
        bp = build_step(adj, r["转股价"])
        cp = step_series(bp, calendar)                       # 转股价（阶梯）
        sprice = raw_close[stock].reindex(calendar)          # 正股价（不复权）
        cbpx = cb_close[code].reindex(calendar)
        conv = 100.0 / cp * sprice                           # 转股价值
        dblow = cbpx + (cbpx - conv) / conv * 100.0          # 双低值（溢价率取百分数）
        above = sprice >= REDEEM_RATIO * cp
        trig = above.rolling(REDEEM_WIN, min_periods=1).sum() >= REDEEM_CNT
        cols[code] = dblow
        trig_cols[code] = trig
        uni_meta[code] = {
            "list_date": r["上市时间"], "size": r["发行规模"],
            "rating": str(r["信用评级"]).strip() if pd.notna(r["信用评级"]) else "",
            "stock": stock,
        }

    dblow = pd.DataFrame(cols, index=calendar)
    trig = pd.DataFrame(trig_cols, index=calendar).fillna(False)
    print(f"    有效转债 {len(cols)} 只，双低面板 {dblow.shape}", flush=True)

    # 周频调仓日 = 每周最后一个交易日
    se = pd.Series(np.arange(len(calendar)), index=calendar)
    pos = se.groupby(calendar.to_period("W")).last().to_numpy(dtype=int)
    rebal = calendar[pos]
    rebal = rebal[(rebal >= START) & (rebal <= END)]
    print(f"    调仓日 {len(rebal)} 个（{rebal[0].date()} ~ {rebal[-1].date()}）", flush=True)

    print("[3] 周频回测 ...", flush=True)
    rp_list, dates_list, hold_cnt = [], [], []
    prev_holdings = set()
    for t_i, t in enumerate(rebal[:-1]):
        t_next = rebal[t_i + 1]
        d = dblow.loc[t]
        tg = trig.loc[t]
        valid = d.notna()
        for code in d.index:
            if not valid[code]:
                continue
            m = uni_meta[code]
            if m["list_date"] is None or t < m["list_date"] + pd.Timedelta(days=LIST_MIN_DAYS):
                valid[code] = False
            elif (m["size"] is not None and m["size"] < MIN_SIZE) or (m["size"] is None):
                valid[code] = False
            elif m["rating"] not in RATING_OK:
                valid[code] = False
            elif tg.get(code, False):
                valid[code] = False
        cand = d[valid].dropna()
        if len(cand) < TOP_N:
            continue
        top = cand.nsmallest(TOP_N).index.tolist()
        px0 = cb_close_ff.loc[t, top]
        px1 = cb_close_ff.loc[t_next, top]
        ret = float((px1 / px0 - 1.0).mean())
        top_set = set(top)
        turnover = 1.0 - len(top_set & prev_holdings) / TOP_N   # 真实换手（等权）
        rp_list.append(ret - COST * turnover)
        prev_holdings = top_set
        dates_list.append(t_next)
        hold_cnt.append(len(top))

        if (t_i + 1) % 52 == 0:
            print(f"    ...{t_i + 1}/{len(rebal) - 1}  {t_next.date()}", flush=True)

    rp = pd.Series(rp_list, index=pd.DatetimeIndex(dates_list), name="周收益")
    nav = (1 + rp).cumprod()
    bench = load_benchmark()

    # ---- 指标 ----
    ann = annualize(rp)
    mdd = max_drawdown(nav)
    total = nav.iloc[-1] / nav.iloc[0] - 1
    print("\n" + "=" * 72)
    print("[指标]")
    print("=" * 72)
    print(f"  累计收益     = {total:.4%}")
    print(f"  年化收益     = {ann:.4%}")
    print(f"  最大回撤     = {mdd:.4%}")
    print(f"  平均持仓数   = {np.mean(hold_cnt):.1f}")

    # ---- 分年度 ----
    yearly = {}
    for y, g in rp.groupby(rp.index.year):
        yearly[y] = (1 + g).prod() - 1
    yearly = pd.Series(yearly)
    print("\n[分年度收益]")
    for y, v in yearly.items():
        flag = "盈利" if v > 0 else "亏损"
        print(f"  {y}  {v:+.2%}  {flag}")

    bench_ann = annualize(bench.reindex(rp.index).ffill().pct_change().dropna()) if bench is not None else np.nan
    print(f"\n  中证转债指数年化(参考) = {bench_ann:.4%}" if bench is not None else "\n  (无基准)")

    # ---- 保存 ----
    print("\n[4] 保存结果 ...", flush=True)
    pd.DataFrame({"周收益": rp, "净值": nav}).to_csv(
        os.path.join(RESULTS, "净值序列.csv"), encoding="utf-8-sig")
    pd.DataFrame({"年份": yearly.index, "年度收益": yearly.values}).to_csv(
        os.path.join(RESULTS, "分年度收益.csv"), encoding="utf-8-sig", index=False)
    plot_nav(nav, bench, os.path.join(RESULTS, "净值曲线.png"))

    print("\n[验收提示] 看 results/净值曲线.png + 净值序列.csv + 分年度收益.csv")
    plt.show()


if __name__ == "__main__":
    main()
