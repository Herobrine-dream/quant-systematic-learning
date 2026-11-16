# -*- coding: utf-8 -*-
"""
练习 02 · ROE 因子完整单因子检验（标准化八步流程）
========================================================

要回答的问题：ROE（归母净利 / 归母权益）这个"盈利/质量"因子，在 A 股横截面上
能否预测未来收益？预测力有多强、多久衰减、分层是否单调、分年度是否稳定？

本脚本严格对应教材《第二章 3.2 标准检验流程（八步）》+《第五章 练习 02》：

  第 1 步 确定调仓日序列            —— 每 20 个交易日调仓一次
  第 2 步 当期可交易股票池(PIT)     —— 上市日 <= t < 退市日，且当期有已披露年报
  第 3 步 计算因子(PIT)            —— ROE = 归母净利 / 归母权益，用「公告日」对齐
  第 4 步 去极值(MAD 3倍) + 转 rank —— 教材 2.2「去极值 + 标准化」，练习02选 rank
  第 5 步 未来收益(可交易价格)      —— 后复权价算未来 1/5/10/20/60 日收益率
  第 6 步 逐期 RankIC              —— 每期因子与未来收益的横截面秩相关
  第 7 步 分层回测(10 组等权)       —— 按因子分 10 组看收益是否单调
  第 8 步 输出报告                 —— IC 时序图 / IC 直方图 / 分层净值 / 衰减曲线 / 分年度表

【关键口径说明】
- 财报口径：年报 only（2010-2025），因此 ROE 用"最新已披露年报"的静态值，非 TTM
  （TTM 需要季报数据，属后续练习；这里严格按已爬取的数据做）。
- PIT 对齐：只使用「公告日(NOTICE_DATE) <= 当期交易日」的年报，避免未来函数。
  年报披露普遍滞后报告期 60~113 天，这正是教材反复强调的"用公告日、别用报告期"。
- 未来收益：用后复权(hfq)价算"收益率"是没问题的（复权只改变价格水平、不改变收益率），
  因此本脚本不涉及"估值因子用 hfq 会偏"的问题。
- 股票池：财报只爬了现存股(5221)，因此本练习天然是"仅现存股"口径，
  存在幸存者偏差，属已知局限（退市股财报是第二阶段补充，见项目说明）。

用法（在 PyCharm 里，右下角选 Anaconda 解释器）：
    python backtest_roe.py

输出：控制台打印 IC 汇总 / 衰减 / 分层 / 分年度表；并在 practice02/results/ 下保存
     5 张 PNG 图 + 4 张 CSV 表（utf-8-sig，可用 Excel 打开）。
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 可调参数 =================
REBAL = 20                 # 调仓间隔（交易日，教材练习02为20日）
N_GROUPS = 10              # 分层组数（教材为10组）
MAD_N = 3.0                # 去极值：中位数 ± MAD_N × 1.4826 × MAD（教材"MAD 3倍"）
HORIZONS = [1, 5, 10, 20, 60]   # IC 衰减要看的未来收益窗口（交易日）
MAIN_HORIZON = 20          # 主回测窗口 = 调仓周期（分层回测用这个持有期收益）
MIN_STOCKS = 30            # 某期有效股票少于这个数就跳过该期（避免噪声）

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice02/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")   # 合并日线(hfq)
UNIVERSE = os.path.join(ROOT, "practice01", "data_dynamic", "universe.csv")  # 动态宇宙表
FIN_DIR = os.path.join(BASE_DIR, "data", "financial")          # 财报三表目录
RESULTS = os.path.join(BASE_DIR, "results")                    # 图表输出目录

# 中文字体（Windows 下 matplotlib 显示中文用）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False                     # 负号正常显示


# ================= 第 1 步：数据准备（第1/2/3步的输入）=================
def load_close():
    """读取合并日线，拼成收盘价宽表。

    返回 DataFrame：index=交易日(date)，columns=股票代码(code)，值=后复权收盘价。
    某只股票上市前/退市后的格子是 NaN —— 这天然就是"点-in-time"的价格数据。
    这里只取 close 一列；未来收益统一用后复权价算收益率（正确，见文件头说明）。
    """
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close"])
    close = panel.pivot(index="date", columns="code", values="close")
    close = close.sort_index()
    close.columns.name = "code"
    close.index.name = "date"
    return close


def load_roe_long():
    """读取全部利润表 + 资产负债表，算出每份年报的 ROE（长表）。

    返回 DataFrame 列：code, REPORT_DATE, NOTICE_DATE, ROE。
    - 利润表取 PARENT_NETPROFIT（归母净利润，单位元）
    - 资产负债表取 TOTAL_PARENT_EQUITY（归属母公司股东权益，单位元）
    - ROE = 归母净利 / 归母权益；归母权益 <= 0（资不抵债）时 ROE 无意义，置 NaN
    """
    import glob

    profit_dir = os.path.join(FIN_DIR, "profit")
    balance_dir = os.path.join(FIN_DIR, "balance")
    files = sorted(glob.glob(os.path.join(profit_dir, "*.parquet")))

    parts = []
    for idx, pf in enumerate(files, 1):
        code = os.path.splitext(os.path.basename(pf))[0]       # 6 位代码
        bf = os.path.join(balance_dir, f"{code}.parquet")
        if not os.path.exists(bf):                             # 缺资产负债表就跳过
            continue

        # 只读需要的列，加快速度
        p = pd.read_parquet(pf, columns=["REPORT_DATE", "NOTICE_DATE", "PARENT_NETPROFIT"])
        b = pd.read_parquet(bf, columns=["REPORT_DATE", "TOTAL_PARENT_EQUITY"])
        p["code"] = code
        b["code"] = code
        # 先把报告期统一成日期(datetime)，避免两张表字符串格式不一致导致 merge 丢行
        p["REPORT_DATE"] = pd.to_datetime(p["REPORT_DATE"])
        b["REPORT_DATE"] = pd.to_datetime(b["REPORT_DATE"])

        # 以 (code, 报告期) 合并两张表；报告期唯一标识一份年报
        m = p.merge(b, on=["code", "REPORT_DATE"], how="inner")
        parts.append(m)

        if idx % 1000 == 0 or idx == len(files):
            print(f"   已读财报 {idx}/{len(files)} 只 ...", flush=True)

    roe = pd.concat(parts, ignore_index=True)
    roe["REPORT_DATE"] = pd.to_datetime(roe["REPORT_DATE"], errors="coerce")
    roe["NOTICE_DATE"] = pd.to_datetime(roe["NOTICE_DATE"], errors="coerce")

    # ROE = 归母净利 / 归母权益；权益非正时 ROE 无意义（置 NaN）
    eq = roe["TOTAL_PARENT_EQUITY"]
    roe["ROE"] = roe["PARENT_NETPROFIT"] / eq.where(eq > 0)

    # 公告日缺失的年报无法做 PIT 对齐，直接丢弃
    roe = roe.dropna(subset=["NOTICE_DATE", "ROE"])
    return roe[["code", "REPORT_DATE", "NOTICE_DATE", "ROE"]]


def build_roe_panel(roe, dates):
    """把「每股的年报 ROE」前向填充成 (交易日 × 股票) 的因子宽表 —— 这就是 PIT。

    原理：因子在交易日 t 的值，应等于「公告日 <= t 的最新一份年报」的 ROE。
    做法：以公告日为行索引做 pivot，再 reindex 到交易日历并 ffill(向前填充)，
          这样每个交易日自动取到"当时已披露"的最新 ROE，绝不用未来披露的数据。
    """
    panel = roe.pivot_table(index="NOTICE_DATE", columns="code",
                            values="ROE", aggfunc="last")
    # 并到交易日历上：披露日可能在非交易日(周末)，也要能对齐
    panel = panel.reindex(panel.index.union(dates)).sort_index().ffill()
    panel = panel.reindex(dates)                               # 只保留交易日
    panel.columns.name = "code"
    panel.index.name = "date"
    return panel


def build_mask(close, uni):
    """构造动态可交易掩码（与 close 同形状的布尔宽表）。

    第 2 步「当期可交易股票池」：交易日 t 满足  上市日 <= t < 退市日。
    - 退市股只在其退市前可用，新股只在其上市后可用（无前视、无幸存者偏差）。
    - 注：因财报只覆盖现存股，退市股在本练习里会被"无 ROE"自然剔除，
      这个掩码主要用来挡住"上市前"和"退市后"的区间，保证口径严谨。
    """
    uni = uni.copy()
    uni["list_date"] = pd.to_datetime(uni["list_date"], errors="coerce")
    uni["delist_date"] = pd.to_datetime(uni["delist_date"], errors="coerce")

    codes = close.columns
    meta = uni.set_index("code").reindex(codes)
    ld = meta["list_date"].to_numpy()
    dd = meta["delist_date"].to_numpy().copy()
    dd[np.isnat(dd)] = np.datetime64("2099-12-31", "ns")       # 未退市 -> 极远未来

    dates = close.index.to_numpy()
    after_listing = dates[:, None] >= ld[None, :]              # 是否已上市
    before_delisting = dates[:, None] < dd[None, :]            # 是否还没退市
    return pd.DataFrame(after_listing & before_delisting,
                        index=close.index, columns=close.columns)


# ================= 第 4 步：去极值 + 转 rank =================
def winsorize_mad(s, n=MAD_N):
    """MAD 去极值（教材 2.2「去极值」）：把超出 中位数 ± n×1.4826×MAD 的值截断。

    MAD = 样本相对中位数的绝对偏差的中位数，是比标准差更抗极端值的离散度量；
    1.4826 是使 MAD 与正态标准差可比的常数。MAD=0（如全期同值）时不做处理。
    """
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    k = n * 1.4826 * mad
    return s.clip(lower=med - k, upper=med + k)


def to_rank(s):
    """转 rank：把因子值变成横截面百分位排名 [0,1]（教材 2.2「标准化/转 rank」）。

    rank 只看排序、不看绝对量纲，天然对分布不敏感，是更稳健的"标准化"。
    """
    return s.rank(pct=True)


# ================= 第 5 步：未来收益 =================
def forward_return(close, i, horizon):
    """未来 horizon 个交易日的持有收益：close[i+horizon] / close[i] - 1。

    这是"真实可交易价格"的后复权收益；缺未来价(退市/停牌/到样本末尾)时为 NaN。
    """
    return close.iloc[i + horizon] / close.iloc[i] - 1


# ================= 第 6/7 步：逐期 RankIC + 分层 =================
def run_tests(close, factor_panel, mask, rebal_positions):
    """核心检验循环：对每个调仓日做 去极值+rank -> 各窗口未来收益 -> RankIC -> 分层。

    返回：
        ic_series    dict：{horizon: Series(按日期) 的 RankIC}
        decile_rets  DataFrame：index=调仓日，columns=组号0..9，值=该组等权未来收益
        coverage     list：每个调仓日的因子覆盖率（有 ROE 的可交易股票占比）
    """
    ic_lists = {h: [] for h in HORIZONS}
    ic_dates = {h: [] for h in HORIZONS}
    decile_rows = []
    decile_dates = []
    coverage = []

    for i in rebal_positions:
        date = close.index[i]
        factor_raw = factor_panel.iloc[i]       # 当期各股最新已披露 ROE (Series)
        act = mask.iloc[i]                      # 当期可交易掩码 (bool Series)

        # 有效股票 = 可交易 且 有 ROE 值（无 ROE 说明尚无已披露年报）
        valid0 = act & factor_raw.notna()
        n_valid = int(valid0.sum())
        n_tradable = int(act.sum())
        coverage.append(n_valid / n_tradable if n_tradable else np.nan)
        if n_valid < MIN_STOCKS:
            continue

        # ---- 第 4 步：去极值(MAD) -> 转 rank（只对当期有效股票做）----
        s_w = winsorize_mad(factor_raw[valid0])
        s_rank = to_rank(s_w)

        # ---- 第 5 步：各窗口未来收益 ----
        close_i = close.iloc[i]
        fwd = {h: close.iloc[i + h] / close_i - 1 for h in HORIZONS}

        # ---- 第 6 步：逐期 RankIC（每窗口分别算，各自剔除缺未来价的股票）----
        for h in HORIZONS:
            vh = valid0 & fwd[h].notna()
            if vh.sum() < MIN_STOCKS:
                continue
            ic = s_rank[vh].corr(fwd[h][vh], method="spearman")
            ic_lists[h].append(ic)
            ic_dates[h].append(date)

        # ---- 第 7 步：分层（按因子分 10 组，用主窗口收益，等权）----
        fwd_main = fwd[MAIN_HORIZON]
        vm = valid0 & fwd_main.notna()
        if vm.sum() < MIN_STOCKS:
            continue
        # 在 rank[0,1] 上按固定 0.1 等宽切 10 组：0=ROE最低，9=ROE最高
        q = pd.cut(s_rank[vm], bins=np.linspace(0, 1, N_GROUPS + 1),
                   labels=False, include_lowest=True)
        grp_ret = fwd_main[vm].groupby(q).mean()   # 每组等权未来收益
        grp_ret = grp_ret.reindex(range(N_GROUPS)) # 对齐到 0..9，缺组为 NaN
        decile_rows.append(grp_ret.to_numpy())
        decile_dates.append(date)

    ic_series = {h: pd.Series(ic_lists[h], index=pd.DatetimeIndex(ic_dates[h]), name=f"IC_{h}d")
                 for h in HORIZONS}
    decile_rets = pd.DataFrame(np.array(decile_rows), index=pd.DatetimeIndex(decile_dates),
                               columns=range(N_GROUPS))
    return ic_series, decile_rets, coverage


# ================= IC 统计 =================
def ic_stats(ic):
    """把 IC 序列汇总成教材 3.1 的核心指标。"""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"IC均值": np.nan, "IC标准差": np.nan, "ICIR": np.nan,
                "IC_t值": np.nan, "IC>0占比": np.nan, "期数": 0}
    mean, std = ic.mean(), ic.std()
    return {"IC均值": mean, "IC标准差": std,
            "ICIR": mean / std if std > 0 else np.nan,   # IC 信息比率 = 均值/标准差
            "IC_t值": (mean / std) * np.sqrt(n) if std > 0 else np.nan,  # t = ICIR×√n
            "IC>0占比": float((ic > 0).mean()),          # 胜率
            "期数": n}


# ================= 第 8 步：绘图与报告 =================
def plot_ic_series(ic_main, out_png):
    """图 1：IC 时序图（主窗口），附滚动均值与零线。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ic_main.index, ic_main.values, lw=0.8, color="#1f77b4", label="逐期 RankIC")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    roll = ic_main.rolling(12, min_periods=1).mean()      # 12 期滚动均值
    ax.plot(roll.index, roll.values, lw=1.6, color="#d62728", label="滚动均值(12期)")
    st = ic_stats(ic_main)
    ax.set_title(f"ROE 因子 RankIC 时序（{MAIN_HORIZON}日）  "
                 f"均值={st['IC均值']:.4f}  ICIR={st['ICIR']:.3f}")
    ax.set_xlabel("调仓日"); ax.set_ylabel("RankIC")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def plot_ic_hist(ic_main, out_png):
    """图 2：IC 分布直方图，看分布中心与离散。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ic_main.dropna(), bins=30, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(ic_main.mean(), color="#d62728", lw=2, ls="--",
               label=f"均值 = {ic_main.mean():.4f}")
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.set_title(f"ROE 因子 RankIC 分布（{MAIN_HORIZON}日）")
    ax.set_xlabel("RankIC"); ax.set_ylabel("频数")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def plot_decile_nav(decile_rets, out_png):
    """图 3：分层净值（10 组累计净值），颜色由浅(低ROE)到深(高ROE)。

    若 ROE 因子有效（正向），第 10 组(高ROE)净值应显著高于第 1 组(低ROE)，
    且整体大致单调 —— 单调是"真因子"的强证据（教材 3.1）。
    """
    nav = (1 + decile_rets).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis
    for g in range(N_GROUPS):
        color = cmap(g / (N_GROUPS - 1))
        ax.plot(nav.index, nav[g].values, lw=1.2, color=color,
                label=f"G{g + 1}" + ("(高ROE)" if g == N_GROUPS - 1 else ""))
    ax.set_title(f"ROE 因子分层累计净值（10组，{MAIN_HORIZON}日持有，等权）")
    ax.set_xlabel("调仓日"); ax.set_ylabel("累计净值(1为起点)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def plot_decay(ic_series, out_png):
    """图 4：IC 衰减曲线 —— 各未来窗口的 IC 均值 ± 1 标准差。"""
    means = [ic_series[h].mean() for h in HORIZONS]
    stds = [ic_series[h].std() for h in HORIZONS]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(HORIZONS, means, yerr=stds, marker="o", capsize=4,
                color="#1f77b4", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("ROE 因子 IC 衰减曲线")
    ax.set_xlabel("未来收益窗口（交易日）"); ax.set_ylabel("RankIC 均值")
    ax.set_xticks(HORIZONS)
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def plot_annual_ic(annual, out_png):
    """图 5：分年度 IC（柱状图），看因子在各年份是否稳定为正。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    years = [str(y) for y in annual.index]
    vals = annual["IC均值"].values
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]
    ax.bar(years, vals, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("ROE 因子分年度 IC 均值")
    ax.set_xlabel("年份"); ax.set_ylabel("RankIC 均值")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def fmt(x):
    """打印时统一格式化：数字留 4 位小数，NaN 显示为 '  nan'。"""
    return f"{x:.4f}" if pd.notna(x) else "  nan"


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 70)
    print("练习 02 · ROE 因子八步单因子检验")
    print("=" * 70)

    # ---- 第 1/2/3 步：数据 ----
    print("\n[第1步] 加载日线(算未来收益用) ...", flush=True)
    close = load_close()
    print(f"   收盘价宽表：{close.shape[0]} 个交易日 × {close.shape[1]} 只股票", flush=True)

    print("[第2/3步] 加载财报并算 ROE（PIT 对齐）...", flush=True)
    roe = load_roe_long()
    print(f"   年报 ROE 记录：{len(roe)} 条（{roe['code'].nunique()} 只股票）", flush=True)
    factor_panel = build_roe_panel(roe, close.index)
    print(f"   ROE 因子宽表：{factor_panel.shape[0]} 日 × {factor_panel.shape[1]} 股", flush=True)

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    mask = build_mask(close, uni)

    # ---- 调仓日序列：每 REBAL 个交易日一次，留出最大 60 日未来收益窗口 ----
    rebal_positions = list(range(0, len(close) - max(HORIZONS), REBAL))
    print(f"\n[第1步] 调仓期数：{len(rebal_positions)}（每 {REBAL} 个交易日一次）", flush=True)

    # ---- 第 4~7 步：检验 ----
    print("[第4~7步] 去极值+rank -> RankIC -> 分层 ...", flush=True)
    ic_series, decile_rets, coverage = run_tests(close, factor_panel, mask, rebal_positions)
    ic_main = ic_series[MAIN_HORIZON]

    # ---- 因子覆盖率（教材门2：财务因子覆盖率是硬指标）----
    cov = pd.Series(coverage).dropna()
    print(f"\n[门2·覆盖率] 平均因子覆盖率 = {cov.mean()*100:.1f}%（财务因子宜 >=60%）", flush=True)

    # ---- 第 8 步：报告 ----
    print("\n" + "=" * 70)
    print(f"[第8步·IC汇总] 主窗口({MAIN_HORIZON}日) RankIC")
    print("=" * 70)
    st = ic_stats(ic_main)
    print(pd.Series(st).map(fmt).to_string())

    print("\n" + "=" * 70)
    print("[第8步·IC衰减] 各未来窗口的 RankIC 均值 / ICIR")
    print("=" * 70)
    decay_rows = []
    for h in HORIZONS:
        s = ic_stats(ic_series[h])
        decay_rows.append({"窗口(日)": h, **s})
    decay_tbl = pd.DataFrame(decay_rows)
    print(decay_tbl[["窗口(日)", "IC均值", "IC标准差", "ICIR", "IC_t值", "IC>0占比"]].map(fmt).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"[第8步·分层] 各组等权未来收益({MAIN_HORIZON}日)均值，看是否单调")
    print("=" * 70)
    decile_mean = decile_rets.mean()
    decile_show = pd.DataFrame({
        "组": [f"G{i + 1}" for i in range(N_GROUPS)],
        "平均期收益": decile_mean.values,
    })
    decile_show["平均期收益"] = decile_show["平均期收益"].map(fmt)
    print(decile_show.to_string(index=False))
    # 单调性：组号(0..9)与各组平均收益的秩相关，越接近 +1 越单调
    mono = decile_mean.reset_index(drop=True).corr(
        pd.Series(range(N_GROUPS)), method="spearman")
    spread = decile_mean.iloc[-1] - decile_mean.iloc[0]   # 最高ROE - 最低ROE
    print(f"单调性(组号与收益的秩相关) = {mono:.3f}；第10组-第1组平均期收益差 = {spread:.4f}")

    print("\n" + "=" * 70)
    print(f"[第8步·分年度] 主窗口({MAIN_HORIZON}日) RankIC 分年度")
    print("=" * 70)
    annual = ic_main.groupby(ic_main.index.year).agg(["mean", "std", "size"])
    annual.columns = ["IC均值", "IC标准差", "期数"]
    print(annual.assign(IC均值=annual["IC均值"].map(fmt),
                        IC标准差=annual["IC标准差"].map(fmt)).to_string())

    # ---- 保存图 + 表 ----
    print("\n[第8步] 保存图表到 results/ ...", flush=True)
    plot_ic_series(ic_main, os.path.join(RESULTS, "ROE因子_1_IC时序图.png"))
    plot_ic_hist(ic_main, os.path.join(RESULTS, "ROE因子_2_IC直方图.png"))
    plot_decile_nav(decile_rets, os.path.join(RESULTS, "ROE因子_3_分层净值.png"))
    plot_decay(ic_series, os.path.join(RESULTS, "ROE因子_4_IC衰减曲线.png"))
    plot_annual_ic(annual, os.path.join(RESULTS, "ROE因子_5_分年度IC.png"))

    pd.DataFrame({f"IC_{h}d": ic_series[h] for h in HORIZONS}).to_csv(
        os.path.join(RESULTS, "ROE因子_IC时序.csv"), encoding="utf-8-sig")
    decay_tbl.to_csv(os.path.join(RESULTS, "ROE因子_IC衰减表.csv"), index=False, encoding="utf-8-sig")
    decile_rets.to_csv(os.path.join(RESULTS, "ROE因子_分层收益.csv"), encoding="utf-8-sig")
    annual.to_csv(os.path.join(RESULTS, "ROE因子_IC分年度表.csv"), encoding="utf-8-sig")

    print("已保存 5 张图 + 4 张表。")
    print("\n[验收提示] 教材练习02要求：先看 IC/分层/衰减/分年度，别急着看总收益。")
    print("  参考：质量/盈利类因子(ROE)在 A 股 RankIC 通常为正但偏弱(0.01~0.03)，")
    print("        衰减比反转慢。若 IC 接近 0 或方向反，先查 PIT 对齐与股票池口径。")

    plt.show()   # 运行结束后弹出所有图


if __name__ == "__main__":
    main()
