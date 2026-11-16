# -*- coding: utf-8 -*-
"""
练习一 · 幸存者偏差回测：反转因子 × 动态/静态口径
========================================================

要回答的问题：如果只用"现在还在的股票"做回测（幸存者偏差），
20日（以及5/10/60日）反转因子的收益会被高估多少？

完整流程（对应 1~5 步）：
  1. 加载数据    —— 把 5321 只股票的收盘价拼成一张"宽表"
                     （行 = 交易日，列 = 股票代码，值 = 收盘价）
  2. 口径过滤    —— 构造两种"谁在池子里"的掩码：
                     动态(正确)：上市日 ~ 退市日之间才算数
                     静态(有偏)：只用今天还上市的股票
  3. 因子计算    —— 过去 N 日收益率取负号 = 反转信号（跌越多信号越强）
  4. 回测        —— 每 21 个交易日调仓，买信号最高的 20%（跌得最多的），
                     同时记录信号最低的 20%（涨得最多的）做对照
  5. 绩效对比    —— 对每个因子、每种口径算 IC（核心）、年化收益/夏普，对比差值

用法：任意目录下运行均可（数据路径基于脚本自身位置自动定位）
    python backtest_reversal.py

结果：打印 IC 表 + 收益/夏普对比表，并保存为 4 个中文命名的 CSV（utf-8，可用 Excel 打开）。
"""

import os

import numpy as np
import pandas as pd

# ================= 可调参数 =================
FACTORS = [5, 10, 20, 60]   # 反转因子的回看窗口（交易日）
REBAL = 21                  # 调仓间隔（交易日，约一个月）
TOP_FRAC = 0.2              # 选信号最高/最低的前 20%（即"五分位"的两端）
MIN_STOCKS = 30             # 某期有效股票少于这个数就跳过该期（避免噪声）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 脚本所在目录(practice01)
DAILY_ALL = os.path.join(BASE_DIR, "data", "daily_all.parquet")  # 合并后的全样本价格数据
UNIVERSE = os.path.join(BASE_DIR, "data_dynamic", "universe.csv")  # 动态宇宙表
RESULTS = os.path.join(BASE_DIR, "results")             # 输出目录（CSV 统一放这里）


# ================= 第 1 步：加载数据 =================
def load_close():
    """读取合并后的单个 parquet 文件，拼成宽表 close。

    数据已预先合并成 data/daily_all.parquet（9,244,796 行），读单个文件：
    - 快（约 5 秒），且无"读目录偶发读错"的问题
    - 路径是单个 str，兼容各版本 pandas 的类型检查

    返回 DataFrame：index=交易日(date)，columns=股票代码(code)，值=后复权收盘价。
    某只股票在它上市前/退市后的格子是 NaN —— 这天然就是"动态"的价格数据。
    """
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close"])
    close = panel.pivot(index="date", columns="code", values="close")
    close = close.sort_index()                 # 按交易日从小到大
    close.columns.name = "code"
    close.index.name = "date"
    return close


# ================= 第 2 步：口径过滤（动态/静态） =================
def build_masks(close, uni):
    """根据上市日/退市日构造两种"可交易"掩码（与 close 同形状的布尔宽表）。

    动态口径（正确）：交易日 t 满足  上市日 <= t < 退市日
        退市股只在其退市前可用，新股只在其上市后可用 —— 无幸存者偏差、无前视。
    静态口径（有偏）：只有"今天还在上市"的股票可用（退市股永远剔除）。
        这正是大多数行情平台默认给的"现存股票列表"。
    """
    uni = uni.copy()
    uni["list_date"] = pd.to_datetime(uni["list_date"], errors="coerce")
    uni["delist_date"] = pd.to_datetime(uni["delist_date"], errors="coerce")

    codes = close.columns                            # 有数据的 5321 只
    meta = uni.set_index("code").reindex(codes)      # 对齐到 close 的列顺序

    ld = meta["list_date"].to_numpy()                # 每只股票的上市日 (N,)
    dd = meta["delist_date"].to_numpy().copy()       # 每只股票的退市日 (N,)，NaT=未退市

    # 未退市的股票，把退市日填成一个极远的未来日期，方便统一用"t < 退市日"判断
    dd[np.isnat(dd)] = np.datetime64("2099-12-31", "ns")

    dates = close.index.to_numpy()                   # 全部交易日 (T,)

    # 广播：(T,1) 与 (1,N) 比较 -> (T,N) 的布尔矩阵
    after_listing = dates[:, None] >= ld[None, :]    # 是否已上市
    before_delisting = dates[:, None] < dd[None, :]  # 是否还没退市
    dynamic_mask = pd.DataFrame(
        after_listing & before_delisting,
        index=close.index, columns=close.columns,
    )

    # 静态口径：每只股票"是否今天还在"是一个固定的布尔值，沿时间轴复制成宽表
    # （复用上面的 meta；先取成 numpy 数组再比较，避免链式比较触发类型误判）
    status_arr = meta["status"].to_numpy()          # 每只股票的当前状态("current"/"delisted")
    is_current = status_arr == "current"            # 布尔数组：True=今天还在
    static_mask = pd.DataFrame(
        np.tile(is_current, (len(close), 1)),
        index=close.index, columns=close.columns,
    )
    return dynamic_mask, static_mask


# ================= 第 3 步：因子计算与回测 =================
def run_backtest(close, signal, mask, rebal_positions):
    """在给定反转信号与可交易掩码下，做"多空反转"组合回测。

    参数：
        close           收盘价宽表（用于算下一期持有收益）
        signal          反转信号宽表（与 close 同形状），值越大 = 过去跌得越多
        mask            可交易掩码（与 close 同形状的布尔宽表）
        rebal_positions 调仓日在 close 中的行位置列表

    返回：
        long_s   做多腿收益序列（买信号最高的 TOP_FRAC，即跌最多的）
        short_s  做空腿收益序列（买信号最低的 TOP_FRAC，即涨最多的，作对照）
        多空价差 = long_s - short_s，是反转因子剥离市场后的纯收益
    """
    # 下一期（REBAL 个交易日后）的收益率：作为持有期收益
    # 注意：若某股在持有期内退市/停牌，shift(-REBAL) 会是 NaN，
    #       下面取均值时用 skipna=True 自动剔除（见注释①）。
    fwd_ret = close.shift(-REBAL) / close - 1

    long_rets, short_rets, valid_dates = [], [], []
    for i in rebal_positions:
        sig = signal.iloc[i]           # 当期每只股票的反转信号
        fwd = fwd_ret.iloc[i]          # 当期每只股票的下一期收益
        act = mask.iloc[i]             # 当期"可交易"的股票

        # 有效股票 = 可交易 且 有信号（信号为 NaN 说明上市不足 N 天，无回看历史）
        valid = act & sig.notna()
        if valid.sum() < MIN_STOCKS:
            continue                   # 有效股票太少，跳过这一期

        sig_v = sig[valid]
        k = max(1, int(len(sig_v) * TOP_FRAC))       # 前/后 20% 的股票数量

        long_codes = sig_v.nlargest(k).index          # 信号最高 = 过去跌最多 -> 做多
        short_codes = sig_v.nsmallest(k).index        # 信号最低 = 过去涨最多 -> 做空对照

        # ① 等权收益；skipna=True 剔除持有期内无下期价格的股票
        long_rets.append(fwd[long_codes].mean(skipna=True))
        short_rets.append(fwd[short_codes].mean(skipna=True))
        valid_dates.append(close.index[i])

    idx = pd.DatetimeIndex(valid_dates)
    long_s = pd.Series(long_rets, index=idx, name="long")
    short_s = pd.Series(short_rets, index=idx, name="short")
    return long_s, short_s


# ================= 第 4 步：绩效统计 =================
def perf_stats(rets):
    """把一个调仓期收益序列汇总成绩效指标（年化按每月约 12 个调仓期）。"""
    rets = rets.dropna()
    if len(rets) == 0:
        return {"累计收益": np.nan, "年化收益": np.nan,
                "年化波动": np.nan, "夏普": np.nan}
    total = (1 + rets).prod() - 1                  # 整个样本期的累计收益
    n = len(rets)                                  # 调仓期数
    ppy = 252 / REBAL                              # 每年约多少个调仓期
    ann = (1 + total) ** (ppy / n) - 1             # 几何年化收益
    vol = rets.std() * np.sqrt(ppy)                # 年化波动率
    sharpe = ann / vol if vol > 0 else np.nan      # 夏普比率（无风险利率取 0）
    return {"累计收益": total, "年化收益": ann,
            "年化波动": vol, "夏普": sharpe}


# ================= 第 5 步：计算 IC（信息系数） =================
def compute_ic(signal, close, mask, rebal_positions):
    """计算每个调仓期的 RankIC：信号值与"下一期收益"在横截面上的 Spearman 秩相关。

    IC 衡量因子对未来收益的预测能力：
        IC > 0  说明信号大的股票（这里=过去跌得多的）未来收益更高，因子方向有效
        IC 越接近 +1 预测越强，接近 0 无预测力，负值说明方向反了
    之所以用 RankIC（秩相关）而不是 Pearson：秩相关只看排序、更抗极端值，
    是 A 股因子研究中约定俗成的 IC 口径。

    返回：一个 Series（index=调仓日期，值=当期横截面 RankIC）。
    """
    fwd_ret = close.shift(-REBAL) / close - 1   # 下一期持有收益（与回测同一口径）

    ic_list, dates = [], []
    for i in rebal_positions:
        sig = signal.iloc[i]           # 当期每只股票的反转信号
        fwd = fwd_ret.iloc[i]          # 当期每只股票的下一期收益
        act = mask.iloc[i]             # 当期"可交易"的股票

        # 算 IC 需要"有信号 且 能观察到下一期收益"的股票（缺未来价则无法判断预测力）
        valid = act & sig.notna() & fwd.notna()
        if valid.sum() < MIN_STOCKS:
            continue

        ic = sig[valid].corr(fwd[valid], method="spearman")   # 横截面秩相关
        ic_list.append(ic)
        dates.append(close.index[i])

    return pd.Series(ic_list, index=pd.DatetimeIndex(dates), name="IC")


def ic_stats(ic_series):
    """把 IC 序列汇总成常用指标：均值、标准差、ICIR、正值占比。"""
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {"IC均值": np.nan, "IC标准差": np.nan,
                "ICIR": np.nan, "IC>0占比": np.nan}
    mean = ic.mean()
    std = ic.std()
    return {
        "IC均值": mean,                          # 平均预测力（越远离 0 越有效）
        "IC标准差": std,                          # 预测力的稳定性
        "ICIR": mean / std if std > 0 else np.nan,  # IC 信息比率 = 均值/标准差
        "IC>0占比": (ic > 0).mean(),               # IC 为正的期数占比（>0.5 方向稳定）
    }


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("[第1步] 加载数据 ...", flush=True)
    close = load_close()
    print(f"   收盘价宽表：{close.shape[0]} 个交易日 × {close.shape[1]} 只股票", flush=True)

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    print("[第2步] 构造动态/静态口径掩码 ...", flush=True)
    dynamic_mask, static_mask = build_masks(close, uni)

    # 从第 60 个交易日开始调仓（给最长的 60 日因子留出回看窗口），
    # 截止到 len-REBAL，保证每期都有完整的下一期持有收益。
    rebal_positions = list(range(60, len(close) - REBAL, REBAL))
    print(f"   调仓期数：{len(rebal_positions)}（每 {REBAL} 个交易日一次）\n", flush=True)

    results = []          # 汇总所有 (因子 × 口径 × 腿) 的绩效
    series = {}           # 顺便保存收益序列，便于以后画图
    ic_results = []       # 汇总所有 (因子 × 口径) 的 IC 指标（练习一核心）
    ic_series = {}        # 保存每个 (因子 × 口径) 的 IC 时间序列

    for N in FACTORS:
        print(f"   计算 {N} 日反转因子 ...", flush=True)
        # 因子：过去 N 日收益率，取负号即"反转"信号（跌越多 -> 信号越强）
        ret_N = close / close.shift(N) - 1
        signal = -ret_N

        for label, mask in [("动态", dynamic_mask), ("静态", static_mask)]:
            long_s, short_s = run_backtest(close, signal, mask, rebal_positions)
            spread = long_s - short_s           # 多空价差（因子纯收益）

            for leg, s in [("做多(买跌)", long_s), ("做空(买涨)", short_s),
                           ("多空价差", spread)]:
                st = perf_stats(s)
                results.append({
                    "因子": f"{N}日反转", "口径": label, "腿": leg,
                    "累计收益": st["累计收益"], "年化收益": st["年化收益"],
                    "年化波动": st["年化波动"], "夏普": st["夏普"],
                })
            series[(N, label, "long")] = long_s
            series[(N, label, "short")] = short_s
            series[(N, label, "spread")] = spread

            # ===== 计算 IC（练习一核心指标）=====
            ic_s = compute_ic(signal, close, mask, rebal_positions)
            ic_results.append({"因子": f"{N}日反转", "口径": label, **ic_stats(ic_s)})
            ic_series[f"{N}日_{label}"] = ic_s

    # ===== 输出对比表 =====
    tbl = pd.DataFrame(results)
    fmt = lambda x: f"{x:.4f}" if pd.notna(x) else "  nan"
    print("=" * 70)
    print("[第3步] 反转因子回测结果（年化收益 / 夏普）")
    print("=" * 70)
    # 只打印关键列，避免太宽：因子、口径、腿、年化收益、夏普
    show = tbl[["因子", "口径", "腿", "年化收益", "夏普"]].copy()
    show["年化收益"] = show["年化收益"].map(fmt)
    show["夏普"] = show["夏普"].map(fmt)
    print(show.to_string(index=False))

    # ===== 幸存者偏差：静态口径相对动态口径"高估"了多少 =====
    print("\n" + "=" * 70)
    print("[第4步] 幸存者偏差（静态口径 减去 动态口径 的年化收益差）")
    print("=" * 70)
    ann_pivot = tbl.pivot_table(index="因子", columns=["口径", "腿"], values="年化收益")
    bias = ann_pivot["静态"] - ann_pivot["动态"]      # 正的 = 静态口径高估了
    bias.columns = [f"高估·{c}" for c in bias.columns]
    print(bias.map(fmt).to_string())

    # ===== IC 表（练习一核心指标，单独着重输出）=====
    print("\n" + "=" * 70)
    print("[第5步] 因子 IC（信号与下一期收益的横截面 Rank 相关）")
    print("=" * 70)
    ic_tbl = pd.DataFrame(ic_results)
    ic_show = ic_tbl.copy()
    for c in ["IC均值", "IC标准差", "ICIR", "IC>0占比"]:
        ic_show[c] = ic_show[c].map(fmt)
    print(ic_show.to_string(index=False))
    ic_tbl.to_csv(os.path.join(RESULTS, "反转因子_IC汇总表.csv"),
                  index=False, encoding="utf-8-sig")

    print("\nIC 的幸存者偏差（静态口径 - 动态口径 的 IC均值差，正=静态高估预测力）：")
    ic_pivot = ic_tbl.pivot_table(index="因子", columns="口径", values="IC均值")
    print((ic_pivot["静态"] - ic_pivot["动态"]).map(fmt).to_string())

    # ===== 保存结果（统一存到 results/ 目录，中文命名）=====
    tbl.to_csv(os.path.join(RESULTS, "反转因子_绩效表.csv"), index=False,
               encoding="utf-8-sig")
    # 收益序列（宽表：列 = 因子_口径_腿），方便画累计净值曲线
    ser_df = pd.DataFrame(series)
    ser_df.to_csv(os.path.join(RESULTS, "反转因子_收益时间序列.csv"),
                  encoding="utf-8-sig")
    # IC 序列，便于画 IC 走势图（时间序列）
    ic_ser_df = pd.DataFrame(ic_series)
    ic_ser_df.to_csv(os.path.join(RESULTS, "反转因子_IC时间序列.csv"),
                     encoding="utf-8-sig")
    print("\n已保存到 results/：反转因子_绩效表.csv、反转因子_IC汇总表.csv、"
          "反转因子_IC时间序列.csv、反转因子_收益时间序列.csv", flush=True)


if __name__ == "__main__":
    main()
