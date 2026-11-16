# -*- coding: utf-8 -*-
"""
练习 07 · 防过拟合三件套（参数网格热力图 + Walk-forward + p 值校正）
======================================================================

要回答的问题：反转策略的"好看"，是"真本事"还是"对着历史调参数调出来的"？

三件套：
  1. 参数网格热力图  —— 调仓周期 H(5/10/20/40/60) × 回看窗口 W(10/20/40/60/120)
                         共 25 格，画全样本年化收益，判断"尖峰 vs 平原"。
  2. Walk-forward   —— 3 年训练 + 1 年测试，滚动 4 次，看样本外表现。
  3. p 值校正       —— 25 组参数的 IC 显著性，做 Bonferroni + BH 校正（加分项）。

关键口径（与练习 01~06 一致）：
  - 收益用【后复权价】算（这是收益类因子，用 hfq 正确）。
  - 股票池用动态口径（上市日 ≤ t < 退市日），含退市股，无幸存者偏差。
  - 反转因子：signal = -(close[t] / close[t-W] - 1)，即过去 W 日收益取负，
    signal 越大 = 过去跌得越狠 = 越强的反转候选。
  - 买入 signal 前 TOP_FRAC=20%，等权持有 H 日，非重叠调仓。
  - 本练习只测【参数敏感度】，不计成本（成本已在练习 06 单独处理）。

用法（PyCharm 选 Anaconda 解释器）：python backtest_overfit.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

# ================= 可调参数 =================
REBAL_GRID = [5, 10, 20, 40, 60]       # 调仓周期 H（= 持有期，交易日）
WIN_GRID = [10, 20, 40, 60, 120]       # 回看窗口 W（反转回看，交易日）
TOP_FRAC = 0.20                        # 买入 signal 前 20%
MIN_STOCKS = 30                        # 某期有效股票少于该数则跳过
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Walk-forward
TRAIN_YEARS = 3                        # 训练窗口长度（年）
N_ROLLS = 4                            # 滚动次数
TEST_START_YEAR = 2018                 # 第 1 个测试年（其训练段 = 前 3 年）

# 多重检验
ALPHA = 0.05                           # 目标显著性水平

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # practice07/
ROOT = os.path.dirname(BASE_DIR)                               # financial data/
DAILY_ALL = os.path.join(ROOT, "practice01", "data", "daily_all.parquet")
UNIVERSE = os.path.join(ROOT, "practice01", "data_dynamic", "universe.csv")
RESULTS = os.path.join(BASE_DIR, "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 数据加载 =================
def load_hfq():
    """后复权 close 宽表（收益类因子只用 close 即可）。"""
    panel = pd.read_parquet(DAILY_ALL, columns=["date", "code", "close"])
    close = panel.pivot(index="date", columns="code", values="close").sort_index()
    close.columns.name = "code"
    close.index.name = "date"
    return close


def load_mask(close):
    """动态股票池掩码：上市日 ≤ t < 退市日（含退市股，无幸存者偏差）。"""
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    uni["list_date"] = pd.to_datetime(uni["list_date"], errors="coerce")
    uni["delist_date"] = pd.to_datetime(uni["delist_date"], errors="coerce")
    meta = uni.set_index("code").reindex(close.columns)
    ld = meta["list_date"].to_numpy()
    dd = meta["delist_date"].to_numpy().copy()
    dd[np.isnat(dd)] = np.datetime64("2099-12-31", "ns")
    dates = close.index.to_numpy()
    return pd.DataFrame((dates[:, None] >= ld[None, :]) & (dates[:, None] < dd[None, :]),
                        index=close.index, columns=close.columns)


# ================= 策略与统计 =================
def run_strategy(close, mask, H, W, top_frac=TOP_FRAC, min_stocks=MIN_STOCKS, with_ic=True):
    """在已按时间切片的 close/mask 上跑反转策略。

    返回 (周期收益 Series, IC Series)。with_ic=False 时 IC 列全空（加速，Walk-forward 训练段用不到 IC）。
    """
    n = len(close)
    rets, ics, dates = [], [], []
    for i in range(W, n - H, H):
        act = mask.iloc[i]
        sig = -(close.iloc[i] / close.iloc[i - W] - 1.0)
        sig = sig.replace([np.inf, -np.inf], np.nan)
        fwd = close.iloc[i + H] / close.iloc[i] - 1.0
        valid = sig.notna() & fwd.notna() & act
        if valid.sum() < min_stocks:
            continue
        k = max(1, int(valid.sum() * top_frac))
        target = sig[valid].nlargest(k).index
        rets.append((close.iloc[i + H][target] / close.iloc[i][target] - 1.0).mean(skipna=True))
        if with_ic:
            ics.append(sig[valid].corr(fwd[valid], method="spearman"))
        dates.append(close.index[i])
    ret_s = pd.Series(rets, index=pd.DatetimeIndex(dates), name="ret")
    ic_s = pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic") if with_ic else pd.Series(dtype=float)
    return ret_s, ic_s


def annualize(rets, H, ppy=252):
    """把 H 日周期收益序列年化。len(rets) 期 × H 日 = 覆盖的总交易日数。"""
    rets = rets.dropna()
    if len(rets) == 0:
        return np.nan
    total = (1 + rets).prod() - 1
    return (1 + total) ** (ppy / (H * len(rets))) - 1


def ic_stats(ics):
    """IC 序列 → (IC均值, ICIR, t值, p值)。p 值 = 纯噪声下撞出这么强 IC 的概率。"""
    ics = ics.dropna()
    if len(ics) < 3:
        return np.nan, np.nan, np.nan, np.nan
    m = ics.mean()
    s = ics.std(ddof=1)
    if s == 0:
        return m, np.nan, np.nan, np.nan
    t = m / (s / np.sqrt(len(ics)))
    p = 2 * (1 - stats.t.cdf(abs(t), df=len(ics) - 1))
    return m, m / s, t, p


def run_grid(close, mask, with_ic=True, verbose=False):
    """对 REBAL_GRID × WIN_GRID 全部组合跑一遍，返回一行一组合的 DataFrame。"""
    rows = []
    for H in REBAL_GRID:
        for W in WIN_GRID:
            rets, ics = run_strategy(close, mask, H, W, with_ic=with_ic)
            ann = annualize(rets, H)
            mic, icir, t, p = ic_stats(ics) if with_ic else (np.nan, np.nan, np.nan, np.nan)
            rows.append({"H调仓": H, "W回看": W, "年化收益": ann,
                         "IC均值": mic, "ICIR": icir, "t值": t, "p值": p,
                         "期数": int(len(rets.dropna()))})
            if verbose:
                print(f"    H={H:>3} W={W:>3}  年化={ann: .4f}  "
                      f"IC={mic: .4f}  t={t: .2f}  p={p: .3f}", flush=True)
    return pd.DataFrame(rows)


def to_matrix(grid, col="年化收益"):
    """长表 → W(行) × H(列) 矩阵，供热力图使用。"""
    mat = grid.pivot(index="W回看", columns="H调仓", values=col)
    return mat.reindex(index=WIN_GRID, columns=REBAL_GRID)


# ================= 尖峰 vs 平原 =================
def spike_plateau(grid):
    """用"最好格 vs 它的 8 邻域"粗略判断尖峰/平原。"""
    heat = to_matrix(grid, "年化收益")
    vals = heat.to_numpy(dtype=float)
    bi, bj = np.unravel_index(np.nanargmax(vals), vals.shape)
    best = vals[bi, bj]
    best_h = REBAL_GRID[bj]
    best_w = WIN_GRID[bi]
    neigh = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = bi + di, bj + dj
            if 0 <= ni < vals.shape[0] and 0 <= nj < vals.shape[1]:
                if np.isfinite(vals[ni, nj]):
                    neigh.append(vals[ni, nj])
    neigh_mean = np.mean(neigh) if neigh else np.nan
    drop = best - neigh_mean
    median_all = np.nanmedian(vals)
    frac_half = np.nansum(vals >= 0.5 * best) / np.sum(np.isfinite(vals))
    return {
        "最优格(H,W)": (best_h, best_w), "最优格年化": best,
        "8邻域平均": neigh_mean, "相对邻域落差": drop,
        "25格中位数": median_all, "≥最优格50%的格子占比": frac_half,
    }


# ================= 多重检验校正 =================
def multiple_testing(grid):
    """对 25 组参数的 IC p 值做 Bonferroni 与 BH 校正，返回 (带标记的 grid, bonf阈值, BH通过数, 有效检验数)。"""
    g = grid.copy()
    ps = g["p值"].dropna().sort_values()
    m = len(ps)
    if m == 0:
        g["Bonf显著"] = False
        g["BH显著"] = False
        return g, np.nan, 0, 0

    # Bonferroni：p <= ALPHA / m 才算显著
    bonf_threshold = ALPHA / m
    g["Bonf显著"] = g["p值"] <= bonf_threshold

    # BH：找最大 k 使 p_(k) <= (k/m)*ALPHA，则 p_1..p_k 显著
    bh_cut = 0
    for k in range(1, m + 1):
        if ps.iloc[k - 1] <= (k / m) * ALPHA:
            bh_cut = k
    bh_threshold = ps.iloc[bh_cut - 1] if bh_cut > 0 else np.nan
    g["BH显著"] = g["p值"] <= bh_threshold if bh_cut > 0 else False

    return g, bonf_threshold, bh_cut, m


# ================= Walk-forward =================
def walk_forward(close, mask):
    """3 年训练 + 1 年测试，滚动 N_ROLLS 次。训练段网格选最优 H/W，测试段用该参数跑样本外。"""
    rows = []
    for r in range(N_ROLLS):
        test_y = TEST_START_YEAR + r
        train_lo, train_hi = test_y - TRAIN_YEARS, test_y - 1
        tr_close = close[(close.index.year >= train_lo) & (close.index.year <= train_hi)]
        te_close = close[close.index.year == test_y]
        tr_mask = mask.loc[tr_close.index]
        te_mask = mask.loc[te_close.index]

        grid = run_grid(tr_close, tr_mask, with_ic=False)   # 训练段只要年化收益来选参
        best = grid.loc[grid["年化收益"].idxmax()]
        H, W = int(best["H调仓"]), int(best["W回看"])

        rets, _ = run_strategy(te_close, te_mask, H, W, with_ic=False)
        oos_ann = annualize(rets, H)
        oos_cum = (1 + rets.dropna()).prod() - 1 if len(rets.dropna()) else np.nan

        rows.append({
            "roll": r + 1,
            "训练段": f"{train_lo}-{train_hi}",
            "测试年": test_y,
            "最优H": H, "最优W": W,
            "训练段最优年化": best["年化收益"],
            "测试段(OOS)年化": oos_ann,
            "测试段累计收益": oos_cum,
            "测试段期数": int(len(rets.dropna())),
        })
        print(f"    roll{r+1} 训练{train_lo}-{train_hi} -> 测试{test_y}  "
              f"选参 H={H},W={W}  样本内年化{best['年化收益']:.4f}  OOS年化{oos_ann:.4f}", flush=True)
    return pd.DataFrame(rows)


# ================= 绘图 =================
def plot_heatmap(mat, out, title):
    """年化收益热力图。红 = 高收益，蓝 = 低/负收益（中式配色：红涨）。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    vals = mat.to_numpy(dtype=float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin, vmax = -0.05, 0.05
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    ax.imshow(vals, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(REBAL_GRID))); ax.set_xticklabels(REBAL_GRID)
    ax.set_yticks(range(len(WIN_GRID))); ax.set_yticklabels(WIN_GRID)
    ax.set_xlabel("调仓周期 H（交易日）"); ax.set_ylabel("回看窗口 W（交易日）")
    ax.set_title(title)
    for i in range(len(WIN_GRID)):
        for j in range(len(REBAL_GRID)):
            v = vals[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v * 100:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    fig.colorbar(ax.images[0], ax=ax, label="年化收益")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_walkforward(wf, out):
    """Walk-forward：训练段(样本内) vs 测试段(样本外) 年化对比柱状图。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(wf))
    w = 0.35
    ax.bar(x - w / 2, wf["训练段最优年化"], w, label="训练段(样本内)最优年化", color="#1f77b4")
    ax.bar(x + w / 2, wf["测试段(OOS)年化"], w, label="测试段(样本外)年化", color="#d62728")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['训练段']}\n→{r['测试年']}" for _, r in wf.iterrows()], fontsize=8)
    ax.set_ylabel("年化收益")
    ax.set_title("Walk-forward：样本内 vs 样本外（样本外更低 = 过拟合证据）")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ================= 主流程 =================
def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("练习 07 · 防过拟合三件套（反转因子 调仓周期 × 回看窗口）")
    print("=" * 72)

    print("\n[1] 加载数据 ...", flush=True)
    close = load_hfq()
    mask = load_mask(close)
    close = close.loc[START_DATE:END_DATE]
    mask = mask.loc[START_DATE:END_DATE]
    print(f"    日线 {close.shape}（{close.index[0].date()} ~ {close.index[-1].date()}）", flush=True)

    # ---- 1. 全样本网格 + 热力图 ----
    print("\n[2] 全样本 5×5 网格（25 组合）...", flush=True)
    grid = run_grid(close, mask, with_ic=True, verbose=True)

    heat = to_matrix(grid, "年化收益")
    heat.to_csv(os.path.join(RESULTS, "网格_年化收益热力图.csv"), encoding="utf-8-sig")
    grid.to_csv(os.path.join(RESULTS, "网格_全部指标.csv"), encoding="utf-8-sig", index=False)
    plot_heatmap(heat, os.path.join(RESULTS, "网格_年化收益热力图.png"),
                 "反转因子 年化收益热力图（全样本 2015–2024，不计成本）")

    # ---- 2. 尖峰 vs 平原 ----
    sp = spike_plateau(grid)
    print("\n[3] 尖峰 vs 平原 度量 ...", flush=True)
    print(f"    最优格 (H={sp['最优格(H,W)'][0]}, W={sp['最优格(H,W)'][1]}) 年化 {sp['最优格年化']:.4f}")
    print(f"    最优格 8 邻域平均 {sp['8邻域平均']:.4f}，相对邻域落差 {sp['相对邻域落差']:.4f}")
    print(f"    25 格中位数 {sp['25格中位数']:.4f}，≥最优格50%的格子占比 {sp['≥最优格50%的格子占比']:.2f}")
    print("    判读：邻域落差小 / ≥50%占比高 → 平原（稳健）；落差大 → 尖峰（过拟合嫌疑）", flush=True)

    # ---- 3. p 值校正 ----
    g2, bonf_th, bh_cut, m = multiple_testing(grid)
    g2.to_csv(os.path.join(RESULTS, "多重检验校正.csv"), encoding="utf-8-sig", index=False)
    print("\n[4] Bonferroni / BH 校正 ...", flush=True)
    print(f"    有效检验数 m = {m}；Bonferroni 阈值 = {ALPHA}/{m} = {bonf_th:.4f}")
    print(f"    Bonferroni 显著 {int(g2['Bonf显著'].sum())} 个；BH 显著 {int(g2['BH显著'].sum())} 个", flush=True)

    # ---- 4. Walk-forward ----
    print("\n[5] Walk-forward（3 年训练 + 1 年测试，滚动 4 次）...", flush=True)
    wf = walk_forward(close, mask)
    wf.to_csv(os.path.join(RESULTS, "Walkforward表.csv"), encoding="utf-8-sig", index=False)
    plot_walkforward(wf, os.path.join(RESULTS, "Walkforward_样本内外对比.png"))
    print("\nWalk-forward 表：")
    print(wf.map(lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) and pd.notna(x) else x)
              .to_string(index=False))

    # ---- 汇总 ----
    print("\n" + "=" * 72)
    print("产出文件（results/）：")
    for f in ["网格_年化收益热力图.png", "网格_年化收益热力图.csv", "网格_全部指标.csv",
              "多重检验校正.csv", "Walkforward表.csv", "Walkforward_样本内外对比.png"]:
        print("   ", f)
    print("\n[验收提示]")
    print("  1) 看热力图：好结果是一大片(平原)还是孤点(尖峰)？")
    print("  2) 看 Walkforward 表：样本外年化是不是系统性低于样本内？")
    print("  3) 看 多重检验校正.csv：最好的格子校正后还显著吗？")
    print("  4) 试验次数诚实记录见《试验次数记录.md》，请核对补充。")
    plt.show()


if __name__ == "__main__":
    main()
