# -*- coding: utf-8 -*-
"""补全现存股上市日期 -> 生成最终宇宙表 + 每年末"当时存在 vs 现在还在"统计表。
- 现存股上市日期：沪主板/科创板/深A 交易所列表(批量, 准确)
- 退市股上市日期/退市日期：来自退市清单(已含)
- 缺上市日期的现存股：回退用其日线首根K线日期
输出: data/universe/universe_final.csv, data/universe/year_end_counts.csv
"""
import glob
import os
import akshare as ak
import pandas as pd

UNIVERSE = "data/universe/universe.csv"
DAILY_DIR = "data/daily"


def fetch_list_dates():
    """返回 {code: 上市日期(date)}，覆盖现存沪深A股。"""
    d = {}
    for sym in ["主板A股", "科创板"]:
        df = ak.stock_info_sh_name_code(symbol=sym)
        c = df[df.columns[0]].astype(str).str.zfill(6)
        dt = pd.to_datetime(df["上市日期"], errors="coerce")
        d.update(dict(zip(c, dt)))
    df = ak.stock_info_sz_name_code(symbol="A股列表")
    c = df["A股代码"].astype(str).str.zfill(6)
    dt = pd.to_datetime(df["A股上市日期"], errors="coerce")
    d.update(dict(zip(c, dt)))
    return d


def main():
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    uni["list_date"] = pd.to_datetime(uni["list_date"], errors="coerce")
    uni["delist_date"] = pd.to_datetime(uni["delist_date"], errors="coerce")

    list_map = fetch_list_dates()
    print(f"交易所列表拿到上市日期 {len(list_map)} 条")

    # 现存股填上市日期
    cur = uni["status"] == "current"
    missing_before = uni.loc[cur, "list_date"].isna().sum()
    uni.loc[cur, "list_date"] = uni.loc[cur, "code"].map(list_map)
    still_missing = uni.loc[cur, "list_date"].isna()

    # 回退：用日线首根K线日期
    if still_missing.any():
        for code in uni.loc[cur & still_missing, "code"]:
            p = os.path.join(DAILY_DIR, f"{code}.parquet")
            if os.path.exists(p):
                d0 = pd.read_parquet(p, columns=["date"])["date"].min()
                uni.loc[uni["code"] == code, "list_date"] = pd.to_datetime(d0)
    after_missing = uni.loc[cur, "list_date"].isna().sum()
    print(f"现存股上市日期: 交易所补全后仍缺 {after_missing} 只(已用首根K线回退)")

    uni = uni.sort_values("code").reset_index(drop=True)
    uni[["code", "name", "market", "list_date", "delist_date", "status"]].to_csv(
        "data/universe/universe_final.csv", index=False, encoding="utf-8-sig")

    # 每年末统计
    rows = []
    for y in range(2015, 2025):
        ye = pd.Timestamp(f"{y}-12-31")
        existed = (uni["list_date"] <= ye) & (uni["delist_date"].isna() | (uni["delist_date"] > ye))
        still_now = existed & (uni["status"] == "current")
        rows.append({
            "year_end": f"{y}-12-31",
            "当时存在": int(existed.sum()),
            "现在还在": int(still_now.sum()),
            "差值(已退市)": int(existed.sum() - still_now.sum()),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv("data/universe/year_end_counts.csv", index=False, encoding="utf-8-sig")

    print("\n=== 每年末股票数统计 ===")
    print(tbl.to_string(index=False))
    print(f"\n最终宇宙表已保存: data/universe/universe_final.csv")
    print(f"统计表已保存: data/universe/year_end_counts.csv")


if __name__ == "__main__":
    main()
