# -*- coding: utf-8 -*-
"""校验已爬取的日线：每只的日期范围、行数，退市股末日应≈退市日，现存股末日应=2024-12-31。"""
import os
import glob
import pandas as pd

UNIVERSE = "data/universe/universe.csv"
DAILY_DIR = "data/daily"

uni = pd.read_csv(UNIVERSE, dtype={"code": str})
uni = uni.set_index("code")

rows = []
for p in sorted(glob.glob(os.path.join(DAILY_DIR, "*.parquet"))):
    code = os.path.splitext(os.path.basename(p))[0]
    df = pd.read_parquet(p)
    meta = uni.loc[code] if code in uni.index else None
    status = meta["status"] if meta is not None else "?"
    delist = meta["delist_date"] if meta is not None else "?"
    n = len(df)
    dmin = df["date"].min()
    dmax = df["date"].max()
    close_min = df["close"].min()
    close_max = df["close"].max()
    rows.append({
        "code": code, "name": meta["name"] if meta is not None else "?",
        "status": status, "delist_date": delist,
        "rows": n, "first": dmin, "last": dmax,
        "close_min": close_min, "close_max": close_max,
    })

out = pd.DataFrame(rows)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 220)

print("=== 逐只校验 ===")
print(out.to_string(index=False))

print("\n=== 异常检查 ===")
issues = []
for _, r in out.iterrows():
    if r["status"] == "delisted":
        # 末日应 <= 退市日，且距离退市日不超过 ~30 自然日
        if pd.notna(r["delist_date"]):
            dd = pd.to_datetime(r["delist_date"])
            gap = (dd - r["last"]).days
            if gap < -1 or gap > 45:
                issues.append(f'{r["code"]} {r["name"]}: 退市日{dd.date()} 末日{r["last"].date()} 差{gap}天')
    else:
        if r["last"] < pd.Timestamp("2024-12-31"):
            issues.append(f'{r["code"]} {r["name"]}: 现存股末日={r["last"].date()} < 2024-12-31')
    if r["last"] > pd.Timestamp("2024-12-31"):
        issues.append(f'{r["code"]} {r["name"]}: 末日{ r["last"].date()} 超出窗口2024-12-31')
    if r["close_min"] <= 0:
        issues.append(f'{r["code"]} {r["name"]}: 收盘价<=0 (min={r["close_min"]})')

print("问题数:", len(issues))
for x in issues:
    print("  !!", x)

# 汇总
print("\n=== 汇总 ===")
print(f"共校验 {len(out)} 只；总行数 {out['rows'].sum()}")
print("按状态:", out.groupby("status")["rows"].agg(["count", "sum"]).to_string())
