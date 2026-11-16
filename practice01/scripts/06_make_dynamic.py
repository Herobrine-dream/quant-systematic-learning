# -*- coding: utf-8 -*-
"""生成动态(point-in-time)样本的宇宙与每年末活跃清单。
价格数据无需复制：全样本 data/daily 每只股票的文件天然只覆盖其实际上市~退市区间。
输出:
  data_dynamic/universe.csv                 —— 每只股票的上市日/退市日(任意日期过滤用)
  data_dynamic/membership_by_year_end.csv   —— 每年末(2015-2024)当时的活跃股票清单
"""
import os

import pandas as pd

UNI = "data/universe/universe_final.csv"
DST = "data_dynamic"


def main():
    uni = pd.read_csv(UNI, dtype={"code": str})
    uni["list_date"] = pd.to_datetime(uni["list_date"], errors="coerce")
    uni["delist_date"] = pd.to_datetime(uni["delist_date"], errors="coerce")

    os.makedirs(DST, exist_ok=True)

    # 1) 宇宙表（含上市/退市日）
    uni[["code", "name", "market", "list_date", "delist_date", "status"]].to_csv(
        os.path.join(DST, "universe.csv"), index=False, encoding="utf-8-sig")

    # 2) 每年末活跃清单：list_date <= 年末 且 (退市日为空 或 退市日 > 年末)
    rows = []
    for y in range(2015, 2025):
        ye = pd.Timestamp(f"{y}-12-31")
        active = uni[(uni["list_date"] <= ye)
                     & (uni["delist_date"].isna() | (uni["delist_date"] > ye))]
        for code, name in zip(active["code"], active["name"]):
            rows.append((f"{y}-12-31", code, name))
    m = pd.DataFrame(rows, columns=["year_end", "code", "name"])
    m.to_csv(os.path.join(DST, "membership_by_year_end.csv"), index=False,
             encoding="utf-8-sig")

    print(f"宇宙表: {len(uni)} 只 -> {DST}/universe.csv")
    print(f"每年末活跃清单: {len(m)} 行 -> {DST}/membership_by_year_end.csv")
    print("各年末活跃数:")
    print(m.groupby("year_end").size().to_string())


if __name__ == "__main__":
    main()
