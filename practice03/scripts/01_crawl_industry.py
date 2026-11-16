# -*- coding: utf-8 -*-
"""爬取申万一级行业分类，得到「股票代码 -> 行业名称」映射。

接口：akshare(申万)
  - sw_index_first_info()   申万一级行业指数列表(31个)，含 行业代码/行业名称
  - index_component_sw()    申万指数成分股
输出：practice03/data/industry_map.csv (code, name, industry)
用法：
  python 01_crawl_industry.py
说明：这是「当前(静态)」的行业分类，不是 point-in-time（申万分类会随调仓调整）。
      对中性化而言，固定一套分类 + 在报告里声明口径即可（教材 1.2 也提示分类会变）。
"""
import os
import time
import random

import akshare as ak
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # practice03/scripts
ROOT = os.path.dirname(BASE_DIR)                        # practice03
OUT = os.path.join(ROOT, "data", "industry_map.csv")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # 1) 申万一级行业列表（31 个）
    board = ak.sw_index_first_info()
    # 列名：行业代码 / 行业名称 / 成份个数 / ...
    codes = board["行业代码"].astype(str).str.split(".").str[0]   # "801010.SI" -> "801010"
    names = board["行业名称"].tolist()
    print(f"申万一级行业 {len(names)} 个", flush=True)

    # 2) 逐个行业取成分股，构建 code -> industry 映射
    rows = []
    for i, (code, name) in enumerate(zip(codes, names), 1):
        for attempt in range(3):
            try:
                cons = ak.index_component_sw(symbol=code)         # 列：序号/证券代码/证券名称/...
                for _, r in cons.iterrows():
                    rows.append((str(r["证券代码"]).zfill(6), r["证券名称"], name))
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [{name}] 失败: {type(e).__name__}: {e}", flush=True)
                time.sleep(1.5 * (attempt + 1))
        time.sleep(random.uniform(0.1, 0.3))
        if i % 10 == 0 or i == len(names):
            print(f"  已取 {i}/{len(names)} 个行业", flush=True)

    df = pd.DataFrame(rows, columns=["code", "name", "industry"]).drop_duplicates("code")
    df = df.sort_values("code").reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"完成：{len(df)} 只股票 -> {OUT}", flush=True)
    print("行业分布（各行业股票数）：", flush=True)
    print(df["industry"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
