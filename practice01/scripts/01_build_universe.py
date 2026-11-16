# -*- coding: utf-8 -*-
"""构建股票宇宙清单：沪深A股全样本（现存 + 已退市），含上市日/退市日。
口径：
- 沪深A股代码前缀：600/601/603/605/688(沪), 000/001/002/003/300/301(深)
- 排除B股(900/200)、北交所(4/8/9开头)
- 退市日期：沪退市清单第4列虽名为"暂停上市日期"，实为终止上市(退市)日期
"""
import akshare as ak
import pandas as pd

A_PREFIX = ("600", "601", "603", "605", "688", "000", "001", "002", "003", "300", "301")


def is_a(code: str) -> bool:
    return code.startswith(A_PREFIX)


def load_delisted():
    sh = ak.stock_info_sh_delist()
    sz = ak.stock_info_sz_delist()
    # 沪: 公司代码/公司简称/上市日期/暂停上市日期(实为退市日期)
    sh = sh.rename(columns={sh.columns[0]: "code", sh.columns[1]: "name",
                            sh.columns[2]: "list_date", sh.columns[3]: "delist_date"})
    sz = sz.rename(columns={sz.columns[0]: "code", sz.columns[1]: "name",
                            sz.columns[2]: "list_date", sz.columns[3]: "delist_date"})
    sh["market"] = "SH"
    sz["market"] = "SZ"
    df = pd.concat([sh, sz], ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")
    return df


def load_current():
    df = ak.stock_info_a_code_name()
    df = df.rename(columns={"code": "code", "name": "name"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["market"] = df["code"].apply(lambda c: "SH" if c.startswith(("6",)) else "SZ")
    df["list_date"] = pd.NaT
    df["delist_date"] = pd.NaT
    return df[["code", "name", "market", "list_date", "delist_date"]]


def main():
    delisted = load_delisted()
    current = load_current()

    # 只保留沪深A股
    delisted_a = delisted[delisted["code"].apply(is_a)].copy()
    current_a = current[current["code"].apply(is_a)].copy()

    # 检查：现存清单里是否混入已退市股（理论不应有重叠）
    overlap = set(current_a["code"]) & set(delisted_a["code"])
    print("现存清单与退市清单重叠的代码:", overlap if overlap else "无")

    delisted_a["status"] = "delisted"
    current_a["status"] = "current"

    uni = pd.concat([current_a, delisted_a], ignore_index=True)
    uni = uni.drop_duplicates(subset="code", keep="first")
    uni = uni.sort_values("code").reset_index(drop=True)

    # 窗口相关：上市日<=2024-12-31 且 (退市日为空 或 退市日>=2015-01-01)
    uni["_list_le"] = (uni["list_date"] <= "2024-12-31") | uni["list_date"].isna()
    uni["_del_ge"] = uni["delist_date"].isna() | (uni["delist_date"] >= "2015-01-01")

    out_path = "data/universe/universe.csv"
    uni[["code", "name", "market", "list_date", "delist_date", "status"]].to_csv(
        out_path, index=False, encoding="utf-8-sig")

    print(f"\n[宇宙清单已保存] {out_path}")
    print(f"总记录数(沪深A股): {len(uni)}")
    print(f"  现存(now上市): {len(current_a)}")
    print(f"  已退市: {len(delisted_a)}")
    print(f"  (已退市中, B股被排除: {len(delisted)-len(delisted_a)})")

    # 窗口内(2015-2024)应存在的股票数
    in_window = uni[uni["_list_le"] & uni["_del_ge"]]
    print(f"\n2015-2024窗口内存在过的股票总数(粗略, 未含现存股上市日后过滤): {len(in_window)}")

    # 退市年份分布（窗口内退市）
    d = delisted_a[(delisted_a["delist_date"] >= "2015-01-01") & (delisted_a["list_date"] <= "2024-12-31")]
    print("窗口内退市股的退市年份分布:")
    print(d["delist_date"].dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
