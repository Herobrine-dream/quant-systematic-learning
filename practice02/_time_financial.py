# -*- coding: utf-8 -*-
"""临时：测量年报版财报接口单次耗时。"""
import time
import akshare as ak

def time_one(name, fn):
    t0 = time.time()
    df = fn()
    dt = time.time() - t0
    print(f"{name}: {len(df)} rows x {len(df.columns)} cols, {dt:.1f}s", flush=True)
    return dt

t1 = time_one("profit_yearly", lambda: ak.stock_profit_sheet_by_yearly_em(symbol="SH600519"))
t2 = time_one("balance_yearly", lambda: ak.stock_balance_sheet_by_yearly_em(symbol="SH600519"))
t3 = time_one("cashflow_yearly", lambda: ak.stock_cash_flow_sheet_by_yearly_em(symbol="SH600519"))
print(f"单只股票三表合计: {t1+t2+t3:.1f}s", flush=True)
