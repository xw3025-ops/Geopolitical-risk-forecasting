import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["COMTRADE_KEY"]
URL = "https://comtradeapi.un.org/data/v1/get/C/M/HS"  # C=商品, M=月度, HS=分类
OUT = Path("data/raw/comtrade")
OUT.mkdir(parents=True, exist_ok=True)


def pull(periods, reporter=156, flows="X,M"):
    """reporter=156 是中国；periods 例如 ['202501','202502',...]"""
    for attempt in range(4):
        r = requests.get(
            URL,
            headers={"Ocp-Apim-Subscription-Key": KEY},  # key 放请求头，不出现在 URL
            params={
                "reporterCode": reporter,
                "period": ",".join(periods),
                "cmdCode": "TOTAL",
                "flowCode": flows,
                "partner2Code": 0,
                "customsCode": "C00",
                "motCode": 0,  # 取汇总口径，避免重复行
            },
            timeout=120,
        )
        if r.status_code == 429:  # 被限速，等一会再试
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return pd.DataFrame(r.json().get("data", []))
    raise RuntimeError("多次被限速，稍后再试")


def main(start_year=2019, end_year=2026):
    for y in range(start_year, end_year + 1):
        f = OUT / f"china_monthly_{y}.parquet"
        if f.exists():  # 断点续跑：已下载的年份跳过
            continue
        months = [f"{y}{m:02d}" for m in range(1, 13)]
        df = pull(months)
        print(y, len(df), "rows")
        df.to_parquet(f)
        time.sleep(2)


if __name__ == "__main__":
    main()
