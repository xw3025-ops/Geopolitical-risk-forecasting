"""
Pull Guardian articles tagged with any of the 20 MENA countries (headline,
summary, publication date, keyword tags) via the Guardian Open Platform API,
for the news-text experiment. Key from the git-ignored .env (GUARDIAN_API_KEY).
About 250 API calls; well inside the free key's daily limit.

    python fetch_guardian.py
Output:
    raw data/guardian_articles.parquet  (one row per article, deduplicated)
"""

import time

import pandas as pd
import requests
from dotenv import dotenv_values

API = "https://content.guardianapis.com/search"
OUT = "raw data/guardian_articles.parquet"
START, END = "2015-02-16", "2026-08-30"
COUNTRY_TAG = {
    "ARE": "world/united-arab-emirates", "EGY": "world/egypt", "IRN": "world/iran", "IRQ": "world/iraq",
    "ISR": "world/israel", "JOR": "world/jordan", "LBY": "world/libya", "PSE": "world/palestinian-territories",
    "SAU": "world/saudiarabia", "SDN": "world/sudan", "SYR": "world/syria", "TUR": "world/turkey",
    "YEM": "world/yemen", "BHR": "world/bahrain", "CYP": "world/cyprus", "KWT": "world/kuwait",
    "LBN": "world/lebanon", "OMN": "world/oman", "QAT": "world/qatar", "DJI": "world/djibouti",
}
TAG_COUNTRY = {v: k for k, v in COUNTRY_TAG.items()}


def main():
    key = dotenv_values(".env")["GUARDIAN_API_KEY"]
    rows, calls = {}, 0
    for iso, tag in COUNTRY_TAG.items():
        page, pages = 1, 1
        while page <= pages:
            for attempt in range(5):
                r = requests.get(API, timeout=120, params={
                    "tag": tag, "from-date": START, "to-date": END, "page-size": 200, "page": page,
                    "order-by": "oldest", "show-fields": "headline,trailText", "show-tags": "keyword", "api-key": key})
                calls += 1
                if r.status_code == 429:  # rate limited: back off
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            resp = r.json()["response"]
            pages = resp["pages"]
            for a in resp["results"]:
                f = a.get("fields", {})
                countries = sorted({TAG_COUNTRY[t["id"]] for t in a.get("tags", []) if t["id"] in TAG_COUNTRY})
                rows[a["id"]] = dict(id=a["id"], date=a["webPublicationDate"][:10], section=a.get("sectionId"),
                                     headline=f.get("headline", ""), summary=f.get("trailText", ""),
                                     countries="|".join(countries))
            page += 1
            time.sleep(0.25)
        print(f"{iso}: done ({pages} pages), {len(rows)} unique articles so far, {calls} calls", flush=True)
    df = pd.DataFrame(rows.values()).sort_values("date").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"saved {len(df)} articles to {OUT} ({calls} API calls)")


if __name__ == "__main__":
    main()
