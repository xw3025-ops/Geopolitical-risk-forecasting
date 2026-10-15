"""Download IMF PortWatch daily chokepoint transits (public, no API key).

Chokepoints: Strait of Hormuz, Bab el-Mandeb, Suez Canal, Cape of Good Hope, Malacca Strait.
PortWatch's daily series starts 2019-01-01, so it cannot reach back to the conflict data's
start (weekly GDELT 2014-12, monthly GDELT 2017-01); it covers the trade transformer's whole
training window (2019-04 onward).

Outputs (data/portwatch/):
  chokepoints_daily.csv    date x chokepoint: vessel counts and capacity (metric tons) by type
  chokepoints_weekly.csv   Monday weeks (complete weeks only), same calendar as the GDELT panel
Run:  python src/fetch_portwatch.py
"""
from config import ROOT
from trade_sources import portwatch, portwatch_weekly

OUT = ROOT / "data" / "portwatch"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily = portwatch()
    daily.to_csv(OUT / "chokepoints_daily.csv", index=False)
    weekly = portwatch_weekly(daily)
    weekly.to_csv(OUT / "chokepoints_weekly.csv", index=False)
    print(f"{len(daily):,} daily rows -> {OUT / 'chokepoints_daily.csv'}")
    print(daily.groupby("chokepoint").date.agg(["min", "max", "count"]))
    h = weekly[weekly.chokepoint == "hormuz"].set_index("week_start")
    print("\nHormuz tanker transits per week, last 12 complete weeks:")
    print(h.n_tanker.tail(12).to_string())


if __name__ == "__main__":
    main()
