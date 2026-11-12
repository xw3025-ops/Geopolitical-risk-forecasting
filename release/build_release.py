"""
Package the production release of the MENA Risk Monitor as one archive for review or redeployment.

    python release/build_release.py [--version v2.2] [--no-screenshots]

-> dist/mena-risk-monitor-<version>-<YYYYMMDD>/ and the matching .zip (dist/ is git-ignored).

The package holds the deployed website, the production pipeline with its model files (repository layout, so the
deploy scripts run from pipeline/), a snapshot of the live data, the papers and screenshots, and the documents in
release/ (README, DEPLOYMENT, THIRD_PARTY) plus generated RELEASE_NOTES and MANIFEST.sha256. Files are taken from
the working tree; the release notes record the commit and any uncommitted changes to packaged files. The build
stops if a file that looks like a credential would be included.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

SITE_URL = "https://mena-risk-monitor.vercel.app"
DATA_URL = "https://storage.googleapis.com/geopolitical-risk-forecasting-site/site/latest.json"

# what runs in production (repository paths); copied to pipeline/ with the same layout
PIPELINE = [
    "build_site_data.py", "live_pipeline.py", "honest_common.py", "model_architectures.py", "risk_index.py",
    "target_variants.py", "run_pipeline.py", "verify_live_pipeline.py", "requirements.txt", ".env.example",
    "cloud_function/main.py", "cloud_function/gdelt_refresh.py", "cloud_function/requirements.txt",
    "cloud_function/deploy.sh", "cloud_function/set_secrets.sh", "cloud_function/setup_site.sh",
    "cloud_function/trade_refresh.py", "cloud_function/deploy_trade.sh", "cloud_function/trade_requirements.txt",
    "src/config.py", "src/trade_sources.py", "src/fetch_imf_imts.py", "src/fetch_portwatch.py",
    "data/hormuz/hormuz_dependence.csv", "paper/v22_stats.json", "paper/linear_baselines.json",
    *[f"model_release_v2.2_{m}_s{s}.weights.h5" for m in ("goldstein", "surge") for s in (42, 1, 2, 3, 4)],
    "artifacts/goldstein_v2.2_preprocessing.joblib", "artifacts/surge_v2.2_preprocessing.joblib",
    "artifacts/risk_v1_calibration.json",
    "artifacts/goldstein_v2.2_test_predictions.npz", "artifacts/surge_v2.2_test_predictions.npz",
]
DOCS = {"paper/risk_paper.pdf": "risk_paper.pdf", "paper/risk_paper_zh.pdf": "risk_paper_zh.pdf"}
SCREENS = [("en", "overview", "/"), ("en", "country_yemen", "/#/c/YEM"), ("en", "data_and_model", "/#/about"),
           ("zh", "overview_zh", "/"), ("zh", "country_yemen_zh", "/#/c/YEM")]
SECRET = re.compile(r"(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY|VERCEL_OIDC_TOKEN=\S+|"
                    r"^(ACLED_PASSWORD|DEEPSEEK_API_KEY|GUARDIAN_API_KEY|NYT_API_KEY)=\S+)", re.M)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def data_snapshot(out, frozen):
    """The data the production site shows: its own shipped copy when frozen, else the live bundle."""
    raw = open("website/data/latest.json", "rb").read() if frozen else urllib.request.urlopen(DATA_URL, timeout=60).read()
    D = json.loads(raw)
    m, stamp = D["meta"], D["meta"]["as_of_week"]
    os.makedirs(out, exist_ok=True)
    open(f"{out}/mena_risk_bundle_{stamp}.json", "wb").write(raw)
    band = lambda v: "severe" if v >= 55 else "high" if v >= 45 else "elevated" if v >= 32 else "low"

    def write(name, rows):  # same columns as the site's Export menu; UTF-8 with BOM so Excel shows Chinese names
        with open(f"{out}/{name}_{stamp}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("country_risk", [dict(
        as_of_week=stamp, model=m["model"], iso3=c["iso3"], name=c["name"], name_zh=c["name_zh"], rank=c["rank"],
        risk=c["risk"], risk_previous_refresh=c["risk_prev"], risk_change=c["risk_change"], band=band(c["risk"]),
        severity=c["severity"], forecast_hostility=c["severity_parts"]["hostility"],
        forecast_surge_risk=c["severity_parts"]["surge"], recent_violence=c["severity_parts"]["violence"],
        trade_exposure=c["exposure"], china_trade_bn_12m=c["trade"]["total_bn"],
        china_exports_bn=c["trade"]["exports_bn"], china_imports_bn=c["trade"]["imports_bn"],
        trade_yoy=c["trade"]["yoy"], goldstein_forecast_13w=c["forecast"], goldstein_normal_52w=c["normal"],
        forecast_vs_normal=c["change"], surge_alerts=c["alerts"], acled_deaths_4w=c["acled_fatalities_4w"],
        hormuz_dependence_low=c["trade"]["hormuz_low"], hormuz_dependence_high=c["trade"]["hormuz_high"])
        for c in D["countries"]])
    write("relationship_forecasts", [dict(
        as_of_week=stamp, actor1=x["a"], actor2=x["b"], goldstein_forecast_13w=x["f"], goldstein_normal_52w=x["n"],
        forecast_vs_normal=round(x["f"] - x["n"], 3), surge_probability=x["s"], surge_alert=x["al"],
        events_per_week_52w=x["v"]) for x in D["dyads"]])
    write("shipping_lanes", [dict(
        lane=l["name"], lane_zh=l["name_zh"], weeks=l["weeks"], status=l["status"], ships_per_week=l["transits_week"],
        normal_ships_per_week=l["normal_week"], change_vs_normal=l["change"], tankers_per_week=l["tankers_week"],
        normal_tankers_per_week=l["normal_tankers"], china_trade_at_stake_bn=l.get("china_trade_at_stake_bn"),
        littoral_states=" ".join(x["iso3"] for x in l["littoral"]), highest_littoral_risk=l["littoral_risk"])
        for l in D["lanes"]])
    events = [dict(country=c["iso3"], date=e["d"], type=e["t"], sub_type=e["s"], actor=e["a"], location=e["l"],
                   latitude=e["lat"], longitude=e["lon"], deaths=e["f"], notes=e["n"])
              for c in D["countries"] for e in ((c.get("acled") or {}).get("events") or [])]
    if events:
        write("acled_events", events)
    return m


def screenshots(out):
    chrome = next((p for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                               "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                               shutil.which("google-chrome") or "", shutil.which("chromium") or ""]
                   if p and os.path.exists(p)), None)
    if not chrome:
        print("no Chrome/Edge found: screenshots skipped")
        return
    os.makedirs(out, exist_ok=True)
    tmp = os.path.abspath(os.path.join(os.path.dirname(out), "_chrome"))  # Chrome needs an absolute profile path
    for lang, name, path in SCREENS:
        subprocess.run([chrome, "--headless=new", "--disable-gpu", f"--lang={'zh-CN' if lang == 'zh' else 'en-US'}",
                        f"--user-data-dir={tmp}_{lang}", "--window-size=1600,900", "--virtual-time-budget=12000",
                        f"--screenshot={os.path.abspath(f'{out}/{name}.png')}", SITE_URL + path],
                       capture_output=True, timeout=120)
    for lang in ("en", "zh"):
        shutil.rmtree(f"{tmp}_{lang}", ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2.2")
    ap.add_argument("--no-screenshots", action="store_true")
    args = ap.parse_args()
    today = datetime.now(timezone.utc)
    name = f"mena-risk-monitor-{args.version}-{today:%Y%m%d}"
    root = os.path.join("dist", name)
    if os.path.exists(root):
        shutil.rmtree(root)

    # 1. website, exactly as deployed (no local Vercel link or token)
    shutil.copytree("website", f"{root}/site", ignore=shutil.ignore_patterns(".env*", ".vercel", "__pycache__",
                                                                            ".gitignore"))
    # 2. production pipeline and model files
    for f in PIPELINE:
        copy(f, f"{root}/pipeline/{f}")
    # 3. the data the site shows (frozen snapshot, or the live bundle)
    frozen = bool(re.search(r"^\s*frozen:\s*true", open("website/config.js", encoding="utf-8").read(), re.M))
    meta = data_snapshot(f"{root}/data_snapshot", frozen)
    # 4. papers and screenshots
    for src, dst in DOCS.items():
        copy(src, f"{root}/docs/{dst}")
    if not args.no_screenshots:
        screenshots(f"{root}/docs/screenshots")

    # 5. documents
    commit = git("rev-parse", "--short", "HEAD")
    gen = meta["generated_utc"][:16].replace("T", " ") + " UTC"
    state = {
        "{{DATA_LINE}}": (f"frozen snapshot shipped with the site (`site/data/latest.json`), refresh of {gen}; the "
                          "automated refresh is paused (DEPLOYMENT.md, section 0)") if frozen
        else f"live bundle, read by the site on every visit: {DATA_URL}",
        "{{REFRESH}}": "paused (designed: forecasts every 2 weeks; scores, trade, shipping and briefs weekly)" if frozen
        else "forecasts every 2 weeks; scores, trade, shipping and briefs weekly; fully automated",
        "{{STATUS}}": "frozen demonstration snapshot; scheduled refresh paused" if frozen else "live, refreshed weekly",
        "{{STATE}}": open("release/STATE_FROZEN.md" if frozen else "release/STATE_LIVE.md", encoding="utf-8").read(),
    }
    fill0 = lambda t: (t.replace("{{VERSION}}", args.version).replace("{{DATE}}", f"{today:%Y-%m-%d}")
                       .replace("{{COMMIT}}", commit).replace("{{AS_OF}}", meta["as_of_week"])
                       .replace("{{GENERATED}}", meta["generated_utc"][:16].replace("T", " ") + " UTC")
                       .replace("{{SITE_URL}}", SITE_URL).replace("{{DATA_URL}}", DATA_URL))

    def fill(t):
        for k, v in state.items():
            t = t.replace(k, v)
        return fill0(t)
    for doc in ("README.md", "DEPLOYMENT.md", "THIRD_PARTY.md"):
        open(f"{root}/{doc}", "w", encoding="utf-8").write(fill(open(f"release/{doc}", encoding="utf-8").read()))
    packaged = ["website"] + PIPELINE + list(DOCS)
    dirty = [ln for ln in git("status", "--porcelain", "--", *packaged).splitlines() if ln.strip()]
    stats = json.load(open("paper/v22_stats.json"))
    notes = fill(open("release/RELEASE_NOTES.md", encoding="utf-8").read())
    notes = notes.replace("{{R2}}", f"{stats['v2.2']['r2']:.3f}").replace(
        "{{DIRTY}}", "none" if not dirty else f"{len(dirty)} (listed below)").replace(
        "{{DIRTY_LIST}}", "" if not dirty else "Packaged files that differ from the commit:\n\n```\n"
                                                + "\n".join(dirty) + "\n```\n")
    open(f"{root}/RELEASE_NOTES.md", "w", encoding="utf-8").write(notes)

    # 6. no credentials in the package
    hits = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            if f.startswith(".env") and f != ".env.example":
                hits.append(p)
            elif f.endswith((".py", ".sh", ".md", ".txt", ".js", ".html", ".json", ".example", ".csv")):
                if SECRET.search(open(p, encoding="utf-8", errors="ignore").read()):
                    hits.append(p)
    if hits:
        sys.exit(f"possible credentials, not packaged: {hits}")

    # 7. checksums and archive
    files = sorted(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/")
                   for dp, _, fs in os.walk(root) for f in fs)
    with open(f"{root}/MANIFEST.sha256", "w", encoding="utf-8", newline="\n") as fh:
        for f in files:
            fh.write(f"{hashlib.sha256(open(os.path.join(root, f), 'rb').read()).hexdigest()}  {f}\n")
    zpath = f"{root}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + ["MANIFEST.sha256"]:
            z.write(os.path.join(root, f), f"{name}/{f}")
    digest = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    print(f"{zpath}: {os.path.getsize(zpath) / 1e6:.1f} MB, {len(files) + 1} files, sha256 {digest}")
    print(f"commit {commit}; uncommitted packaged files: {len(dirty)}" + ("" if not dirty else " (see RELEASE_NOTES)"))


if __name__ == "__main__":
    main()
