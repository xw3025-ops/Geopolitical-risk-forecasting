# Third-party data and software

Check each provider's current terms before any public or commercial use; the notes below are a summary, not
legal advice.

## Data

| Source | Used for | Notes on terms |
|---|---|---|
| GDELT 2.0 (GDELT Project) | all model inputs and targets | open data; cite the GDELT Project |
| ACLED (Armed Conflict Location & Event Data) | recent violence, event map | registered access; terms restrict redistribution of event data — the public data bundle contains ACLED events, so confirm the terms before the site is public |
| IMF International Merchandise Trade Statistics | China's trade with each country | IMF data terms; attribute the IMF |
| IMF PortWatch (IMF and University of Oxford) | shipping transits through chokepoints | attribute IMF PortWatch |
| UN Comtrade | oil and gas composition of China's 2024 imports | UN Comtrade terms of use |
| EIA, IEA | Hormuz bypass parameters (compiled by hand) | cited in `pipeline/data/hormuz/hormuz_dependence.csv` |
| The Guardian Open Platform, NYT Article Search | news cited in the briefs (title, date, link) | API terms; only links and short snippets are shown |
| DeepSeek API | writing the bilingual briefs | API terms; briefs are labelled as machine-written |
| Natural Earth (via world-atlas) | map boundaries | public domain |

## Software

| Software | Licence | Used for |
|---|---|---|
| Three.js | MIT | 3D map |
| D3.js | ISC | projection, charts, 2D map |
| topojson-client | ISC | map boundaries |
| Inter, Michroma, JetBrains Mono (Google Fonts) | SIL Open Font License | typography |
| Keras, TensorFlow | Apache 2.0 | models |
| PyTorch | BSD-3-Clause | model training (GPU) |
| NumPy, pandas, scikit-learn | BSD-3-Clause | data processing, benchmarks, calibration |
| LightGBM | MIT | benchmarks |
| Google Cloud client libraries, functions-framework | Apache 2.0 | BigQuery, Cloud Storage, Cloud Functions |
| requests | Apache 2.0 | API calls |
| python-docx | MIT | paper documents |
