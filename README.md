# 📰 NewsLens — Real-Time Topic Modeling for News Articles

A working implementation of the system described in *“Real-Time Topic Modeling for Streaming News Articles”*
(paper, project report and presentation): live news ingestion → NLP preprocessing → **LDA** and **BERTopic**
topic modeling → interactive **Streamlit + Plotly** dashboard, with evaluation (coherence, diversity,
human interpretability, latency), trend/burst detection, alerts, export, REST API and a background scheduler.

```
 NewsAPI ─┐                                    ┌─► LDA (gensim)  ───────────┐
 GNews  ──┤   ┌──────────┐   ┌─────────────┐   │                            │   ┌────────────┐   ┌────────────────────┐
 Google ──┼──►│ fetchers │──►│ preprocess  │───┤                            ├──►│  metrics   │──►│ Streamlit dashboard│
 News RSS │   │ + store  │   │ clean/lemma │   │                            │   │ labels     │   │ FastAPI  /topics   │
 RSS ─────┤   │ (dedupe) │   │ lang filter │   └─► BERTopic: MiniLM → UMAP ─┘   │ trends     │   │ scheduler + alerts │
 sample / ┘   └──────────┘   └─────────────┘         → HDBSCAN → c-TF-IDF       │ comparison │   │ evaluate.py        │
 upload                                                                         └────────────┘   └────────────────────┘
```

---

## 1. Quick start

**Python 3.10 – 3.12 recommended** (3.13 works with the lite back-ends; some NLP wheels lag behind).

### Windows (PowerShell / cmd)
```bat
run.bat
```
### Linux / macOS
```bash
./run.sh
```
Both scripts create a virtual environment, install `requirements-full.txt` (falling back to the light
`requirements.txt`), download the NLP models
(`scripts/setup_nlp.py`) and start the dashboard at **http://localhost:8501**.

### Manual
```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements-full.txt                   # full paper stack
#   (if that fails, e.g. no C++ build tools for hdbscan:  pip install -r requirements.txt  = light version)
python scripts/setup_nlp.py                            # NLTK data, spaCy model, MiniLM model + health report
copy .env.example .env                                 # optional: add NEWSAPI_KEY / GNEWS_KEY
streamlit run main.py                                  # (streamlit run project.py also works)
```

> **No API key needed.** Google News RSS and publisher RSS feeds work out of the box.
> The **sample dataset** mode works fully offline.

`python scripts/setup_nlp.py --check` shows which back-ends are active, e.g.
`LDA: gensim`, `embeddings: SentenceTransformer (all-MiniLM-L6-v2)`, `dim_reduction: UMAP`.
If a heavy library is missing, the system automatically uses a lighter equivalent and says so
in the dashboard (**System** tab) - it never crashes because of a missing package.

---

## 2. Using the app

NewsLens looks and works like a news website; the topic model runs underneath it.

| Part | What it does |
|---|---|
| **Top bar** | date, edition and LIVE indicator · search box · time window · **Settings** · **Refresh** |
| **Masthead + section menu** | Home · Politics · World · Business · Tech · Sport · Health · Entertainment · **More ▾** (Climate, Law & Justice, Science & Space, Startups, Energy, Other) · **Insights** |
| **Trending ticker** | scrolling list of breaking / trending stories detected by the trend analysis (hover to pause) |
| **Home** | lead story with picture and related coverage, *Top stories* grid, *Most covered* and *Latest* rails, hot keywords, newspaper-style section blocks |
| **Story clusters** | every topic found by BERTopic/LDA becomes a story: lead article + coverage from other publishers + "Full coverage" link |
| **Full coverage page** | all articles of one story, timeline, who is covering it, related stories |
| **Insights** | the analytics for the paper: the tabs below |

Settings (gear pop-over): data mode (Live / Sample / Upload), sources, language, region, articles per source,
streaming + interval, API keys, model (BERTopic / LDA / Compare), number of topics, advanced options, dark mode.

### Insights tabs (analytics)

<details><summary>All settings</summary>

| Control | What it does |
|---|---|
| 🌙 Dark mode | light / dark theme (charts follow) |
| Data mode | **Live (real-time)** · **Static (sample dataset)** · **Static (upload CSV/JSON)** |
| Search query | keyword(s) sent to the sources (empty = top headlines) |
| Sources | Google News RSS, RSS feeds (BBC, Al Jazeera, CNN, Indian Express, Guardian, NPR), NewsAPI, GNews |
| Language / Region | API language and Google News edition |
| Articles per source | 10 – 100 |
| 🔄 Auto-refresh + interval | streams new articles every N seconds; models re-run only when new articles arrive |
| Time window | last 6 h … 7 days / all stored |
| Model | **BERTopic**, **LDA** or **Compare both** |
| Number of topics | LDA *k*; BERTopic `nr_topics` (or let HDBSCAN decide) |
| Advanced | min topic size, outlier re-assignment, embedding model, LDA passes, LDA input (BoW / TF-IDF), English-only filter |
| 📥 Fetch now / ♻️ Re-run models | manual refresh / model recalibration |

</details>

| Tab | Content |
|---|---|
| 📊 Overview | clickable **news-landscape treemap** (category → topic), topic donut, hot-keywords chart, **live feed** cards, sources |
| 🧩 Topics | grid of colour-coded topic cards with trend badges, **topic explorer** (click a topic pill → keywords, **word cloud**, article cards), keyword **topic search** |
| 📈 Trends | **topic evolution over time** (count / share / stacked), animated **topic race** (▶ Play), momentum chart, table of 🆕 emerging · 📈 trending · ➖ stable · 📉 fading topics with burst z-scores |
| 🗺️ Topic Map | **intertopic distance map** (Jensen-Shannon / cosine + MDS, like pyLDAvis) and **document cluster map**; embedded **pyLDAvis** for gensim LDA if installed |
| ⚖️ Compare Models | C_V / NPMI / UMass coherence, topic diversity, outliers, runtime, human-evaluation results, LDA↔BERTopic agreement (ARI, NMI) and topic-alignment heat-map |
| 📄 Articles & Export | filter by source / topic / keyword / date range; **download CSV / JSON** (articles + topics) and topics + metrics JSON |
| ✅ Evaluate | human-interpretability review form (saved to `logs/feedback.jsonl`, aggregated in Compare Models) + usability score |
| 🛠️ System | active back-ends, preprocessing stats, latency per stage, store status, log viewer |

Emerging / bursting topics appear in the TRENDING ticker and as Breaking / Trending labels on stories.

---

## 3. Real-time operation

* The dashboard keeps a rolling **article store** (`data/store/articles.jsonl`), de-duplicated by headline
  hash and URL, pruned after 14 days.
* **Background ingestion** (independent of the browser):
  ```bash
  python -m pipeline.scheduler --interval 10 --query "india" --sources google_news rss
  ```
  Fetches every 10 minutes, re-runs BERTopic, logs alerts and (optionally) pushes them to a
  Slack/Teams/Discord webhook (`ALERT_WEBHOOK_URL` in `.env`).
* Docker: `docker compose up` starts the dashboard **and** the scheduler sharing one store.
* Sentence embeddings are cached per article, so each refresh only embeds *new* articles.

**API limits to know:** NewsAPI's free *Developer* plan serves articles with a **24-hour delay**,
100 requests/day and is for development only; GNews free has similar limits. For genuinely live data use
Google News RSS / RSS feeds (default) or a paid plan. With NewsAPI enabled keep the scheduler interval
≥ 15 minutes. Several keys can be given comma-separated - they are rotated automatically on rate limits.

---

## 4. Reproducing the evaluation for the paper / report

```bash
python evaluate.py --source live --runs 3                    # live Google News + RSS
python evaluate.py --source store --hours 72 --runs 3        # what the dashboard/scheduler collected
python evaluate.py --source 20newsgroups --max-docs 2000     # offline benchmark (downloads once)
python evaluate.py --source sample --stress                  # synthetic demo data + 520-article stress test
```
Each run writes `results/eval_<time>.md` (ready-to-paste table) and `.json` with:
coherence C_V / NPMI / UMass, topic diversity, #topics, raw HDBSCAN outlier rate, model runtime,
mean latency per pipeline stage, ARI / NMI against ground truth where labels exist (20 Newsgroups,
sample), LDA↔BERTopic agreement and the top words of every topic. Use `--bertopic-auto` to let HDBSCAN
choose the number of topics.

Human interpretability: open the **Evaluate** tab, let each evaluator rate the topics, then read the
percentages in **Compare Models**.

> ⚠️ The bundled sample dataset is **synthetic** (template-generated, fictional publishers) and exists for
> offline demos and tests. Do **not** report its scores as research results.

---

## 5. Running from GitHub

Push this folder to a GitHub repository (`git init && git add . && git commit -m "init" && git push`), then:

| Option | What it gives you | How |
|---|---|---|
| **GitHub Codespaces** | the full stack running in your browser (cloud VM) | repo page → **Code → Codespaces → Create codespace**. `.devcontainer/` installs everything and starts the dashboard (first start ≈ 10 min) |
| **GitHub Actions** | automatic tests on every push | already configured in `.github/workflows/tests.yml` → see the **Actions** tab |
| **Streamlit Community Cloud** | free public URL (`<name>.streamlit.app`) | share.streamlit.io → **Create app** → pick the repo, main file `main.py`, Python 3.11. Add keys under **Settings → Secrets**: `NEWSAPI_KEY = "..."` |

`requirements.txt` is the light version on purpose: Streamlit Community Cloud has a small memory limit that
torch + MiniLM do not fit, so the cloud app runs in lite mode (all features work, lighter NLP back-ends;
NLTK data is downloaded automatically). Codespaces / your PC use `requirements-full.txt`. The article store there is temporary (reset on reboot) and the
background scheduler does not run; use the dashboard's auto-refresh instead. GitHub Pages **cannot** run it
(Pages only serves static files).

## 6. REST API (optional)

```bash
pip install fastapi uvicorn
uvicorn api:app --port 8000          # docs: http://localhost:8000/docs
```
`GET /health` · `POST /fetch?query=` · `GET /articles?hours=24` · `GET /topics?model=BERTopic&hours=48`

---

## 7. Tests

```bash
pip install pytest
pytest -q                                   # 48 tests: fetchers, preprocessing, store, metrics, models,
                                            # engine, trends, scheduler, evaluation, API, headless dashboard
pytest tests/test_full_stack.py -v          # verifies gensim + real BERTopic/MiniLM/UMAP on your machine
```
The dashboard test runs the complete Streamlit app headlessly in every mode (sample, upload, live with
mocked network, all three model settings, dark mode, search, evaluation form) and checks that every chart
references valid data columns.

---

## 8. Project structure

```
├── main.py / project.py      Streamlit entry points
├── config.py                 settings (.env), feeds, paths
├── data/
│   ├── fetchers.py           NewsAPI, GNews, Google News RSS, RSS/Atom, newspaper3k, sample, upload
│   ├── store.py              JSONL article store, de-duplication, pruning
│   └── sample_articles.json  offline synthetic demo set
├── utils/
│   ├── preprocess.py         cleaning, tokenisation, lemmatisation, stop-words, language detection
│   ├── metrics.py            C_V / NPMI / UMass coherence, diversity, timers
│   ├── trends.py             topics over time, emerging/trending/fading, bursts, alerts
│   ├── labels.py             topic category labels
│   └── logging_setup.py      logs/rttm.log (INFO / WARNING / ERROR)
├── models/
│   ├── lda_model.py          gensim LDA (scikit-learn fallback)
│   ├── bertopic_model.py     BERTopic (+ lite fallback), embedding cache
│   ├── ctfidf.py             class-based TF-IDF, topic merging
│   └── engine.py             end-to-end pipeline, fallback BERTopic→LDA, comparison, intertopic map
├── dashboard/
│   ├── app.py                Streamlit UI
│   └── charts.py             Plotly figures, word clouds
├── pipeline/scheduler.py     background ingestion + alert webhook
├── api.py                    FastAPI service
├── evaluate.py               reproducible LDA vs BERTopic evaluation + stress test
├── scripts/                  setup_nlp.py (models + health check), make_sample_data.py
├── tests/                    pytest suite + fixtures
├── .github/workflows/      CI tests;  .devcontainer/  GitHub Codespaces
├── Dockerfile, docker-compose.yml, Procfile, run.bat, run.sh
└── requirements*.txt         light (default, Streamlit Cloud) / full (paper stack) / optional
```

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install -r requirements-full.txt` fails on `hdbscan` / `gensim` | use Python 3.11; on Windows install "Microsoft C++ Build Tools", or `pip install -r requirements.txt` (light back-ends) |
| First BERTopic run is slow | the MiniLM model (~90 MB) and torch load once; later refreshes reuse cached embeddings |
| "BERTopic failed … falling back to LDA" | fewer than 10 articles or everything was an outlier - widen the time window / fetch more |
| "⚠️ source(s) unavailable" (HTTP 429 / 403) | the news site rate-limited (429) or blocked (403) this server. Cloud machines (Codespaces, Streamlit Cloud) share IPs and are limited more often; the other sources keep working, try again in a few minutes |
| NewsAPI returns nothing recent | free plan = 24 h delay; use Google News RSS |
| Dark mode only partly applied | some Streamlit versions ignore runtime theme changes; use ☰ → Settings → Theme |
