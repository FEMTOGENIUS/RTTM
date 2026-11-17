# Cross-check: what the documents promise vs. what this code delivers

Sources checked: research paper (`rttm_research_paper1.docx` / `researchpaper_raj.pdf`, identical text),
project report (`project_report_final_1_fonal.pdf`, 72 pp.) and presentation
(`Real-Time_Topic_Modeling_for_News_Articles.pptx`).

Legend: ✅ implemented & tested here · 🟡 implemented, verify on your machine (needs libraries/internet
not available in the build sandbox) · ➖ declared *future work* in your own documents (intentionally not built)

## A. Data ingestion

| # | Promise (where) | Status | Where in code |
|---|---|---|---|
| 1 | Real-time fetching from NewsAPI.org (paper §Objective, report 3.1, slides) | ✅ | `data/fetchers.py::fetch_newsapi` (keyword/date/language, `/everything` + `/top-headlines`) |
| 2 | GNews API (report 2.6, slides) | ✅ | `fetch_gnews` |
| 3 | Google RSS (report 2.6) | ✅ | `fetch_google_news` - also keeps the up-to-5 related headlines Google bundles per story |
| 4 | RSS feeds via feedparser: BBC, CNN, Al Jazeera, Indian Express, Reuters (report 4.1 B) | ✅ | `fetch_rss` + `config.RSS_FEEDS`. Reuters has no public RSS any more; Reuters stories arrive through Google News |
| 5 | Filter by time, language, keywords (paper Scope) | ✅ | sidebar query / language / region / time window |
| 6 | Scheduled fetching every 5–10 min, `schedule` module (report 3.1, 3.6 C, 4.1 B) | ✅ | `pipeline/scheduler.py` (+ dashboard auto-refresh) |
| 7 | JSON parsing & standardisation | ✅ | `normalize_article` |
| 8 | De-duplication with headline / URL hashes (report 3.4, 3.2) | ✅ | `article_id`, `ArticleStore.add`, preprocessing |
| 9 | Caching to avoid repeated API calls, multiple keys with rotation (report 3.2) | ✅ | persistent store; comma-separated keys rotated on 401/429 |
| 10 | Newspaper3k full-text extraction (report 3.6 C) | 🟡 | `extract_full_text` (optional checkbox) |
| 11 | Store timestamps, source, raw text; session/temp storage, periodic flush (report 3.4 F) | ✅ | `data/store.py` (JSONL, prune after 14 days / 20k items) |
| 12 | 20 Newsgroups for offline validation (slides 8) | 🟡 | `python evaluate.py --source 20newsgroups` (downloads the dataset once) |

## B. Preprocessing

| # | Promise | Status | Where |
|---|---|---|---|
| 13 | Lower-casing; removal of HTML, URLs, punctuation, digits, special characters | ✅ | `utils/preprocess.py::clean_text` |
| 14 | Tokenisation, lemmatisation with spaCy / NLTK | ✅ / 🟡 | spaCy → NLTK WordNet → rule-based (auto) |
| 15 | Standard + custom / domain-specific stop-words | ✅ | sklearn + NLTK + 107 news stop-words |
| 16 | Language detection (langdetect), English-only filter | ✅ | `detect_language` (langdetect or built-in heuristic) |
| 17 | Filter short / redundant articles | ✅ | `preprocess_articles` stats shown in System tab |
| 18 | Embedding preparation (Sentence-Transformers) | 🟡 | `embed_texts` (+ cache) |

## C. Topic modeling

| # | Promise | Status | Where |
|---|---|---|---|
| 19 | LDA with gensim | 🟡 | `models/lda_model.py` (scikit-learn LDA used automatically if gensim is absent - tested) |
| 20 | LDA on a TF-IDF document-term matrix (paper) | ✅ | option "LDA input: tfidf" (default is bag-of-words, the statistically standard input - see D-4) |
| 21 | BERTopic: all-MiniLM-L6-v2 → UMAP → HDBSCAN → c-TF-IDF | 🟡 | `models/bertopic_model.py::_fit_full`; `tests/test_full_stack.py` verifies it on your PC |
| 22 | Same four-stage pipeline usable without heavy installs | ✅ | BERTopic-lite (LSA embeddings/PCA/HDBSCAN/own c-TF-IDF) |
| 23 | Number-of-topics tuning, dynamic topic reduction | ✅ | sidebar slider, `nr_topics`, c-TF-IDF merging |
| 24 | Dominant topic per article, top keywords, topic distribution | ✅ | `TopicResult`, `doc_frame` |
| 25 | Toggle / side-by-side comparison of LDA and BERTopic | ✅ | sidebar Model radio, Compare Models tab (+ ARI/NMI, topic alignment heat-map) |
| 26 | Fallback: if BERTopic fails → LDA (report 3.6 H) | ✅ | `engine.run_pipeline` (tested) |
| 27 | Topic tracking over time; trending, fading topics, bursts (report 2.6, 5.1) | ✅ | `utils/trends.py`, Trends tab, alerts |

## D. Dashboard / visualisation

| # | Promise | Status |
|---|---|---|
| 28 | Streamlit dashboard, `streamlit run main.py` | ✅ (also `project.py`, the name used in the report screenshots) |
| 29 | Word-frequency bar chart, topic-distribution pie | ✅ Overview |
| 30 | Word cloud per topic | ✅ Topics (wordcloud package, Plotly fallback) |
| 31 | Time-series / timeline of topic evolution | ✅ Trends |
| 32 | Drill-down to articles of each topic, "show articles" toggle | ✅ Topics |
| 33 | Filters: keyword, language, date range, topic number, source, time window | ✅ sidebar + Articles tab |
| 34 | Keyword topic search | ✅ Topics |
| 35 | Intertopic distance map (pyLDAvis) | ✅ own map for both models; 🟡 embedded pyLDAvis when installed |
| 36 | UMAP/HDBSCAN cluster visualisation | ✅ document map |
| 37 | Live feed of clustered articles | ✅ Overview |
| 38 | Light/dark toggle, minimal pastel UI | ✅ |
| 39 | Real-time refresh or manual refresh; model re-run button | ✅ |
| 40 | Toggle static vs. real-time streaming mode | ✅ Data mode |
| 41 | Export CSV / JSON (user feedback, report 4.2 F) | ✅ |
| 42 | Notifications for trending topics (feedback / future work) | ✅ in-app banners + webhook in scheduler (bonus) |

## E. Evaluation

| # | Promise | Status |
|---|---|---|
| 43 | Coherence score via gensim `CoherenceModel` | ✅ C_V (gensim when installed, identical built-in implementation otherwise) + NPMI + UMass |
| 44 | Topic diversity | ✅ unique words in top-25 (Dieng et al.) |
| 45 | Human interpretability review | ✅ Evaluate tab → percentages in Compare Models |
| 46 | Runtime / responsiveness (ingestion → dashboard latency) | ✅ System tab, `evaluate.py` |
| 47 | Usability feedback from test users | ✅ Evaluate tab |
| 48 | Stress tests: 500+ article burst, blank fields, API failures | ✅ `evaluate.py --stress`, tests |

## F. Engineering

| # | Promise | Status |
|---|---|---|
| 49 | Modular structure `data/ models/ utils/ dashboard/ main.py requirements.txt` (report 3.6 B) | ✅ |
| 50 | Logging to `logs/` with INFO / WARNING / ERROR; try/except for API, model, UI failures | ✅ |
| 51 | Unit, integration and UI tests | ✅ 48 tests (44 run here; 4 need gensim/BERTopic/FastAPI) |
| 52 | API keys in `.env`, excluded from version control | ✅ `.env.example`, `.gitignore` |
| 53 | Docker, Streamlit Cloud / Heroku / Render ready | ✅ `Dockerfile`, `docker-compose.yml`, `Procfile` |
| 54 | Optional FastAPI endpoints | 🟡 `api.py` |
| 55 | Tested on Windows and Linux | 🟡 Linux tested here; `run.bat` provided for Windows |

**Future work in your own documents, intentionally not built:** sentiment / stance analysis, multilingual
models (XLM-R), credibility scoring, user accounts, external database (Supabase/Firebase), WebSockets,
Kubernetes, GPT summarisation. (The paper's *Limitations* section explicitly excludes sentiment, so adding
it would contradict the paper.)

---

## Validation performed while building

1. **Real headlines.** 68 real Google News India headlines (17 Nov 2026), grouped by Google into 13 stories
   (Quad summit, Cyclone Akvan, Bilkis Bano case, Sensex/Nifty, Nvidia chip curbs, Amazon renewables …).
   Measured against Google's own story grouping:

   | | BERTopic (lite back-end) | LDA (scikit-learn) |
   |---|---|---|
   | ARI vs Google stories | **0.855** | 0.363 |
   | NMI vs Google stories | **0.953** | 0.700 |
   | Coherence C_V | **0.622** | 0.503 |
   | Topic diversity (top-25) | **99 %** | 37 % |

   This supports the paper's qualitative claim (BERTopic is better on short news text). It is a
   small sanity check, not a paper result. The full MiniLM back-end could not run in the build sandbox.
2. **Synthetic demo set** (221 articles after cleaning, 9 known topics, a built-in burst and a fading topic):
   BERTopic-lite ARI = 1.00, LDA ARI = 0.79 ± 0.05; the cyclone burst is flagged *Emerging* and the film
   topic *Fading*. The stress test (520 articles incl. blank/malformed records) completes in ≈ 0.5 s with no
   crash.
3. **Google News RSS structure** was checked against the live feed and reproduced in a test fixture.
4. **Headless dashboard test**: every mode and tab renders without errors.

---

## Things in the documents that need care before submission

1. **Result numbers have no code behind them yet.** The report's C_V 0.51 / 0.67, diversity 87 %,
   interpretability 70 % / 90 %, stage latencies (2.1 / 4.5 / 6.3 / 1.8 s), error rates, "10 participants",
   "3,000+ articles" and "100 % uptime" are not produced by any script in the original material.
   Re-run `evaluate.py` and the Evaluate tab, then replace them with your measured values and state the
   dataset and date. (Interestingly, the real-headline check above gave C_V 0.50 vs 0.62, close to the
   paper's numbers.)
2. **"Real-time" with NewsAPI.** The free NewsAPI plan delays articles by 24 hours and allows 100
   requests/day (dev use only). Say so, or describe Google News RSS / RSS as the real-time source (the
   default here).
3. **BERTopic "updates without retraining"** (report 5.1). Standard BERTopic is refitted on each refresh.
   This system makes refreshes cheap by caching embeddings, but it does not update incrementally.
   Reword it, e.g. "refits on a sliding window with cached embeddings".
4. **LDA on TF-IDF** (paper). LDA is defined on word counts. Both options exist; the default is
   bag-of-words. Mention which one your reported numbers use.
5. **BERTopic output samples** in the paper ("Ukraine conflict, Russian invasion, NATO response") are
   human-written labels. BERTopic outputs keyword lists. Present them as "labelled from keywords".
6. **Slide 10** still contains the placeholder note *"The original script references four external
   screenshots; they were not included …"*. Replace it with screenshots of this dashboard.
7. **"Using Big Data"** appears in the title, but no big-data technology is described. The system handles
   thousands of articles on a single machine. Either drop the phrase or justify it (volume and velocity of
   streaming news).
8. Small typos: `diwakarji701@gmai.com` (gmai → gmail?) in the paper's author block; "he developed system"
   (→ "The developed system") at the start of the paper's *Result* section; report file name "fonal".
