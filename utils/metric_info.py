"""Plain-language explanations and quality bands for every metric shown in the UI."""
from __future__ import annotations

import math

# key: (display name, what it means, how to read it, higher_is_better, bands[(threshold, label)] ascending)
METRICS: dict[str, dict] = {
    "coherence_cv": dict(
        name="Coherence C_V", scale="0 – 1", better="higher",
        means="How often a topic's top words appear together in the same articles (Röder et al., 2015). "
              "Coherent topics are easier for people to understand.",
        read="< 0.40 weak · 0.40–0.55 fair · 0.55–0.70 good · > 0.70 very strong (common on small or clean data)",
        bands=[(0.40, "Weak"), (0.55, "Fair"), (0.70, "Good"), (math.inf, "Very strong")]),
    "coherence_npmi": dict(
        name="Coherence NPMI", scale="−1 – 1", better="higher",
        means="Normalised pointwise mutual information between pairs of top words: do they co-occur more "
              "often than chance?",
        read="< 0 unrelated words · 0–0.10 weak · 0.10–0.20 good · > 0.20 strong",
        bands=[(0.0, "Unrelated"), (0.10, "Weak"), (0.20, "Good"), (math.inf, "Strong")]),
    "coherence_umass": dict(
        name="Coherence UMass", scale="≤ 0", better="closer to 0",
        means="Document-level co-occurrence of top words (Mimno et al., 2011). Always negative; values "
              "closer to 0 are better.",
        read="< −3 weak · −3 to −1.5 fair · > −1.5 good",
        bands=[(-3.0, "Weak"), (-1.5, "Fair"), (math.inf, "Good")]),
    "diversity": dict(
        name="Topic diversity", scale="0 – 100 %", better="higher",
        means="Share of unique words among the top-25 words of all topics (Dieng et al., 2020). High = topics "
              "don't repeat each other. It does NOT measure whether a topic makes sense - read it together "
              "with coherence.",
        read="< 50 % topics overlap a lot · 50–80 % fair · > 80 % distinct topics",
        bands=[(0.5, "Overlapping"), (0.8, "Fair"), (math.inf, "Distinct")]),
    "n_topics": dict(
        name="Topics found", scale="count", better="—",
        means="Number of topics (story clusters). LDA uses the number you choose; BERTopic can choose it "
              "itself with HDBSCAN.", read="More is not better - it depends on how much news there is.",
        bands=None),
    "outlier_rate_raw": dict(
        name="Raw outliers", scale="0 – 100 %", better="lower",
        means="Share of articles HDBSCAN could not place in any topic (BERTopic only). They are then "
              "re-assigned to the nearest topic so every article appears on the site.",
        read="< 15 % low · 15–35 % moderate · > 35 % high (short, noisy text increases it)",
        bands=None),
    "runtime_s": dict(
        name="Model runtime", scale="seconds", better="lower",
        means="Time to fit the model on the current articles (excluding fetching).",
        read="Matters for real time: the whole pipeline should finish well within the refresh interval.",
        bands=None),
    "ari": dict(
        name="Adjusted Rand Index (ARI)", scale="−1 – 1", better="—",
        means="How similarly the two models group the same articles. 1 = identical grouping, 0 = no more "
              "similar than random.", read="> 0.6 strong agreement · 0.3–0.6 partial · < 0.3 different views",
        bands=None),
    "nmi": dict(
        name="Normalised Mutual Information (NMI)", scale="0 – 1", better="—",
        means="Shared information between the two groupings; less sensitive to different numbers of topics "
              "than ARI.", read="> 0.7 strong agreement · 0.4–0.7 partial · < 0.4 different views",
        bands=None),
    "jaccard": dict(
        name="Topic alignment (Jaccard)", scale="0 – 1", better="—",
        means="Overlap of the top-10 words of an LDA topic and a BERTopic topic. 1 = same words.",
        read="> 0.3 the two models found the same story", bands=None),
}


def rating(key: str, value) -> str:
    info = METRICS.get(key, {})
    bands = info.get("bands")
    if not bands or value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    for thr, label in bands:
        if value < thr:
            return label
    return bands[-1][1]


def fmt(key: str, value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if key in ("diversity", "outlier_rate_raw"):
        return f"{100 * value:.1f}%"
    if key == "n_topics":
        return str(int(value))
    if key == "runtime_s":
        return f"{value:.2f}s"
    return f"{value:.3f}"


def better_of(key: str, a, b) -> int:
    """-1 if a is better, 1 if b is better, 0 if tie / not comparable."""
    want = METRICS.get(key, {}).get("better", "—")
    if a is None or b is None or a != a or b != b or want == "—" or abs(a - b) < 1e-9:
        return 0
    if want in ("higher", "closer to 0"):
        return -1 if a > b else 1
    if want == "lower":
        return -1 if a < b else 1
    return 0
