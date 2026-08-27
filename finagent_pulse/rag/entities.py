"""Finance-domain entity extraction for the mini knowledge graph.

Headlines are short, noisy and heavy on tickers and abbreviations, where a
general-purpose NER model tends to mislabel ("Fed" as a person, "Apple" as an
organisation with no market meaning).  A curated alias lexicon is both more
accurate on this corpus and fully explainable -- every graph edge can be traced
back to the exact surface form that produced it.
"""
from __future__ import annotations

import re
from collections import defaultdict

# entity id -> (type, surface aliases)
LEXICON: dict[str, tuple[str, list[str]]] = {
    # ---- mega-cap companies ------------------------------------------------
    "AAPL":   ("company", ["apple", "aapl", "iphone"]),
    "MSFT":   ("company", ["microsoft", "msft", "azure"]),
    "GOOGL":  ("company", ["google", "alphabet", "googl"]),
    "AMZN":   ("company", ["amazon", "amzn"]),
    "NVDA":   ("company", ["nvidia", "nvda"]),
    "META":   ("company", ["meta platforms", "facebook", "meta"]),
    "TSLA":   ("company", ["tesla", "tsla", "musk"]),
    "NFLX":   ("company", ["netflix", "nflx"]),
    "JPM":    ("company", ["jpmorgan", "jp morgan", "jpm"]),
    "GS":     ("company", ["goldman sachs", "goldman"]),
    "BAC":    ("company", ["bank of america"]),
    "WFC":    ("company", ["wells fargo"]),
    "BRK":    ("company", ["berkshire", "buffett"]),
    "XOM":    ("company", ["exxon", "exxonmobil"]),
    "CVX":    ("company", ["chevron"]),
    "BA":     ("company", ["boeing"]),
    "INTC":   ("company", ["intel"]),
    "AMD":    ("company", ["amd", "advanced micro"]),
    "DIS":    ("company", ["disney"]),
    "WMT":    ("company", ["walmart"]),
    "PFE":    ("company", ["pfizer"]),
    "MRNA":   ("company", ["moderna"]),
    "SVB":    ("company", ["silicon valley bank", "svb"]),
    "CS":     ("company", ["credit suisse"]),
    "FRC":    ("company", ["first republic"]),

    # ---- indices & instruments --------------------------------------------
    "SP500":  ("index", ["s&p 500", "s&p500", "sp 500", "spx", "spy", "s&p"]),
    "NASDAQ": ("index", ["nasdaq", "ndx", "qqq"]),
    "DJIA":   ("index", ["dow jones", "dow", "djia"]),
    "RUSSELL":("index", ["russell 2000", "russell"]),
    "VIX":    ("index", ["vix", "volatility index", "fear gauge"]),
    "BONDS":  ("asset", ["treasury", "treasuries", "yield", "10-year", "bond"]),
    "GOLD":   ("asset", ["gold", "bullion"]),
    "OIL":    ("asset", ["oil", "crude", "wti", "brent", "opec"]),
    "CRYPTO": ("asset", ["bitcoin", "crypto", "ethereum"]),
    "DOLLAR": ("asset", ["dollar", "greenback", "dxy"]),

    # ---- institutions & policy --------------------------------------------
    "FED":    ("institution", ["federal reserve", "fed", "fomc", "powell"]),
    "ECB":    ("institution", ["ecb", "european central bank", "lagarde"]),
    "SEC":    ("institution", ["sec", "securities and exchange"]),
    "TREASURY_DEPT": ("institution", ["treasury department", "yellen"]),
    "WHITEHOUSE":    ("institution", ["white house", "biden", "trump", "congress"]),

    # ---- macro themes ------------------------------------------------------
    "INFLATION":  ("macro", ["inflation", "cpi", "ppi", "price index", "deflation"]),
    "RATES":      ("macro", ["interest rate", "rate hike", "rate cut", "tightening",
                             "hawkish", "dovish", "basis points", "monetary policy"]),
    "RECESSION":  ("macro", ["recession", "downturn", "contraction", "hard landing",
                             "soft landing"]),
    "JOBS":       ("macro", ["jobs report", "payrolls", "unemployment", "labor market",
                             "jobless"]),
    "GDP":        ("macro", ["gdp", "economic growth"]),
    "EARNINGS":   ("macro", ["earnings", "profit", "revenue", "guidance", "eps",
                             "quarterly results"]),
    "TARIFFS":    ("macro", ["tariff", "trade war", "trade deal", "sanctions"]),
    "COVID":      ("macro", ["covid", "coronavirus", "pandemic", "lockdown", "omicron"]),
    "UKRAINE":    ("macro", ["ukraine", "russia", "putin", "invasion"]),
    "BANKCRISIS": ("macro", ["bank failure", "banking crisis", "bank run", "contagion",
                             "bailout"]),
    "AI":         ("macro", ["artificial intelligence", " ai ", "chatgpt", "generative ai",
                             "chips act", "semiconductor"]),
    "DEBTCEILING":("macro", ["debt ceiling", "default", "shutdown"]),
    "HOUSING":    ("macro", ["housing", "mortgage", "real estate"]),

    # ---- market regime vocabulary -----------------------------------------
    "SELLOFF":    ("regime", ["selloff", "sell-off", "plunge", "tumble", "slump",
                              "crash", "rout", "correction", "bear market"]),
    "RALLY":      ("regime", ["rally", "surge", "soar", "record high", "all-time high",
                              "jump", "bull market", "rebound"]),
    "VOLATILITY": ("regime", ["volatility", "turbulence", "swing", "choppy"]),
}

# Precompiled matchers. Word boundaries stop "ai" matching "said" and "fed"
# matching "federated"; "s&p" needs a literal escape.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (eid, re.compile(r"(?<![a-z0-9])" + re.escape(alias.strip()) + r"(?![a-z0-9])"))
    for eid, (_t, aliases) in LEXICON.items()
    for alias in aliases
]

ENTITY_TYPE = {eid: t for eid, (t, _a) in LEXICON.items()}


def extract_entities(text: str) -> list[str]:
    """Return the sorted unique entity ids mentioned in ``text``."""
    low = f" {text.lower()} "
    found = {eid for eid, pat in _PATTERNS if pat.search(low)}
    return sorted(found)


def build_knowledge_graph(docs: "list[dict]"):
    """Co-occurrence knowledge graph over headline entities.

    Nodes carry a mention count and mean FinBERT sentiment; edges carry the
    number of headlines in which the pair co-occurred plus the mean sentiment
    of those headlines.  Edge weight is what drives query expansion.
    """
    import networkx as nx

    g = nx.Graph()
    node_sent: dict[str, list[float]] = defaultdict(list)
    edge_sent: dict[tuple[str, str], list[float]] = defaultdict(list)

    for doc in docs:
        ents = doc.get("entities") or []
        sent = float(doc.get("sentiment", 0.0))
        for e in ents:
            node_sent[e].append(sent)
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                edge_sent[(a, b)].append(sent)

    for eid, sents in node_sent.items():
        g.add_node(eid,
                   entity_type=ENTITY_TYPE.get(eid, "other"),
                   mentions=len(sents),
                   mean_sentiment=float(sum(sents) / len(sents)))

    for (a, b), sents in edge_sent.items():
        g.add_edge(a, b,
                   weight=len(sents),
                   mean_sentiment=float(sum(sents) / len(sents)))
    return g


# Entities mentioned in more than this share of the corpus act as stopwords.
# Every headline in this dataset is about the S&P 500, so "SP500", "DJIA" and
# "NASDAQ" carry no discriminative signal and must not drive query expansion.
HUB_DF_THRESHOLD = 0.15


def _hub_entities(graph) -> set[str]:
    total = sum(d.get("mentions", 0) for _n, d in graph.nodes(data=True))
    if total <= 0:
        return set()
    # Normalise against the busiest node so the cut-off is corpus-relative.
    busiest = max(d.get("mentions", 0) for _n, d in graph.nodes(data=True)) or 1
    return {n for n, d in graph.nodes(data=True)
            if d.get("mentions", 0) / busiest > HUB_DF_THRESHOLD}


def expand_query(query: str, graph, top_n: int = 4) -> tuple[str, list[str]]:
    """Widen a query with the most *distinctively* associated graph neighbours.

    Neighbours are ranked by Salton association strength

        w(a,b) / sqrt(mentions(a) * mentions(b))

    rather than by raw co-occurrence count.  Raw counts are dominated by hub
    entities -- "S&P 500", "Dow Jones" and "Nasdaq" appear in a large share of
    headlines and co-occur with everything, so expanding on them drags the
    query toward generic index boilerplate.  Normalising by mention frequency
    keeps the neighbours that are specifically informative about the seed:
    "Fed" then expands to INFLATION and RATES rather than to SP500.
    """
    seeds = [e for e in extract_entities(query) if graph.has_node(e)]
    if not seeds:
        return query, []

    hubs = _hub_entities(graph)
    scores: dict[str, float] = defaultdict(float)
    for seed in seeds:
        m_seed = graph.nodes[seed].get("mentions", 1)
        for nbr in graph.neighbors(seed):
            if nbr in seeds or nbr in hubs:
                continue
            m_nbr = graph.nodes[nbr].get("mentions", 1)
            co = graph[seed][nbr]["weight"]
            scores[nbr] += co / ((m_seed * m_nbr) ** 0.5)

    neighbours = [e for e, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]]
    if not neighbours:
        return query, []

    terms = [LEXICON[e][1][0] for e in neighbours if e in LEXICON]
    return f"{query} {' '.join(terms)}", neighbours


if __name__ == "__main__":
    for t in ["Fed signals rate cut as inflation cools",
              "Nvidia earnings beat sends Nasdaq to record high",
              "Silicon Valley Bank collapse triggers banking contagion fears"]:
        print(f"{t}\n  -> {extract_entities(t)}\n")
