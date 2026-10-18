"""Generate the SYNTHETIC multi-hop corpus and question set (seeded, fully reproducible).

    python data/make_data.py          # rewrites data/corpus.jsonl and data/questions.jsonl

Everything here is invented: person, company, product, city and country names are built from
syllables (any resemblance to a real name is coincidence), and the facts are random. The sentences
follow a few fixed templates on purpose, so the offline RuleBasedExtractor can read them: the
benchmark measures retrieval, not an LLM's extraction.

Facts per world (seed 20261014): 60 companies, one founder each, a headquarters city (12 cities in
5 countries), one product each, and 20 acquisitions (distinct acquirers and targets).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261014
N_COMPANIES = 60
N_ACQUISITIONS = 20
N_CITIES = 12
N_COUNTRIES = 5
N_FILLER = 20

FIRST = ["Alma", "Bruno", "Chen", "Dara", "Elio", "Farah", "Gael", "Hana", "Ivo", "Jun", "Kofi",
         "Lena", "Mira", "Nils", "Omar", "Pia", "Quin", "Rhea", "Sven", "Tavi"]
SUR_A = ["Ver", "Kal", "Mor", "Tel", "Bran", "Sol", "Dun", "Fen", "Lor", "Pel", "Ash", "Wex"]
SUR_B = ["wick", "dane", "mere", "vane", "holt", "stow", "ley", "rin", "cott", "shaw", "mont",
         "gar"]
CO_A = ["Kal", "Orr", "Vel", "Hal", "Nor", "Tir", "Zan", "Bel", "Cor", "Dra", "Eln", "Fyr", "Gri",
        "Hesp", "Ilm", "Jor", "Kyr", "Lum", "Mav", "Nex"]
CO_B = ["den", "in", "trix", "cyon", "mar", "vos", "dex", "lin", "tek", "ron", "sa", "mir"]
CO_KIND = ["Labs", "Dynamics", "Systems", "Works", "Analytics", "Robotics", "Networks"]
SECTORS = ["software", "logistics", "energy", "robotics", "materials", "health-tech", "mapping"]
CITY_A = ["Tel", "Mor", "Vand", "Ash", "Brin", "Calder", "Dov", "Eskan", "Fal", "Gor", "Hesk",
          "Ilm"]
CITY_B = ["mouth", "ford", "haven", "wick", "port", "stead", "bury", "ness"]
COUNTRY = ["Veldor", "Anquar", "Tessaly", "Morvane", "Ghalis", "Ostrel", "Bremar"]
PROD_A = ["Aur", "Zen", "Quil", "Brev", "Tal", "Kes", "Lyr", "Mor", "Nov", "Pax", "Rix", "Sol",
          "Tyr", "Ulm", "Vex", "Wyn", "Xan", "Yor", "Zel", "Cal"]
PROD_B = ["ex", "tra", "lo", "ia", "os", "rel", "is", "on", "ar", "ik"]


def unique(rng: random.Random, make, n: int, taken: set[str]) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        v = make(rng)
        if v not in taken:
            taken.add(v)
            out.append(v)
    return out


def build() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    taken: set[str] = set()
    people = unique(rng, lambda r: f"{r.choice(FIRST)} {r.choice(SUR_A)}{r.choice(SUR_B)}",
                    N_COMPANIES, taken)
    companies = unique(rng, lambda r: f"{r.choice(CO_A)}{r.choice(CO_B)} {r.choice(CO_KIND)}",
                       N_COMPANIES, taken)
    cities = unique(rng, lambda r: f"{r.choice(CITY_A)}{r.choice(CITY_B)}", N_CITIES, taken)
    countries = rng.sample(COUNTRY, N_COUNTRIES)
    products = unique(rng, lambda r: f"{r.choice(PROD_A)}{r.choice(PROD_B)}-{r.randint(2, 9)}",
                      N_COMPANIES, taken)

    city_country = {c: countries[i % N_COUNTRIES] for i, c in enumerate(cities)}
    hq = {co: rng.choice(cities) for co in companies}
    founder = dict(zip(companies, people, strict=True))
    product = dict(zip(companies, products, strict=True))
    year = {co: rng.randint(1998, 2021) for co in companies}
    sector = {co: rng.choice(SECTORS) for co in companies}

    shuffled = companies[:]
    rng.shuffle(shuffled)
    acquirers = shuffled[:N_ACQUISITIONS]
    targets = shuffled[N_ACQUISITIONS:2 * N_ACQUISITIONS]
    acquired_by = dict(zip(targets, acquirers, strict=True))      # target -> acquirer
    acq_year = {t: rng.randint(2015, 2025) for t in targets}

    docs: list[dict] = []
    for co in companies:
        p = founder[co]
        docs.append({"id": f"person-{len(docs):03d}", "text":
                     f"# {p}\n{p} is an engineer. {p} founded {co} in {year[co]}."})
    for co in companies:
        docs.append({"id": f"company-{len(docs):03d}", "text":
                     f"# {co}\n{co} is a {sector[co]} company headquartered in {hq[co]}. "
                     f"{co} makes {product[co]}."})
    for t in targets:
        a = acquired_by[t]
        docs.append({"id": f"deal-{len(docs):03d}", "text":
                     f"# {a} acquires {t}\n{a} acquired {t} in {acq_year[t]}."})
    for c in cities:
        docs.append({"id": f"city-{len(docs):03d}", "text":
                     f"# {c}\n{c} is a port in {city_country[c]}."})
    for co in companies:
        docs.append({"id": f"product-{len(docs):03d}", "text":
                     f"# {product[co]}\n{product[co]} is a consumer device released in "
                     f"{year[co] + 2}."})
    filler = [
        "Analysts say that every software company needs a clear founder story, and that many a "
        "founded firm starts small.",
        "Industry notes: the city a company is headquartered in matters less than its customers.",
        "A weekly digest about which company acquired which rival, and what founders did next.",
        "Port cities attract logistics companies, and the country they sit in sets the tax regime.",
        "Which product is made where? Supply chains for devices cross many a city and country.",
    ]
    for i in range(N_FILLER):
        docs.append({"id": f"note-{len(docs):03d}", "text":
                     f"# Industry note {i + 1}\n{filler[i % len(filler)]}"})
    rng.shuffle(docs)

    by_prefix = {}
    for d in docs:
        title = d["text"].split("\n", 1)[0][2:]
        by_prefix.setdefault(d["id"].rsplit("-", 1)[0], {})[title] = d["id"]
    pid = lambda person: by_prefix["person"][person]                       # noqa: E731
    cid = lambda comp: by_prefix["company"][comp]                          # noqa: E731
    did = lambda target: by_prefix["deal"][f"{acquired_by[target]} acquires {target}"]  # noqa: E731
    yid = lambda city: by_prefix["city"][city]                             # noqa: E731

    qs: list[dict] = []

    def add(hops: int, kind: str, question: str, gold: list[str], answer: str) -> None:
        qs.append({"id": f"q{len(qs) + 1:03d}", "hops": hops, "kind": kind, "question": question,
                   "gold_docs": sorted(set(gold)), "answer": answer})

    pick = rng.sample(companies, 20)
    for co in pick[:10]:
        add(1, "hq", f"In which city is {co} headquartered?", [cid(co)], hq[co])
    for co in pick[10:]:
        add(1, "product", f"Which product does {co} make?", [cid(co)], product[co])

    group = rng.sample(companies, 40)
    for co in group[:10]:
        add(2, "founder>hq",
            f"In which city is the company founded by {founder[co]} headquartered?",
            [pid(founder[co]), cid(co)], hq[co])
    for co in group[10:20]:
        add(2, "founder>product", f"Which product is made by the company founded by {founder[co]}?",
            [pid(founder[co]), cid(co)], product[co])
    for t in rng.sample(targets, 10):
        a = acquired_by[t]
        add(2, "acquirer>hq", f"In which city is the company that acquired {t} headquartered?",
            [did(t), cid(a)], hq[a])
    for t in rng.sample(targets, 10):
        a = acquired_by[t]
        add(2, "acquirer>founder", f"Who founded the company that acquired {t}?",
            [did(t), pid(founder[a])], founder[a])

    for t in rng.sample(targets, 10):
        a = acquired_by[t]
        add(3, "founder>acquirer>product",
            f"Which product is made by the company that acquired the company founded by "
            f"{founder[t]}?", [pid(founder[t]), did(t), cid(a)], product[a])
    for co in rng.sample(companies, 10):
        add(3, "founder>hq>country",
            f"In which country is the city where the company founded by {founder[co]} is "
            f"headquartered?", [pid(founder[co]), cid(co), yid(hq[co])], city_country[hq[co]])
    return docs, qs


def main() -> None:
    here = Path(__file__).resolve().parent
    docs, qs = build()
    (here / "corpus.jsonl").write_text(
        "".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8", newline="\n")
    (here / "questions.jsonl").write_text(
        "".join(json.dumps(q, sort_keys=True) + "\n" for q in qs), encoding="utf-8", newline="\n")
    print(f"wrote {len(docs)} documents and {len(qs)} questions")


if __name__ == "__main__":
    main()
