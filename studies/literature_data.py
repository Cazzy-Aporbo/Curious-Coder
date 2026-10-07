"""Acquire a bounded Europe PMC API snapshot without crawling publisher websites."""

from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from studies.data import ROOT, digest


ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SNAPSHOT = ROOT / "data/literature_evidence"
QUERIES = {
    "prediction_evaluation": '(TITLE_ABS:"clinical prediction" AND (TITLE_ABS:validation OR TITLE_ABS:calibration)) AND FIRST_PDATE:[2018-01-01 TO 2025-12-31] AND SRC:MED',
    "protein_models": '(TITLE_ABS:"protein language model" OR TITLE_ABS:"protein representation") AND FIRST_PDATE:[2018-01-01 TO 2025-12-31] AND SRC:MED',
    "measurement_quality": '(TITLE_ABS:"sequencing quality" OR TITLE_ABS:"assay validation") AND FIRST_PDATE:[2018-01-01 TO 2025-12-31] AND SRC:MED',
}
ANCHORS = [
    {"key": "reporting", "doi": "10.1136/bmj-2023-078378", "question": "What should a report disclose about the population, intended use, and evaluation of a clinical prediction model?"},
    {"key": "selection_bias", "doi": "10.1186/1471-2105-7-91", "question": "Why can tuning a classifier and estimating its error with the same cross validation give an optimistic result?"},
    {"key": "decision_utility", "doi": "10.1177/0272989x06295361", "question": "How can the consequences of false positive decisions be weighed against correct detection when comparing prediction models?"},
    {"key": "protein_model", "doi": "10.1126/science.ade2574", "question": "Can a model trained on protein sequences learn information about atomic structure without relying on a multiple sequence alignment at inference?"},
    {"key": "read_encoding", "doi": "10.1093/nar/gkp1137", "question": "Why do sequencing quality strings need an explicit encoding convention when exchanging FASTQ files?"},
    {"key": "assay_window", "doi": "10.1177/108705719900400206", "question": "Which screening assay statistic combines variability of controls with the distance between their signals?"},
]


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(value):
    parser = PlainText()
    parser.feed(unescape(value or ""))
    return " ".join(" ".join(parser.parts).split())


def normalize_record(record, query_name):
    identifier = str(record.get("id", ""))
    if record.get("source") != "MED" or not identifier.isdigit() or not record.get("title"):
        raise ValueError("This snapshot requires a MED identifier and a nonempty title.")
    license_name = " ".join(str(record.get("license", "")).lower().split())
    reusable = license_name in {"cc by", "cc0"}
    abstract = plain_text(record.get("abstractText", ""))
    corrections = record.get("commentCorrectionList", {}).get("commentCorrection", [])
    return {
        "key": f"MED:{identifier}", "pmid": identifier, "pmcid": record.get("pmcid"),
        "doi": str(record.get("doi", "")).strip().lower(), "title": plain_text(record["title"]),
        "authors": record.get("authorString", ""), "year": record.get("pubYear"),
        "journal": record.get("journalInfo", {}).get("journal", {}).get("title", "unknown"),
        "language": record.get("language", "unknown"), "publication_types": record.get("pubTypeList", {}).get("pubType", []),
        "license_metadata": license_name or "not supplied", "abstract_present_upstream": bool(abstract),
        "abstract_included": bool(abstract) and reusable, "abstract": abstract if reusable else "",
        "abstract_policy": "Retain text only when API license metadata is cc by or cc0; otherwise metadata/title only.",
        "source_url": f"https://europepmc.org/article/MED/{identifier}",
        "notice_links": [{"type": entry.get("type", "unknown"), "source": entry.get("source"), "id": entry.get("id")} for entry in corrections],
        "queries": [query_name],
    }


def merge_record(records, incoming):
    key = incoming["doi"] or incoming["key"]
    if key in records:
        existing = records[key]
        existing["queries"] = sorted(set(existing["queries"] + incoming["queries"]))
        return False
    records[key] = incoming
    return True


def fetch_page(query, cursor="*", page_size=25, *, wait=time.sleep):
    if not 1 <= page_size <= 25:
        raise ValueError("This teaching client caps each API page at 25 records.")
    url = ENDPOINT + "?" + urlencode({"query": query, "cursorMark": cursor, "pageSize": page_size, "format": "json", "resultType": "core"})
    for attempt in range(3):
        wait(1.0)
        request = Request(url, headers={"User-Agent": "Curious-Coder-literature-methods/1.0", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read(3_000_001)
            if len(payload) > 3_000_000:
                raise ValueError("API response exceeds the bounded response size.")
            return json.loads(payload), {"url": url, "response_sha256": digest(payload)}
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            retry_after = (error.headers or {}).get("Retry-After", "")
            delay = 2 ** (attempt + 1)
            if retry_after.isdigit():
                delay = float(retry_after)
            elif retry_after:
                try:
                    deadline = parsedate_to_datetime(retry_after)
                    if deadline.tzinfo is None:
                        deadline = deadline.replace(tzinfo=timezone.utc)
                    delay = max(0, (deadline - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError):
                    pass
            if delay > 60:
                raise
            wait(delay)
    raise RuntimeError("Unreachable retry state")


def audit_records(records):
    counts = Counter(record["journal"] for record in records)
    n = len(records)
    return {"records": n, "years": dict(sorted(Counter(str(record["year"]) for record in records).items())),
            "languages": dict(Counter(record["language"] for record in records)),
            "license_metadata": dict(Counter(record["license_metadata"] for record in records)),
            "abstract_missing_upstream": sum(not record["abstract_present_upstream"] for record in records),
            "abstract_withheld_by_reuse_policy": sum(record["abstract_present_upstream"] and not record["abstract_included"] for record in records),
            "abstract_included": sum(record["abstract_included"] for record in records),
            "records_with_notice_links": sum(bool(record["notice_links"]) for record in records),
            "top_journals": [list(item) for item in counts.most_common(8)], "journal_concentration_hhi": sum(count * count for count in counts.values()) / (n * n) if n else None}


def acquire(destination=SNAPSHOT, pages_per_query=2):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use a new snapshot directory; acquisition does not overwrite retained evidence.")
    if pages_per_query not in {1, 2}:
        raise ValueError("Acquisition is bounded to one or two pages per declared query.")
    records, trace, anchor_tasks = {}, [], []
    for name, query in QUERIES.items():
        cursor, seen = "*", set()
        for _ in range(pages_per_query):
            if cursor in seen:
                raise ValueError("The API repeated a cursor; stopping instead of looping indefinitely.")
            seen.add(cursor)
            page, provenance = fetch_page(query, cursor)
            results = page.get("resultList", {}).get("result", [])
            trace.append({**provenance, "query_name": name, "hit_count": page.get("hitCount"), "returned": len(results), "cursor": cursor})
            for result in results:
                merge_record(records, normalize_record(result, name))
            next_cursor = page.get("nextCursorMark")
            if not results or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
    for task in ANCHORS:
        page, provenance = fetch_page(f'DOI:"{task["doi"]}" AND SRC:MED', page_size=2)
        matches = [row for row in page.get("resultList", {}).get("result", []) if str(row.get("doi", "")).lower() == task["doi"]]
        if len(matches) != 1:
            raise ValueError(f"Anchor DOI did not resolve uniquely: {task['doi']}")
        record = normalize_record(matches[0], "explicit_anchor")
        merge_record(records, record)
        anchor_tasks.append({**task, "target": records[record["doi"]]["key"],
                             "judgment_scope": "Authored known-item teaching query; not a complete or blinded relevance judgment."})
        trace.append({**provenance, "query_name": "explicit_anchor", "hit_count": page.get("hitCount"), "returned": len(matches)})
    corpus = sorted(records.values(), key=lambda record: record["key"])
    corpus_bytes = (json.dumps(corpus, indent=2, ensure_ascii=False) + "\n").encode()
    tasks_bytes = (json.dumps(anchor_tasks, indent=2, ensure_ascii=False) + "\n").encode()
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "source": ENDPOINT,
                "queries": QUERIES, "pages_per_query": pages_per_query, "page_size": 25, "request_trace": trace,
                "files": {"corpus.json": digest(corpus_bytes), "queries.json": digest(tasks_bytes)}, "audit": audit_records(corpus),
                "selection_limits": ["MED-indexed records only; default upstream ranking; capped pages; English query vocabulary.",
                    "Six known-item anchors were explicitly added; presence is guaranteed by construction.",
                    "No publisher-site crawling, full-text acquisition, email extraction, or generated scientific answers.",
                    "License metadata and notice links are retained, not independently adjudicated; absence of a notice is not a validity guarantee."]}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "corpus.json").write_bytes(corpus_bytes)
    (destination / "queries.json").write_bytes(tasks_bytes)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_snapshot(directory=SNAPSHOT):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("corpus.json", "queries.json"):
        if digest((directory / name).read_bytes()) != manifest["files"][name]:
            raise ValueError(f"Literature snapshot integrity mismatch: {name}")
    corpus = json.loads((directory / "corpus.json").read_text())
    queries = json.loads((directory / "queries.json").read_text())
    for record in corpus:
        record["title"] = plain_text(record["title"])
        record["abstract"] = plain_text(record["abstract"])
    return corpus, queries, manifest
