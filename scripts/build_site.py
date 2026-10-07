"""Render the existing learning notes; fail on broken local file links."""

import argparse
import csv
import json
from html import escape
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit, urlunsplit

import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIRS = ("Core-algorithms", "Biological-Systems", "Environmental", "Explore-PyTorch", "explore_stuff", "studies", "data", "assets", "biotech", "tests")
STYLE = """
:root { color-scheme:light; --ink:#203a3b; --paper:#faf8f2; --accent:#237b70; --rule:#d4dcd2; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:17px/1.8 system-ui,sans-serif; }
header { border-top:5px solid var(--ink); border-bottom:1px solid var(--rule); padding:1.2rem max(5vw,1rem); display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }
header a { font-weight:750; text-decoration:none; } header .brand { font-family:Georgia,serif; font-size:1.45rem; }
header span { display:block; font-size:.7rem; letter-spacing:.15em; text-transform:uppercase; font-family:system-ui,sans-serif; font-weight:500; }
nav { margin-left:auto; display:flex; flex-wrap:wrap; gap:1.4rem; font-size:.8rem; }
main { max-width:1240px; margin:auto; padding:2.5rem 2rem 6rem; overflow-wrap:anywhere; }
main>p,main>ul,main>ol,main>blockquote { max-width:80ch; }
main>p:has(img) { max-width:none; margin:2.2rem -1rem; }
h1,h2,h3 { line-height:1.2; letter-spacing:-.025em; text-wrap:balance; }
h1 { font:500 clamp(2.4rem,5vw,4.3rem)/1.08 Georgia,serif; max-width:23ch; margin:2.4rem 0 1.4rem; }
h2 { font:500 clamp(1.5rem,3vw,2.2rem)/1.2 Georgia,serif; margin-top:3rem; padding-top:1.4rem; border-top:1px solid var(--rule); }
h3 { font-size:1.1rem; margin-top:2rem; }
a { color:var(--accent); text-underline-offset:.23em; text-decoration-thickness:1px; } a:hover { color:#984525; }
a:focus-visible,summary:focus-visible { outline:3px solid #bc5939; outline-offset:4px; }
pre { background:var(--ink); color:#edf4ed; padding:1.4rem; overflow:auto; border-radius:3px; font-size:.85rem; line-height:1.7; }
code { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.86em; } :not(pre)>code { background:#e8eee4; padding:.12em .3em; }
table { display:block; overflow:auto; border-collapse:collapse; margin:1.8rem 0; font-size:.9rem; line-height:1.65; }
th,td { border-bottom:1px solid var(--rule); padding:.85rem; text-align:left; vertical-align:top; min-width:8rem; }
th { font-size:.75rem; letter-spacing:.055em; text-transform:uppercase; background:#eaf0e7; }
blockquote { border-left:3px solid #bc5939; margin:1.5rem 0; padding:.1rem 1.2rem; background:#f2eee2; }
img { max-width:100%; height:auto; } details { border-block:1px solid var(--rule); padding:.5rem 0; font-size:.8rem; }
summary { cursor:pointer; font-weight:650; letter-spacing:.04em; } details .toc { columns:2; padding-right:1rem; }
footer { border-top:1px solid var(--rule); padding:1.5rem; text-align:center; font-size:.8rem; }
.skip { position:absolute; left:-9999px; } .skip:focus { position:static; }
@media(max-width:650px) { main { padding:1rem 1.2rem 3rem; } nav { margin-left:0; gap:1rem; } details .toc { columns:1; } }
@media print { nav,details,.skip { display:none; } main { max-width:none; } pre { white-space:pre-wrap; } }
@media(prefers-reduced-motion:no-preference) { html { scroll-behavior:smooth; } }
"""


class ImageHeadingLabels(Treeprocessor):
    def run(self, root):
        used = {item.get("id") for item in root.iter() if item.get("id")}
        for heading in root.iter():
            if heading.tag not in {"h1", "h2", "h3", "h4", "h5", "h6"} or "".join(heading.itertext()).strip():
                continue
            label = " ".join(image.get("alt", "") for image in heading.iter("img")).strip()
            if label:
                heading.set("data-toc-label", label)
                base = re.sub(r"[^\w-]+", "-", label.lower()).strip("-") or "heading"
                identifier, suffix = base, 1
                while identifier in used:
                    identifier, suffix = f"{base}-{suffix}", suffix + 1
                heading.set("id", identifier)
                used.add(identifier)


class AccessibleImageHeadings(Extension):
    def extendMarkdown(self, renderer):
        renderer.treeprocessors.register(ImageHeadingLabels(renderer), "image_heading_labels", 6)


def content_files(root):
    files = list(root.glob("*.md"))
    files += [p for folder in CONTENT_DIRS for p in (root / folder).rglob("*")
              if p.is_file() and not p.is_symlink() and p.suffix in {".md", ".html", ".py", ".svg", ".png", ".gif", ".json", ".csv", ".txt", ".data", ".dly", ".fasta", ".fastq", ".vcf", ".css", ".mjs", ".sql"}]
    return sorted(p for p in files if p.name != "AGENTS.md")


def page_path(path):
    return Path("index.html") if path == Path("README.md") else path.with_suffix(".html")


def rewrite_link(url):
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path.endswith(".md"):
        return url
    path = "index.html" if parsed.path == "README.md" else parsed.path[:-3] + ".html"
    return urlunsplit(parsed._replace(path=path))


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.targets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attribute in ("href", "src"):
            if attrs.get(attribute):
                self.targets.append(attrs[attribute])


def validate_links(site):
    errors = []
    for page in sorted(site.rglob("*.html")):
        links = Links()
        links.feed(page.read_text(encoding="utf-8"))
        for url in links.targets:
            target = urlsplit(url)
            if target.scheme or target.netloc:
                continue
            path = (page.parent / unquote(target.path)).resolve() if target.path else page.resolve()
            if not path.is_relative_to(site.resolve()) or not path.exists():
                errors.append(f"{page.relative_to(site)}: missing local target {url}")
    if errors:
        raise ValueError("\n".join(errors))


def inline_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")


def search_index(root):
    entries = []
    for path in content_files(root):
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        headings = re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE)
        headings = [re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", heading) for heading in headings]
        entries.append({"title": headings[0] if headings else path.stem,
                        "path": page_path(path.relative_to(root)).as_posix(),
                        "text": re.sub(r"<[^>]+>", " ", text[:1800]) + " " + " ".join(headings)})
    return entries


def explorer_markup(root):
    report_path = root / "studies/results/benchmark.json"
    predictions_path = root / "studies/results/test_predictions.csv"
    if not report_path.exists() or not predictions_path.exists():
        return ""
    report = json.loads(report_path.read_text())
    models = list(report["test_evaluation"])
    with predictions_path.open() as source:
        rows = [{"malignant": int(row["malignant"]), **{name: float(row[name]) for name in models}} for row in csv.DictReader(source)]
    selected = report["selected_by_validation_log_loss"]
    options = ''.join(f'<option value="{escape(name, quote=True)}" {"selected" if name == selected else ""}>{escape(name)}</option>' for name in models)
    counts = dict(tn=0, fp=0, fn=0, tp=0)
    for row in rows:
        counts[("tp" if row[selected] >= .5 else "fn") if row["malignant"] else ("fp" if row[selected] >= .5 else "tn")] += 1
    cards = ''.join(f'<div class="metric {"error" if key in ("fp", "fn") else ""}"><dt>{label}</dt><dd id="count-{key}">{counts[key]}</dd><div class="metric-track" aria-hidden="true"><div class="metric-bar" id="bar-{key}"></div></div></div>'
                    for key, label in (("tn", "True negatives"), ("fp", "False positives"), ("fn", "False negatives"), ("tp", "True positives")))
    return f'''<section class="explorer" id="threshold-explorer" aria-labelledby="explorer-title">
<p class="eyebrow">Try the decision rule</p><h2 id="explorer-title">What changes when I move the cutoff?</h2>
<p>I keep the recorded model predictions fixed. Change the threshold to see which errors move with it.</p>
<div class="explorer-controls"><label>Model<select id="model-choice">{options}</select></label>
<label>Probability threshold <output id="threshold-value" for="decision-threshold">0.50</output><input id="decision-threshold" type="range" min="0.05" max="0.95" step="0.01" value="0.50"></label>
<button id="reset-threshold" type="button" class="secondary">Reset comparison</button></div>
<dl class="metric-grid">{cards}</dl><p id="decision-summary" aria-live="polite"></p>
<p class="caveat">These are {len(rows)} already-visible test records, not new predictions. Exploring a cutoff here does not validate it or authorize choosing a clinical threshold from the test set.</p>
<script type="application/json" id="prediction-data">{inline_json({"selected": selected, "rows": rows})}</script></section>'''


TOOLS = '''<progress id="reading-progress" value="0" max="1" aria-label="Reading progress"></progress>
<div class="site-tools"><details id="search-panel"><summary>Find a study</summary><div class="search-fields"><label class="visually-hidden" for="study-search">Search studies</label><input id="study-search" type="search" placeholder="Try calibration, protein, or retries" aria-controls="search-results"><button id="clear-search" type="button" class="secondary">Clear</button></div><p id="search-status" aria-live="polite"></p><ul id="search-results"></ul></details><button id="theme-toggle" type="button" class="secondary" aria-pressed="false">Night palette</button></div>
<div id="interaction-status" class="visually-hidden" role="status" aria-live="polite"></div>
<dialog id="figure-dialog" aria-label="Figure inspection"><div class="dialog-tools"><button type="button" id="close-figure">Close figure</button><label for="figure-zoom">Zoom<input id="figure-zoom" type="range" min="100" max="200" step="25" value="100"><output id="zoom-value" for="figure-zoom">100%</output></label></div><p id="figure-caption"></p><div class="dialog-image"><img alt=""></div></dialog>'''


def build_site(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output or root / "_site").resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("The output must be a separate build directory.")
    if output.exists() and any(output.iterdir()) and not (output / ".curious-coder-site").exists():
        raise ValueError("Refusing to overwrite a nonempty directory not created by this builder.")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".curious-coder-site").touch()
    count = 0
    interactive = (root / "assets/site.mjs").exists() and (root / "assets/site.css").exists()
    index = search_index(root) if interactive else []
    route = [(path, label) for path, label in (
        ("README.md", "Overview"), ("START_HERE.md", "Begin here"), ("studies/evidence_retrieval.md", "Find the evidence"), ("studies/evidence_contracts.md", "Bound the explanation"), ("studies/learning_signals.md", "Check the optimized signal"),
        ("biotech/README.md", "Inspect the measurement"), ("biotech/facility_workflow.md", "Follow the process record"),
        ("studies/clinical_benchmark.md", "Compare models"), ("studies/statistical_validation.md", "Examine uncertainty"),
        ("studies/protein_adaptation.md", "Inspect weight updates"), ("studies/training_math.md", "Check distributed training"),
        ("studies/engineering.md", "Deliver the result"), ("studies/execution_contracts.md", "Check execution constraints"),
        ("studies/technology_map.md", "Choose the next integration")) if (root / path).exists()]
    for source in content_files(root):
        relative = source.relative_to(root)
        destination = output / (page_path(relative) if source.suffix == ".md" else relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix != ".md":
            shutil.copyfile(source, destination)
            continue
        renderer = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists", "md_in_html", AccessibleImageHeadings()])
        body = renderer.convert(source.read_text(encoding="utf-8"))
        body = re.sub(r'(href|src)="([^"]*)"',
                      lambda match: f'{match[1]}="{escape(rewrite_link(unquote_html(match[2])), quote=True)}"', body)
        headings = renderer.toc_tokens
        title = headings[0]["name"] if headings else source.stem.replace("_", " ")
        home = os.path.relpath(output / "index.html", destination.parent).replace(os.sep, "/")
        source_url = "https://github.com/Cazzy-Aporbo/Curious-Coder/blob/main/" + relative.as_posix()
        navigation = []
        for label, target in (("Start here", "START_HERE.md"), ("Biotech QC", "biotech/README.md"), ("Protein adaptation", "studies/protein_adaptation.md"), ("Measured ML", "studies/clinical_benchmark.md"), ("Tools & interfaces", "studies/technology_map.md")):
            if (root / target).exists():
                link = os.path.relpath(output / page_path(Path(target)), destination.parent).replace(os.sep, "/")
                navigation.append(f'<a href="{link}">{label}</a>')
        resources, controls = "", ""
        if interactive:
            css = os.path.relpath(output / "assets/site.css", destination.parent).replace(os.sep, "/")
            js = os.path.relpath(output / "assets/site.mjs", destination.parent).replace(os.sep, "/")
            resources = f'<link rel="stylesheet" href="{css}"><script type="module" src="{js}"></script>'
            controls = TOOLS + f'<script type="application/json" id="search-data">{inline_json(index)}</script>'
            body = body.replace('<div class="interactive-results"></div>', explorer_markup(root))
        chapter_links = []
        route_paths = [path for path, _ in route]
        if relative.as_posix() in route_paths:
            position = route_paths.index(relative.as_posix())
            for adjacent, direction in ((position - 1, "Previous"), (position + 1, "Next")):
                if 0 <= adjacent < len(route):
                    path, label = route[adjacent]
                    link = os.path.relpath(output / page_path(Path(path)), destination.parent).replace(os.sep, "/")
                    chapter_links.append(f'<a class="button secondary" href="{escape(link, quote=True)}">{direction} · {escape(label)}</a>')
            body += f'<nav class="chapter-nav" aria-label="Learning route"><p>Learning route · {position + 1} of {len(route)}</p>{"".join(chapter_links)}</nav>'
        page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} · Curious Coder</title><style>{STYLE}</style>{resources}</head>
<body><a class="skip" href="#main">Skip to content</a>
<header><a class="brand" href="{home}">Curious Coder<span>Scientific computing / Cazandra Aporbo</span></a><nav aria-label="Study navigation">{' '.join(navigation)}</nav></header>
{controls}<main id="main"><details><summary>On this page</summary>{renderer.toc}</details>{body}</main>
<footer>Curious Coder · Methods, assumptions, evidence. <a href="{escape(source_url, quote=True)}">Read this page's source</a></footer>
</body></html>'''
        destination.write_text(page, encoding="utf-8")
        count += 1
    (output / ".nojekyll").touch()
    validate_links(output)
    return count


def unquote_html(value):
    from html import unescape
    return unescape(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(f"Built {build_site(output=args.output)} learning pages; local file links verified.")
