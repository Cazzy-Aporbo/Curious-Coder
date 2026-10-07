"""Compare citation retrieval quality, weight precision, and measured CPU cost."""

import argparse
from copy import deepcopy
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from studies.data import ROOT, digest, download
from studies.literature_data import acquire, audit_records, load_snapshot


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WEIGHTS_SHA = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
MODEL_DIR = ROOT / "artifacts/minilm-retrieval"
METHODS = ("TF-IDF", "MiniLM FP32", "MiniLM INT8-linear", "Hybrid RRF")


def acquire_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md", "1_Pooling/config.json"):
        path = MODEL_DIR / name
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        payload = path.read_bytes() if path.exists() else download(url, limit=110_000_000)
        if name == "model.safetensors" and digest(payload) != WEIGHTS_SHA:
            raise ValueError("Checkpoint bytes differ from the pinned publisher digest.")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        files[name] = {"url": url, "sha256": digest(payload)}
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0 model card", "files": files}
    manifest_path = MODEL_DIR / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def ranked_indices(scores, keys):
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(keys),) or not np.isfinite(values).all() or len(set(keys)) != len(keys):
        raise ValueError("Ranking requires aligned finite scores and unique document identifiers.")
    return np.lexsort((np.asarray(keys), -values))


def reciprocal_rank_fusion(rankings, constant=60):
    if not rankings or constant <= 0:
        raise ValueError("Require rankings and a positive rank constant.")
    size = len(rankings[0])
    scores = np.zeros(size, dtype=float)
    for ranking in rankings:
        if sorted(np.asarray(ranking).tolist()) != list(range(size)):
            raise ValueError("Each ranking must be a complete permutation of the same corpus.")
        scores[np.asarray(ranking)] += 1 / (constant + np.arange(1, size + 1))
    return scores


def known_item_metrics(rankings, targets, k=5):
    if len(rankings) != len(targets) or not rankings or k < 1:
        raise ValueError("Require aligned, nonempty diagnostic queries and positive k.")
    ranks = [ranking.index(target) + 1 if target in ranking else None for ranking, target in zip(rankings, targets)]
    return {f"known_target_hit_at_{k}": float(np.mean([rank is not None and rank <= k for rank in ranks])),
            "mean_anchor_reciprocal_rank": float(np.mean([1 / rank if rank else 0 for rank in ranks])), "target_ranks": ranks}


class LexicalIndex:
    def __init__(self, texts, keys):
        self.keys = keys
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
                                         token_pattern=r"(?u)\b\w+(?:-\w+)*\b")
        self.matrix = self.vectorizer.fit_transform(texts)
        self.numeric_bytes = sum(array.nbytes for array in (self.matrix.data, self.matrix.indices, self.matrix.indptr))

    def search(self, query):
        encoded = self.vectorizer.transform([query])
        scores = (self.matrix @ encoded.T).toarray().ravel()
        return ranked_indices(scores, self.keys), scores


def serialize_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


class DenseIndex:
    def __init__(self, model, tokenizer, texts, keys):
        self.model, self.tokenizer, self.keys = model.eval(), tokenizer, keys
        self.matrix = self.encode(texts)
        self.numeric_bytes = self.matrix.nbytes

    def encode(self, texts, batch_size=8):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Text batches must contain nonempty strings.")
        chunks = []
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[offset:offset + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt")
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
        result = np.concatenate(chunks)
        if not np.isfinite(result).all():
            raise ValueError("Encoder produced nonfinite representations.")
        return result

    def search(self, query):
        scores = (self.matrix @ self.encode([query])[0]).ravel()
        return ranked_indices(scores, self.keys), scores


def load_models(include_int8=True):
    from transformers import AutoModel, AutoTokenizer

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Model revision mismatch.")
    for name, metadata in manifest["files"].items():
        path = (MODEL_DIR / name).resolve()
        if not path.is_relative_to(MODEL_DIR.resolve()) or digest(path.read_bytes()) != metadata["sha256"]:
            raise ValueError("Model-resource integrity mismatch.")
    if digest((MODEL_DIR / "model.safetensors").read_bytes()) != WEIGHTS_SHA:
        raise ValueError("Publisher weight digest mismatch.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    start = perf_counter()
    fp32 = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval().requires_grad_(False)
    load_ms = 1000 * (perf_counter() - start)
    if not include_int8:
        return tokenizer, fp32, None, load_ms, 0.0
    if torch.backends.quantized.engine == "none":
        available = [name for name in ("x86", "fbgemm", "qnnpack") if name in torch.backends.quantized.supported_engines]
        if not available:
            raise RuntimeError("This PyTorch build has no supported quantized CPU engine; do not report a fabricated INT8 result.")
        torch.backends.quantized.engine = available[0]
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        quantized = torch.ao.quantization.quantize_dynamic(deepcopy(fp32), {torch.nn.Linear}, dtype=torch.qint8).eval()
    quantize_ms = 1000 * (perf_counter() - start)
    return tokenizer, fp32, quantized, load_ms, quantize_ms


def run(output=ROOT / "studies/results/evidence_retrieval.json", repeats=12):
    if repeats < 2:
        raise ValueError("Use at least two timing repeats.")
    torch.set_num_threads(1)
    corpus, tasks, manifest = load_snapshot()
    keys = [record["key"] for record in corpus]
    texts = [record["title"] + (" " + record["abstract"] if record["abstract_included"] else "") for record in corpus]
    targets = [task["target"] for task in tasks]
    if any(target not in keys for target in targets):
        raise ValueError("Every declared known-item target must exist in the corpus.")
    tokenizer, fp32, int8, load_ms, quantize_ms = load_models()
    indexes, build_ms = {}, {}
    for name, constructor in (("TF-IDF", lambda: LexicalIndex(texts, keys)),
                              ("MiniLM FP32", lambda: DenseIndex(fp32, tokenizer, texts, keys)),
                              ("MiniLM INT8-linear", lambda: DenseIndex(int8, tokenizer, texts, keys))):
        start = perf_counter()
        indexes[name] = constructor()
        build_ms[name] = 1000 * (perf_counter() - start)
    def search(name, query):
        if name == "Hybrid RRF":
            lexical, _ = indexes["TF-IDF"].search(query)
            dense, _ = indexes["MiniLM FP32"].search(query)
            score = reciprocal_rank_fusion([lexical, dense])
            return ranked_indices(score, keys), score
        return indexes[name].search(query)
    for name in METHODS:
        for task in tasks:
            search(name, task["question"])
    timing = {name: [] for name in METHODS}
    order_rng = np.random.default_rng(42)
    for _ in range(repeats):
        for index in order_rng.permutation(len(tasks)):
            for name in order_rng.permutation(METHODS):
                start = perf_counter()
                search(name, tasks[index]["question"])
                timing[name].append(1000 * (perf_counter() - start))
    rankings, records = {}, []
    for name in METHODS:
        rankings[name] = []
        for task in tasks:
            indices, scores = search(name, task["question"])
            ranked = [keys[index] for index in indices]
            rankings[name].append(ranked)
            records.append({"method": name, "query_key": task["key"], "question": task["question"],
                            "target": task["target"], "target_rank": ranked.index(task["target"]) + 1,
                            "top5": [{"key": keys[index], "title": corpus[index]["title"], "source_url": corpus[index]["source_url"],
                                      "score": float(scores[index]), "notice_links": corpus[index]["notice_links"]} for index in indices[:5]]})
    states = {"MiniLM FP32": serialize_size(fp32), "MiniLM INT8-linear": serialize_size(int8), "TF-IDF": 0}
    summaries = {}
    for name in METHODS:
        summaries[name] = {**known_item_metrics(rankings[name], targets),
                           "warm_query_mean_ms": float(np.mean(timing[name])),
                           "warm_query_p50_ms": float(np.quantile(timing[name], .5)), "warm_query_p95_ms": float(np.quantile(timing[name], .95)),
                           "timed_requests": len(timing[name]), "raw_warm_query_ms": timing[name],
                           "index_build_ms": build_ms.get(name, build_ms["TF-IDF"] + build_ms["MiniLM FP32"]),
                           "numeric_index_bytes": indexes[name].numeric_bytes if name in indexes else indexes["TF-IDF"].numeric_bytes + indexes["MiniLM FP32"].numeric_bytes,
                           "serialized_model_state_bytes": states.get(name, states["MiniLM FP32"])}
    similarities = (indexes["MiniLM FP32"].matrix * indexes["MiniLM INT8-linear"].matrix).sum(axis=1)
    top5_overlap = [len(set(a[:5]) & set(b[:5])) / 5 for a, b in zip(rankings["MiniLM FP32"], rankings["MiniLM INT8-linear"])]
    token_lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]) for text in texts]
    report = {"schema_version": 1, "snapshot_sha256": manifest["files"], "source_audit": audit_records(corpus),
              "model": {"id": MODEL_ID, "revision": REVISION, "weights_sha256": WEIGHTS_SHA, "max_wordpieces": 256,
                        "parameter_count_fp32": sum(parameter.numel() for parameter in fp32.parameters()),
                        "quantized_linear_modules": sum(isinstance(module, torch.ao.nn.quantized.dynamic.Linear) for module in int8.modules()),
                        "quantization_engine": torch.backends.quantized.engine, "load_fp32_ms": load_ms, "quantization_ms": quantize_ms},
              "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "torch_threads": 1,
                              "packages": {name: version(name) for name in ("torch", "transformers", "numpy", "scikit-learn")}},
              "input_audit": {"records_truncated_at_256_wordpieces": sum(length > 256 for length in token_lengths), "max_wordpieces_before_truncation": max(token_lengths)},
              "methods": summaries, "query_results": records,
              "precision_comparison": {"mean_document_cosine_agreement": float(similarities.mean()), "min_document_cosine_agreement": float(similarities.min()),
                                       "mean_top5_overlap_fraction": float(np.mean(top5_overlap))},
              "implementation_sha256": digest(Path(__file__).read_bytes()),
              "acquisition_and_normalization_sha256": digest((ROOT / "studies/literature_data.py").read_bytes()),
              "input_text_policy": "Title plus explicitly reusable abstract; decode presentation entities/markup at load time; neural encoders truncate at 256 wordpieces; lexical index uses the full supplied text.",
              "latency_projection": {"assumed_requests": 10000,
                  "serial_wall_seconds": {name: float(np.mean(values)) * 10000 / 1000 for name, values in timing.items()},
                  "assumptions": "Repeated same query mix and corpus, already loaded and indexed; additive observed mean wall time, no concurrency, review time, SLA, or monetary estimate."},
              "limits": ["Six authored known-item tasks with injected anchors; not complete relevance labels, a systematic review, or a general retrieval benchmark.",
                         "Scores are ranking signals, not evidence quality or clinical probabilities. No generated scientific answers.",
                         "Timing is local, warmed, single-threaded CPU query encoding plus ranking; excludes network, cold start, concurrency, and human review.",
                         "Storage measures serialized model state and numeric index arrays, not process RSS or total deployment footprint.",
                         "INT8 conversion targets Linear layers; embeddings and other operations remain floating point. No retraining or medical validation.",
                         "MiniLM is a compact text encoder, not a generative clinical LLM. Its training sources include scientific text; overlap has not been excluded."]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({name: {key: values[key] for key in ("known_target_hit_at_5", "mean_anchor_reciprocal_rank", "warm_query_p50_ms", "warm_query_p95_ms", "serialized_model_state_bytes")} for name, values in summaries.items()}, indent=2))
    return report


def search_once(query, method="TF-IDF"):
    if not query.strip() or len(query) > 2000 or method not in METHODS:
        raise ValueError("Use a nonempty query of at most 2,000 characters and a declared method.")
    torch.set_num_threads(1)
    corpus, _, manifest = load_snapshot()
    keys = [record["key"] for record in corpus]
    texts = [record["title"] + (" " + record["abstract"] if record["abstract_included"] else "") for record in corpus]
    lexical = LexicalIndex(texts, keys) if method in {"TF-IDF", "Hybrid RRF"} else None
    if method == "TF-IDF":
        ranking, score = lexical.search(query)
    else:
        tokenizer, fp32, quantized, _, _ = load_models(include_int8=method == "MiniLM INT8-linear")
        dense = DenseIndex(quantized if quantized is not None else fp32, tokenizer, texts, keys)
        ranking, score = dense.search(query)
        if method == "Hybrid RRF":
            lexical_ranking, _ = lexical.search(query)
            score = reciprocal_rank_fusion([lexical_ranking, ranking])
            ranking = ranked_indices(score, keys)
    results = [] if method == "TF-IDF" and not np.any(score > 0) else [
        {"key": corpus[index]["key"], "title": corpus[index]["title"], "authors": corpus[index]["authors"],
         "year": corpus[index]["year"], "doi": corpus[index]["doi"], "source_url": corpus[index]["source_url"],
         "score": float(score[index]), "notice_links": corpus[index]["notice_links"], "abstract_included": corpus[index]["abstract_included"]}
        for index in ranking[:5]]
    return {"query": query, "method": method, "corpus_sha256": manifest["files"]["corpus.json"],
            "results": results, "scope": "Retrieved citations, not verified scientific conclusions. This CLI rebuilds its index; benchmark timings are warm-query timings."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/evidence_retrieval.json")
    parser.add_argument("--timing-repeats", type=int, default=12)
    parser.add_argument("--query")
    parser.add_argument("--method", choices=METHODS, default="TF-IDF")
    args = parser.parse_args()
    if args.acquire:
        acquire()
    if args.download_model:
        acquire_model()
    if args.query is not None:
        print(json.dumps(search_once(args.query, args.method), indent=2, ensure_ascii=False))
    else:
        run(args.output, args.timing_repeats)
