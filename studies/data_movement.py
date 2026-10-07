"""Check CPU buffer sharing and copy boundaries instead of claiming universal zero-copy."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np

from studies.data import ROOT


def mapped_checksum(path, channel):
    try:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        channel.send({"sum": float(data[:, 0].sum(dtype=np.float64)), "rows": len(data)})
    finally:
        channel.close()


def inspect_buffers(path):
    import torch

    mapped = np.load(path, mmap_mode="c", allow_pickle=False)
    if mapped.dtype != np.float32 or mapped.ndim != 2 or len(mapped) < 4 or mapped.shape[1] < 1:
        raise ValueError("The example requires a two-dimensional float32 array with at least four rows.")
    view = mapped[1:4]
    selected = mapped[[1, 2, 3]]
    converted = view.astype(np.float64, copy=False)
    tensor = torch.from_numpy(view)
    original = float(view[0, 0])
    tensor[0, 0] = original + 100
    disk = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"basic_slice_shares_memory": bool(np.shares_memory(mapped, view)),
            "advanced_index_shares_memory": bool(np.shares_memory(mapped, selected)),
            "dtype_conversion_shares_memory": bool(np.shares_memory(view, converted)),
            "tensor_shares_cpu_pointer": tensor.data_ptr() == view.__array_interface__["data"][0],
            "tensor_mutation_visible_in_view": float(view[0, 0]) == original + 100,
            "copy_on_write_preserves_disk": float(disk[1, 0]) == original,
            "mode": "copy-on-write; CPU only; no GPU transfer or cross-host shared address space"}


def run(output=ROOT / "studies/results/data_movement.json", rows=1000000):
    if type(rows) is not int or not 4 <= rows <= 2000000:
        raise ValueError("Use 4–2,000,000 rows for this bounded memory exercise.")
    with tempfile.TemporaryDirectory(prefix="curious-mapping-") as temporary:
        path = Path(temporary) / "synthetic.npy"
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, 2))
        for offset in range(0, rows, 8192):
            indices = np.arange(offset, min(rows, offset + 8192), dtype=np.float32)
            mapped[offset:offset + len(indices), 0] = indices
            mapped[offset:offset + len(indices), 1] = np.sin(indices * .001)
        mapped.flush()
        del mapped
        contracts = inspect_buffers(path)
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=mapped_checksum, args=(str(path), child))
        started = perf_counter()
        process.start()
        child.close()
        try:
            if not parent.poll(30):
                process.terminate()
                raise TimeoutError("Mapped reader did not complete within the bounded demonstration.")
            result = parent.recv()
            process.join(timeout=5)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("Mapped reader did not terminate successfully.")
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
            process.join()
        elapsed = 1000 * (perf_counter() - started)
        expected = rows * (rows - 1) / 2
        if result["sum"] != expected:
            raise AssertionError("Cross-process reduction differs from the analytic sum.")
        report = {"evidence_type": "synthetic float32 array; two columns, no medical images or patient records", "shape": [rows, 2],
                  "numeric_bytes": rows * 2 * 4, "contracts": contracts, "mapped_child_result": result,
                  "analytic_sum": expected, "spawn_map_reduce_wall_ms": elapsed,
                  "environment": {"system": platform.system(), "machine": platform.machine(), "numpy": np.__version__},
                  "limits": "IPC sends a path and small result, not the array. Page faults, filesystem cache, process startup, dtype conversion, private copy-on-write pages, and device transfer are not free."}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/data_movement.json")
    parser.add_argument("--rows", type=int, default=1000000)
    args = parser.parse_args()
    run(args.output, args.rows)
