#!/usr/bin/env python3
"""
Unitary Synthesis CX-Count Benchmark
=====================================
Compares CX gate counts produced by Qiskit, TKet, Pennylane, and BQSKit
for random unitaries on 3–10 qubits, targeting IBM Marrakesh (heavy-hex)
and IQM Garnet (square-grid) coupling maps.

Requirements
------------
    pip install qiskit qiskit-aer \
                pytket pytket-qiskit \
                pennylane pennylane-qiskit \
                bqskit \
                numpy scipy matplotlib

Usage
-----
    python unitary_benchmark.py              # full run
    python unitary_benchmark.py --plot-only  # re-plot from saved JSONs

Outputs
-------
    results_ibm_marrakesh.json
    results_iqm_garnet.json
    cx_benchmark.png / .pdf
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import multiprocessing as mp
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from scipy.stats import unitary_group


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

QUBIT_MIN = 3
QUBIT_MAX = 10
SEED = 42
POOL_LIBRARIES = ["qiskit", "tket", "pennylane"]
BQSKIT_LABEL   = "bqskit"

#Note
IQM_KEY = os.environ["IQM_KEY"]
IBM_TOKEN = os.environ["IBM_TOKEN"]
IBM_INSTANCE = os.environ["IBM_INSTANCE"]
# ═══════════════════════════════════════════════════════════════════════
#  Architecture Definitions
# ═══════════════════════════════════════════════════════════════════════


def _heavy_hex_edges() -> list[tuple[int, int]]:
    """
    27-qubit heavy-hex lattice (IBM Falcon / Heron style).

    Layout (degree ≤ 3):
        0 ─ 1 ─ 2 ─ 3 ─ 4 ─ 5 ─ 6
            │       │       │
            7       8       9
            │       │       │
       10 ─11 ─12 ─13 ─14 ─15 ─16
                │       │       │
               17      18      19
                │       │       │
       20 ─21 ─22 ─23 ─24 ─25 ─26
    """
    return [
        (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),
        (1,7),(3,8),(5,9),
        (7,11),(8,13),(9,15),
        (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),
        (12,17),(14,18),(16,19),
        (17,22),(18,24),(19,26),
        (20,21),(21,22),(22,23),(23,24),(24,25),(25,26),
    ]


def _grid_edges() -> list[tuple[int, int]]:
    """Square-grid lattice (IQM Garnet style), 4×5 = 20 qubits."""
    return [
        (0, 1), (0, 3), (1, 4), (5, 6), (5, 4), (3, 4), (3, 2),
        (6, 11), (5, 10), (4, 9), (3, 8), (2, 7),
        (11, 10), (10, 9), (9, 8), (8, 7),
        (11, 16), (10, 15), (9, 14), (8, 13), (7, 12),
        (16, 15), (15, 14), (14, 13), (13, 12),
        (15, 19), (14, 18), (13, 17),
        (19, 18), (18, 17)
    ]


ARCHITECTURES: dict[str, dict] = {
    "ibm_marrakesh": {
        "edges": _heavy_hex_edges(),
        "n_physical": 27,
        "label": "IBM Marrakesh (Heavy-Hex)",
    },
    "iqm_garnet": {
        "edges": _grid_edges(),
        "n_physical": 20,
        "label": "IQM Garnet (Square Grid)",
    },
}

# ═══════════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def _unique_edges(edges):
    s = set()
    for a, b in edges:
        s.add((min(a, b), max(a, b)))
    return sorted(s)


def _edge_set(edges):
    """Build a set of undirected edges for O(1) adjacency checks."""
    s = set()
    for a, b in edges:
        s.add((min(a, b), max(a, b)))
    return s


def generate_random_unitary(n_qubits: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + n_qubits)
    return unitary_group.rvs(2**n_qubits, random_state=rng)


def _cx_count_qiskit(qc) -> int:
    ops = qc.count_ops()
    return ops.get("cx", 0) + ops.get("CX", 0) + ops.get("cnot", 0)


# ═══════════════════════════════════════════════════════════════════════
#  Routing verification
#
#  Checks that every 2-qubit gate in the output circuit acts on a pair
#  of physical qubits that are adjacent in the coupling map.
#  Returns (n_2q_gates, n_violations).
# ═══════════════════════════════════════════════════════════════════════

def _verify_routing_qiskit(qc, edges) -> tuple[int, int]:
    allowed = _edge_set(edges)
    n_2q = 0
    violations = 0
    for instr in qc.data:
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        if len(qubits) == 2:
            n_2q += 1
            pair = (min(qubits), max(qubits))
            if pair not in allowed:
                violations += 1
    return n_2q, violations


def _verify_routing_tket(tk_circ, edges) -> tuple[int, int]:
    allowed = _edge_set(edges)
    n_2q = 0
    violations = 0
    for cmd in tk_circ.get_commands():
        qubits = [q.index[0] for q in cmd.qubits]
        if len(qubits) == 2:
            n_2q += 1
            pair = (min(qubits), max(qubits))
            if pair not in allowed:
                violations += 1
    return n_2q, violations


def _log_verify(lib: str, n: int, arch: str, n_2q: int, violations: int):
    if violations > 0:
        print(f"    ⚠ VERIFY {lib} n={n} {arch}: "
              f"{violations}/{n_2q} two-qubit gates violate coupling map!")
    else:
        print(f"    ✓ VERIFY {lib} n={n} {arch}: "
              f"all {n_2q} two-qubit gates respect coupling map")


# ═══════════════════════════════════════════════════════════════════════
#  Qiskit bare synthesis — used as input for TKet and Pennylane
#
#  This deliberately does NOT use a coupling map, so the output circuit
#  will contain CX gates between arbitrary qubit pairs.  The downstream
#  routers (TKet / Pennylane) must then route it.
# ═══════════════════════════════════════════════════════════════════════

def _qiskit_synthesise_bare(U: np.ndarray, n: int):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate

    qc = QuantumCircuit(n)
    qc.append(UnitaryGate(U), range(n))
    return transpile(
        qc,
        basis_gates=["cx", "u", "id"],
        optimization_level=3,
        seed_transpiler=SEED,
    )


def _diagnose_bare_circuit(qc, edges, lib_tag: str, n: int, arch: str):
    """Show how many CX pairs in the bare circuit already violate the
    coupling map — i.e. how much routing MUST add."""
    allowed = _edge_set(edges)
    total_cx = 0
    nonlocal_cx = 0
    for instr in qc.data:
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        if len(qubits) == 2:
            total_cx += 1
            pair = (min(qubits), max(qubits))
            if pair not in allowed:
                nonlocal_cx += 1
    print(f"    DIAG bare({lib_tag}) n={n} {arch}: "
          f"{nonlocal_cx}/{total_cx} CX are non-local "
          f"→ routing MUST insert SWAPs")


# ═══════════════════════════════════════════════════════════════════════
#  Qiskit — native synthesis + transpilation (coupling-map-aware)
# ═══════════════════════════════════════════════════════════════════════

def run_qiskit(U, n, edges, n_phys, arch_name) -> int:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.transpiler import CouplingMap

    qc = QuantumCircuit(n)
    qc.append(UnitaryGate(U), range(n))
    out = transpile(
        qc,
        coupling_map=CouplingMap(couplinglist=edges),
        basis_gates=["cx", "u", "id"],
        optimization_level=3,
        seed_transpiler=SEED,
    )
    n_2q, violations = _verify_routing_qiskit(out, edges)
    _log_verify("qiskit", n, "?", n_2q, violations)
    return _cx_count_qiskit(out)


# ═══════════════════════════════════════════════════════════════════════
#  TKet — route + optimise a pre-synthesised Qiskit circuit
#
#  FIX: FullPeepholeOptimise rebases CX → TK2 / ZZMax / ZZPhase
#  internally, so counting OpType.CX alone massively undercounts.
#  We add auto_rebase_pass back to CX after all optimisation, then
#  assert every 2-qubit gate is CX.
# ═══════════════════════════════════════════════════════════════════════

def run_tket(U, n, edges, n_phys, arch_name) -> int:
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.architecture import Architecture
    from pytket.passes import (
        DecomposeBoxes,
        FullPeepholeOptimise,
        DefaultMappingPass,
        SequencePass,
        AutoRebase,
    )
    from pytket.extensions.qiskit import IBMQBackend
    from pytket.extensions.iqm import IQMBackend
    from pytket import Circuit, OpType
    from pytket.circuit import MultiplexedRotationBox
    from pytket.passes import DecomposeBoxes
    from qiskit_ibm_runtime import QiskitRuntimeService

    if arch_name == "iqm_garnet":
        backend = IQMBackend('garnet', api_token=IQM_KEY)
    else:
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=IBM_TOKEN, instance=IBM_INSTANCE, overwrite=True)
        backend = IBMQBackend("ibm_marrakesh")
    # 1. Get bare Qiskit circuit & convert
    qc_base = _qiskit_synthesise_bare(U, n)
    tk_circ = qiskit_to_tk(qc_base)
    backend.default_compilation_pass(optimisation_level=2).apply(tk_circ)

    # 4. Count only CX (should now be the ONLY 2-qubit gate)
    cx_count = tk_circ.n_gates_of_type(OpType.CX)
    cx_count += tk_circ.n_gates_of_type(OpType.CZ)

    # 5. Sanity: count ALL 2-qubit gates to confirm rebase worked
    all_2q = sum(
        1 for cmd in tk_circ.get_commands()
        if len(cmd.qubits) == 2
    )
    if all_2q != cx_count:
        print(f"    ⚠ TKet rebase incomplete: {cx_count} CX but "
              f"{all_2q} total 2q gates  (delta = {all_2q - cx_count})")
        cx_count = all_2q  # conservative: count all 2q gates

    # 6. Verify routing
    n_2q, violations = _verify_routing_tket(tk_circ, edges)
    _log_verify("tket", n, "?", n_2q, violations)

    return cx_count


# ═══════════════════════════════════════════════════════════════════════
#  Pennylane — route a pre-synthesised Qiskit circuit
#
#  FIX: Previous approach used QNode.tape (removed in newer PL) or
#  tape-level transform that silently returned the original.
#
#  New approach:
#    1. Expand the Qiskit circuit into PennyLane ops via AnnotatedQueue.
#    2. Build a QuantumScript.
#    3. Apply qml.transforms.transpile on the script.
#    4. The transform inserts SWAP gates.  Count CNOT + 3×SWAP.
#    5. Verify routing on the output tape.
# ═══════════════════════════════════════════════════════════════════════

def run_pennylane(U, n, edges, n_phys, arch_name) -> int:
    import pennylane as qml

    qc_base = _qiskit_synthesise_bare(U, n)
    _diagnose_bare_circuit(qc_base, edges, "pennylane", n, "?")

    # Coupling map: only the n_phys-qubit device edges
    pl_cmap = [list(e) for e in _unique_edges(edges)]

    # ── Convert Qiskit → PennyLane tape ──────────────────────────
    qfunc = qml.from_qiskit(qc_base)

    with qml.queuing.AnnotatedQueue() as q:
        qfunc(wires=range(n))

    tape = qml.tape.QuantumScript(q.queue, [qml.probs(wires=range(n))])

    # ── Apply routing transform ──────────────────────────────────
    # qml.transforms.transpile returns (batch_of_tapes, postproc_fn)
    result = qml.transforms.transpile(tape, coupling_map=pl_cmap)

    if isinstance(result, tuple):
        batch, _ = result
        routed_tape = batch[0] if isinstance(batch, (list, tuple)) else batch
    else:
        # Some PL versions return a TransformProgram or single tape
        routed_tape = result[0] if hasattr(result, "__getitem__") else result

    # ── Count gates ──────────────────────────────────────────────
    cnot_count = 0
    swap_count = 0
    all_2q_pairs: list[tuple[int, int]] = []

    for op in routed_tape.operations:
        wires = [int(w) for w in op.wires]
        if op.name in ("CNOT", "CX"):
            cnot_count += 1
            if len(wires) == 2:
                all_2q_pairs.append(
                    (min(wires), max(wires))
                )
        elif op.name == "SWAP":
            swap_count += 1
            if len(wires) == 2:
                all_2q_pairs.append(
                    (min(wires), max(wires))
                )

    # Each SWAP = 3 CX in the standard decomposition
    total_cx = cnot_count + swap_count * 3

    # ── Verify routing ───────────────────────────────────────────
    allowed = _edge_set(edges)
    violations = sum(1 for p in all_2q_pairs if p not in allowed)
    _log_verify("pennylane", n, "?",
                len(all_2q_pairs), violations)

    if violations > 0:
        print(f"    ⚠ PennyLane routing failed: {violations} non-local "
              f"gates remain.  CX count is unreliable.")

    return total_cx




# ═══════════════════════════════════════════════════════════════════════
#  Multiprocessing worker (Qiskit / TKet / Pennylane)
# ═══════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "qiskit":    run_qiskit,
    "tket":      run_tket,
    "pennylane": run_pennylane,
}


def _worker(args: tuple) -> dict:
    lib, n, arch_name, edges, n_phys, U = args

    result = dict(
        library=lib, n_qubits=n, architecture=arch_name,
        cx_count=None, time_s=None, error=None,
    )

    t0 = time.perf_counter()
    try:
        cx = _DISPATCH[lib](U, n, edges, n_phys, arch_name)
        result["cx_count"] = int(cx)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    result["time_s"] = round(time.perf_counter() - t0, 2)

    tag = "OK" if result["error"] is None else "FAIL"
    print(
        f"  [{tag}] {result['library']:10s} | n={n:>2d} | "
        f"{arch_name:15s} | CX={str(result['cx_count']):>8s} | "
        f"{result['time_s']:>8.1f}s"
        + (f"  !! {result['error']}" if result["error"] else "")
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_benchmarks() -> dict:
    print("=" * 72)
    print(" UNITARY SYNTHESIS  —  CX-COUNT BENCHMARK")
    print("=" * 72)

    qubit_range = range(QUBIT_MIN, QUBIT_MAX + 1)

    # 1. Pre-generate unitaries
    unitaries: dict[int, np.ndarray] = {}
    for n in qubit_range:
        print(f"  Generating random {n}-qubit unitary  "
              f"(dim {2**n}×{2**n}) …")
        unitaries[n] = generate_random_unitary(n)

    # 2. Pool tasks (Qiskit / TKet / Pennylane)
    pool_tasks: list[tuple] = []
    for arch_name, arch in ARCHITECTURES.items():
        for n in qubit_range:
            for lib in POOL_LIBRARIES:
                pool_tasks.append((
                    lib, n, arch_name,
                    arch["edges"], arch["n_physical"],
                    unitaries[n],
                ))

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"\n  {len(pool_tasks)} pool tasks  ×  {n_workers} workers")
    print("-" * 72)

    # 3a. Parallel pool
    with mp.Pool(processes=n_workers) as pool:
        pool_results = pool.map(_worker, pool_tasks)

    # 4. Merge results
    all_libs = POOL_LIBRARIES
    # all_libs = POOL_LIBRARIES + [BQSKIT_LABEL]
    results: dict = {
        arch: {lib: {} for lib in all_libs}
        for arch in ARCHITECTURES
    }

    for r in pool_results:
        results[r["architecture"]][r["library"]][str(r["n_qubits"])] = {
            "cx_count": r["cx_count"],
            "time_s":   r["time_s"],
            "error":    r["error"],
        }


    # 5. Save JSON
    for arch_name in ARCHITECTURES:
        fname = f"results_{arch_name}.json"
        with open(fname, "w") as fh:
            json.dump(results[arch_name], fh, indent=2)
        print(f"\n  ✓  Saved  {fname}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_results(results: dict | None = None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if results is None:
        results = {}
        for arch_name in ARCHITECTURES:
            fname = f"results_{arch_name}.json"
            if not Path(fname).exists():
                print(f"  ⚠  {fname} not found — skipping.")
                continue
            with open(fname) as fh:
                results[arch_name] = json.load(fh)
        if not results:
            print("  Nothing to plot.")
            return

    # all_libs = POOL_LIBRARIES + [BQSKIT_LABEL]
    all_libs = POOL_LIBRARIES
    qubits = list(range(QUBIT_MIN, QUBIT_MAX + 1))

    COLORS  = {"qiskit": "#6929C4", "tket": "#009D9A",
               "pennylane": "#EE538B", "bqskit": "#FF832B"}
    MARKERS = {"qiskit": "o", "tket": "s", "pennylane": "^", "bqskit": "D"}

    n_arch = len(results)
    fig, axes = plt.subplots(1, n_arch, figsize=(8 * n_arch, 6),
                             squeeze=False)

    for idx, (arch_name, arch_data) in enumerate(results.items()):
        ax = axes[0][idx]
        label = ARCHITECTURES.get(arch_name, {}).get("label", arch_name)

        for lib in all_libs:
            if lib not in arch_data:
                continue
            xs, ys = [], []
            for n in qubits:
                entry = arch_data[lib].get(str(n), {})
                cx = entry.get("cx_count") if isinstance(entry, dict) else None
                if cx is not None:
                    xs.append(n)
                    ys.append(cx)
            if xs:
                ax.plot(
                    xs, ys,
                    color=COLORS[lib], marker=MARKERS[lib],
                    linewidth=2.2, markersize=8,
                    label=lib.capitalize(), alpha=0.92,
                )

        ax.set_title(label, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Number of Qubits", fontsize=12)
        ax.set_ylabel("CX Gate Count", fontsize=12)
        ax.set_xticks(qubits)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Unitary Synthesis — CX Gate Count by Transpiler",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()


    plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════

def main():
    global QUBIT_MIN, QUBIT_MAX
    QUBIT_MIN = 3
    QUBIT_MAX = 10
    parser = argparse.ArgumentParser(
        description="Benchmark unitary synthesis CX counts.",
    )
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip benchmarks; re-plot from existing JSONs.")
    parser.add_argument("--qmin", type=int, default=QUBIT_MIN)
    parser.add_argument("--qmax", type=int, default=QUBIT_MAX)
    args = parser.parse_args()

    QUBIT_MIN = args.qmin
    QUBIT_MAX = args.qmax

    if args.plot_only:
        plot_results()
    else:
        results = run_benchmarks()
        plot_results(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()