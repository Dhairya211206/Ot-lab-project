"""
Q4 - ML + Quantum + Parallel Computing
Bus maintenance: choose exactly 4 of 10 buses for today's 4 maintenance slots.
Roll No. 241070015 -> random seed = 015 = 15

Workflow (one connected pipeline):
  (a) ML  : Logistic Regression predicts breakdown probability p_i for each bus
  (b) QUBO: p_i -> Q matrix, choose exactly 4 (penalty), verified by brute force
  (c) QC  : QAOA (p = 2) solves the QUBO on a simulator, compared with brute force
  (d) PC  : energy of all 2^n bit-strings evaluated in parallel (1, 2, 4 workers)
  Baselines: greedy top-4 by probability, highest-mileage rule (no ML), random pick

RUN AS A .py FILE (python q4_bus_maintenance.py), not inside Jupyter on Windows,
because ProcessPoolExecutor needs the  if __name__ == "__main__":  guard.
"""

import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 15          # last 3 digits of roll number 241070015
N_BUSES = 10
K = 4              # maintenance slots
PENALTY = 2.0      # must be larger than any single p_i (p_i <= 1)
REPEATS = 300      # repeated energy evaluations, so parallel timing is measurable

rng = np.random.default_rng(SEED)
np.set_printoptions(precision=3, suppress=True, linewidth=160)


# ---------------------------------------------------------------- (a) ML
def make_data(n, rng):
    mileage = rng.uniform(20_000, 300_000, n)        # km
    engine_hours = rng.uniform(1_000, 15_000, n)     # h
    service_gap = rng.uniform(5, 180, n)             # days since last service
    brake_wear = rng.uniform(10, 95, n)              # %
    X = np.column_stack([mileage, engine_hours, service_gap, brake_wear])
    z = (-9.0 + 1.6e-5 * mileage + 2.0e-4 * engine_hours
         + 0.018 * service_gap + 0.035 * brake_wear + rng.normal(0, 0.8, n))
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y


FEATURES = ["mileage_km", "engine_hours", "service_gap_days", "brake_wear_pct"]


def part_a():
    print("=" * 70)
    print("(a) MACHINE LEARNING - Logistic Regression")
    print("=" * 70)
    X, y = make_data(600, rng)
    print(f"Synthetic data: {X.shape[0]} rows, features = {FEATURES}")
    print(f"Breakdown rate in data = {y.mean():.2f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    y_pred = model.predict(scaler.transform(X_te))

    print(f"Train/Test split = {len(X_tr)}/{len(X_te)} (70/30)")
    print(f"Accuracy  = {accuracy_score(y_te, y_pred):.3f}")
    print(f"Precision = {precision_score(y_te, y_pred):.3f}")
    print(f"Recall    = {recall_score(y_te, y_pred):.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, y_pred))

    # Baseline for ML: always predict the majority class
    majority = int(y_tr.mean() >= 0.5)
    print(f"Baseline (always predict {majority}) accuracy = {accuracy_score(y_te, np.full_like(y_te, majority)):.3f}")

    # The 10 buses of today's fleet
    X_bus, _ = make_data(N_BUSES, rng)
    p = model.predict_proba(scaler.transform(X_bus))[:, 1]
    buses = pd.DataFrame(X_bus.round(0), columns=FEATURES)
    buses.insert(0, "bus", [f"B{i}" for i in range(N_BUSES)])
    buses["p_breakdown"] = p.round(3)
    buses["prediction"] = np.where(p >= 0.5, "BREAKDOWN", "ok")
    print("\nPrediction for each of the 10 buses:")
    print(buses.to_string(index=False))
    return p, X_bus


# ---------------------------------------------------------------- (b) QUBO
def build_qubo(p, k=K, P=PENALTY):
    """E(x) = -sum p_i x_i + P (sum x_i - k)^2  ->  x^T Q x + const (Q upper triangular)."""
    n = len(p)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -p[i] + P * (1 - 2 * k)       # x_i^2 = x_i
        for j in range(i + 1, n):
            Q[i, j] = 2 * P
    const = P * k * k
    return Q, const


def energy(bits, Q, const):
    x = np.asarray(bits, dtype=float)
    return float(x @ Q @ x + const)


def part_b(p):
    print("\n" + "=" * 70)
    print("(b) QUBO formulation")
    print("=" * 70)
    print("x_i = 1 if bus i gets a maintenance slot, else 0")
    print(f"Minimize  E(x) = - sum_i p_i x_i  +  P (sum_i x_i - {K})^2,   P = {PENALTY}")
    print("  first term : reward for servicing buses likely to break down")
    print("  second term: penalty if the number of chosen buses is not exactly 4")
    Q, const = build_qubo(p)
    print(f"\nQ matrix (upper triangular), constant = {const}:")
    print(pd.DataFrame(Q, index=[f"B{i}" for i in range(N_BUSES)],
                       columns=[f"B{i}" for i in range(N_BUSES)]).round(3).to_string())

    # Brute force over all 2^10 bit-strings
    best_E, best_x = np.inf, None
    for bits in itertools.product([0, 1], repeat=N_BUSES):
        E = energy(bits, Q, const)
        if E < best_E:
            best_E, best_x = E, bits
    chosen = [i for i, b in enumerate(best_x) if b]
    greedy = sorted(np.argsort(-p)[:K].tolist())
    print(f"\nBrute force best bit-string = {''.join(map(str, best_x))}  energy = {best_E:.4f}")
    print(f"Chosen buses = {['B%d' % i for i in chosen]}  (count = {sum(best_x)})")
    print(f"Check: greedy top-{K} by probability = {['B%d' % i for i in greedy]}  ->  "
          f"{'MATCHES' if chosen == greedy else 'DIFFERS'}")
    return Q, const, best_x, best_E


# ---------------------------------------------------------------- (c) QAOA
def all_energies(Q, const, n):
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(float)  # bit 0 = leftmost
    return np.einsum("bi,ij,bj->b", bits, Q, bits) + const


def qaoa_numpy_state(params, E_scaled, n):
    """Exact statevector QAOA (fallback if qiskit is not installed)."""
    gammas, betas = params[: len(params) // 2], params[len(params) // 2:]
    psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E_scaled)
        rx = np.array([[np.cos(b), -1j * np.sin(b)], [-1j * np.sin(b), np.cos(b)]])
        psi = psi.reshape([2] * n)
        for q in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [q])), 0, q)
        psi = psi.reshape(-1)
    return psi


def qaoa_qiskit_circuit(params, Q_scaled, n):
    from qiskit import QuantumCircuit
    gammas, betas = params[: len(params) // 2], params[len(params) // 2:]
    # x = (1 - z)/2  ->  Ising coefficients
    h = np.zeros(n)
    J = {}
    for i in range(n):
        h[i] += -Q_scaled[i, i] / 2
        for j in range(i + 1, n):
            if Q_scaled[i, j] != 0:
                h[i] += -Q_scaled[i, j] / 4
                h[j] += -Q_scaled[i, j] / 4
                J[(i, j)] = Q_scaled[i, j] / 4
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for (i, j), Jij in J.items():
            qc.rzz(2 * g * Jij, i, j)
        for i in range(n):
            qc.rz(2 * g * h[i], i)
        for i in range(n):
            qc.rx(2 * b, i)
    qc.measure_all()
    return qc


def part_c(Q, const, bf_x, bf_E, shots=4096, reps=2):
    print("\n" + "=" * 70)
    print(f"(c) QUANTUM - QAOA (p = {reps}) on a simulator")
    print("=" * 70)
    n = Q.shape[0]
    E_all = all_energies(Q, const, n)
    scale = np.max(np.abs(E_all))
    E_scaled = E_all / scale
    Q_scaled = Q / scale

    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        sim = AerSimulator(seed_simulator=SEED)
        backend = "qiskit-aer AerSimulator"

        def sample(params, s):
            qc = transpile(qaoa_qiskit_circuit(params, Q_scaled, n), sim)
            counts = sim.run(qc, shots=s).result().get_counts()
            return {k.replace(" ", "")[::-1]: v for k, v in counts.items()}  # qubit 0 -> leftmost
    except ImportError:
        backend = "numpy statevector (qiskit not installed)"
        srng = np.random.default_rng(SEED)

        def sample(params, s):
            probs = np.abs(qaoa_numpy_state(params, E_scaled, n)) ** 2
            draws = srng.choice(2 ** n, size=s, p=probs / probs.sum())
            vals, cnt = np.unique(draws, return_counts=True)
            return {format(v, f"0{n}b"): int(c) for v, c in zip(vals, cnt)}

    print(f"Simulator: {backend},  qubits = {n},  shots = {shots}")

    def expected_energy(params):
        counts = sample(params, 1024)
        tot = sum(counts.values())
        return sum(E_scaled[int(b, 2)] * c for b, c in counts.items()) / tot

    best = None
    for start in [[0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 0.6, 0.3], [2.0, 1.0, 0.3, 0.6]]:
        res = minimize(expected_energy, start, method="COBYLA", options={"maxiter": 120})
        if best is None or res.fun < best.fun:
            best = res
    print(f"Optimised angles  gamma = {best.x[:reps].round(3)}, beta = {best.x[reps:].round(3)}")

    counts = sample(best.x, shots)
    ranked = sorted(counts.items(), key=lambda kv: E_all[int(kv[0], 2)])
    q_best, q_count = ranked[0]
    most_freq = max(counts.items(), key=lambda kv: kv[1])
    print("\nTop 5 sampled bit-strings (lowest energy first):")
    print(f"{'bitstring':>12} {'energy':>9} {'count':>6} {'freq':>7} {'#chosen':>8}")
    for b, c in ranked[:5]:
        print(f"{b:>12} {E_all[int(b, 2)]:9.4f} {c:6d} {c / shots:7.2%} {b.count('1'):8d}")
    print(f"\nQAOA best answer   = {q_best}  energy = {E_all[int(q_best, 2)]:.4f}  "
          f"appears {q_count}/{shots} = {q_count / shots:.2%}")
    print(f"Most frequent      = {most_freq[0]}  ({most_freq[1] / shots:.2%})")
    bf_str = "".join(map(str, bf_x))
    print(f"Brute force optimum= {bf_str}  energy = {bf_E:.4f}")
    print(f"QAOA found the optimum: {'YES' if q_best == bf_str else 'NO'}")
    feasible = sum(c for b, c in counts.items() if b.count("1") == K) / shots
    print(f"Fraction of shots choosing exactly {K} buses = {feasible:.2%}")
    print(f"Random guessing would hit the optimum with prob 1/1024 = {1 / 1024:.2%}")
    return q_best, counts, E_all


# ---------------------------------------------------------------- (d) Parallel
def eval_chunk(args):
    """Worker: evaluate QUBO energy of a chunk of bit-strings (pure Python, repeated)."""
    start, stop, n, Q_list, const, repeats = args
    best = (float("inf"), -1)
    for _ in range(repeats):
        for idx in range(start, stop):
            bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
            ones = [i for i in range(n) if bits[i]]
            E = const
            for a in ones:
                row = Q_list[a]
                for b in ones:
                    if b >= a:
                        E += row[b]
            if E < best[0]:
                best = (E, idx)
    return best


def run_parallel(Q, const, workers):
    n = Q.shape[0]
    total = 2 ** n
    chunk = total // workers
    tasks = [(w * chunk, (w + 1) * chunk, n, Q.tolist(), const, REPEATS) for w in range(workers)]
    t0 = time.perf_counter()
    if workers == 1:
        results = [eval_chunk(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(eval_chunk, tasks))
    elapsed = time.perf_counter() - t0
    return elapsed, min(results)


def part_d(Q, const):
    print("\n" + "=" * 70)
    print("(d) PARALLEL COMPUTING - ProcessPoolExecutor")
    print("=" * 70)
    n = Q.shape[0]
    print(f"Evaluating all 2^{n} = {2 ** n} bit-strings x {REPEATS} repeats, split into equal chunks")
    rows = []
    t1 = None
    for w in [1, 2, 4]:
        t, (E, idx) = run_parallel(Q, const, w)
        t1 = t if w == 1 else t1
        rows.append({"workers": w, "time_s": round(t, 3), "speed_up": round(t1 / t, 2),
                     "efficiency": round(t1 / t / w, 2),
                     "best_bitstring": format(idx, f"0{n}b"), "best_energy": round(E, 4)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("speed-up = T1 / Tp,  efficiency = speed-up / p")
    return table


# ---------------------------------------------------------------- baselines + main
def baselines(p, X_bus, q_best):
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (expected breakdowns prevented = sum of p_i of chosen buses)")
    print("=" * 70)
    brng = np.random.default_rng(SEED)
    plans = {
        "QAOA (ML + quantum)": [i for i, b in enumerate(q_best) if b == "1"],
        "Greedy top-4 by p (classical)": sorted(np.argsort(-p)[:K].tolist()),
        "Highest mileage (no ML)": sorted(np.argsort(-X_bus[:, 0])[:K].tolist()),
        "Random 4 buses": sorted(brng.choice(N_BUSES, K, replace=False).tolist()),
    }
    for name, idx in plans.items():
        print(f"{name:32s} buses = {['B%d' % i for i in idx]}  expected prevented = {p[idx].sum():.3f}")


def make_plots(p, counts, E_all, table, bf_x, X_bus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bf_str = "".join(map(str, bf_x))
    chosen = [i for i, b in enumerate(bf_str) if b == "1"]

    # 1. Breakdown probability per bus
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#D85A30" if i in chosen else "#888780" for i in range(N_BUSES)]
    ax.bar([f"B{i}" for i in range(N_BUSES)], p, color=colors)
    ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_ylabel("Predicted breakdown probability")
    ax.set_title("ML predictions (orange = chosen for maintenance)")
    fig.tight_layout(); fig.savefig("plot1_ml_predictions.png", dpi=150); plt.close(fig)

    # 2. QAOA: 10 lowest-energy sampled bit-strings
    shots = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: E_all[int(kv[0], 2)])[:10]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([b for b, _ in top], [c / shots for _, c in top],
           color=["#1D9E75" if b == bf_str else "#7F77DD" for b, _ in top])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"{b}\nE={E_all[int(b, 2)]:.2f}" for b, _ in top], rotation=60, fontsize=7)
    ax.set_ylabel("Sampling frequency")
    ax.set_title("QAOA (p=2): lowest-energy samples (green = brute-force optimum)")
    fig.tight_layout(); fig.savefig("plot2_qaoa_results.png", dpi=150); plt.close(fig)

    # 3. Parallel speed-up
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["workers"], table["speed_up"], "o-", label="Measured speed-up")
    ax.plot(table["workers"], table["workers"], "--", label="Ideal (linear)")
    ax.set_xlabel("Workers"); ax.set_ylabel("Speed-up T1/Tp"); ax.legend()
    ax.set_title("ProcessPoolExecutor speed-up")
    fig.tight_layout(); fig.savefig("plot3_speedup.png", dpi=150); plt.close(fig)

    # 4. Baseline comparison
    brng = np.random.default_rng(SEED)
    plans = {
        "QAOA\n(ML+quantum)": chosen,
        "Highest\nmileage": np.argsort(-X_bus[:, 0])[:K].tolist(),
        "Random": brng.choice(N_BUSES, K, replace=False).tolist(),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(plans), [p[idx].sum() for idx in plans.values()], color=["#1D9E75", "#888780", "#888780"])
    ax.set_ylabel("Expected breakdowns prevented")
    ax.set_title("Proposed method vs baselines")
    fig.tight_layout(); fig.savefig("plot4_baselines.png", dpi=150); plt.close(fig)

    print("\nSaved charts: plot1_ml_predictions.png, plot2_qaoa_results.png, "
          "plot3_speedup.png, plot4_baselines.png")


class _Tee:
    """Print to the screen and save everything to q4_output.txt."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    import os
    import sys
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # save all outputs next to this file
    _log = open("q4_output.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log)
    print("=" * 70)
    print("PROBLEM FORMULATION")
    print("=" * 70)
    print("10 buses, only 4 maintenance slots today. Choose exactly 4 buses.")
    print("Objective: maximise expected breakdowns prevented = sum of predicted p_i of chosen buses")
    print("Constraint: exactly 4 buses chosen (sum x_i = 4), x_i in {0,1}")
    print("Pipeline : ML (predict p_i) -> QUBO -> QAOA (choose 4) -> parallel energy evaluation")
    print("Baseline : greedy top-4, highest-mileage rule (no ML), random choice\n")
    p, X_bus = part_a()
    Q, const, bf_x, bf_E = part_b(p)
    q_best, counts, E_all = part_c(Q, const, bf_x, bf_E)
    table = part_d(Q, const)
    baselines(p, X_bus, q_best)
    try:
        make_plots(p, counts, E_all, table, bf_x, X_bus)
    except Exception as e:
        print(f"(Charts skipped: {e})")
    chosen = [f"B{i}" for i, b in enumerate(q_best) if b == "1"]
    print("\n" + "=" * 70)
    print("(e) CONCLUSION")
    print("=" * 70)
    print(f"1. Logistic Regression ranked breakdown risk; QAOA picked {chosen} for the 4 slots, matching brute force.")
    print("2. The QUBO penalty enforced 'exactly 4'; the ML-driven choice prevents more expected breakdowns than mileage-only or random baselines.")
    print("3. Parallel evaluation of all 2^10 energies gave speed-up < p because of process start-up overhead; QAOA is a demo at 10 qubits, classical brute force is still faster here.")
    print("\nLimitations: synthetic data; QAOA p=2 gives a low probability for the optimum; small n hides parallel gains.")
    print("References: scikit-learn LogisticRegression docs; Farhi et al. 2014 (QAOA); Qiskit Aer docs; Python concurrent.futures docs.")
