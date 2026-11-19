# Q4 - Bus maintenance: ML + Quantum + Parallel Computing

Roll No. 241070015 (random seed = 15)

Choose exactly 4 of 10 buses for today's 4 maintenance slots.

| Step | Domain | What it does |
|---|---|---|
| (a) | Machine Learning | Logistic Regression on 600 synthetic rows (70/30 split) predicts each bus's breakdown probability |
| (b) | QUBO | E(x) = -sum p_i x_i + P (sum x_i - 4)^2, Q matrix built from the predictions, checked by brute force |
| (c) | Quantum | QAOA (p = 2) on the Qiskit Aer simulator, compared with brute force |
| (d) | Parallel | Energy of all 2^10 bit-strings evaluated with ProcessPoolExecutor on 1, 2, 4 workers (speed-up, efficiency) |
| Baseline | - | Greedy top-4, highest-mileage rule (no ML), random pick |

## Run

```
pip install -r requirements.txt
python q4_bus_maintenance.py
```

Or double-click `run_q4.bat`. Run it as a .py file, not in Jupyter.

## Submit
- q4_bus_maintenance.py
- Screenshots: model results, Q matrix, QAOA result, speed-up table
- 3-line conclusion (printed at the end of the output)
