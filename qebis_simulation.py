"""
=============================================================
QEBIS - Quantum-Enhanced Battery Intelligence System
Simulation Script
=============================================================
Team Quant_Power | VTU Quantum Summit 2026 - Qubitathon
A S Harish & Keerthana Bhat | Nitte Meenakshi Institute of Technology

WHAT THIS SCRIPT DOES:
-----------------------
Since we don't have real quantum sensing hardware (NMR / NV-center)
or a physical EIS chip prototype yet, this script SIMULATES the
entire QEBIS pipeline using physics-based battery models.

Think of it this way:
  - PyBaMM = our "virtual battery" that ages realistically
  - Internal PyBaMM variables (SEI thickness, capacity loss) = 
    stand-in for what NMR/NV-center would measure in real life
  - Synthetic EIS parameters = what our ₹500 chip would measure
  - Isolation Forest ML model = the AI that detects anomalies
  - The key result: QEBIS detects degradation ~cycles earlier 
    than a classical BMS voltage threshold would

HOW TO RUN:
-----------
  pip install pybamm impedance numpy scikit-learn
  python qebis_simulation.py

The script will generate a file called 'qebis_data.json' which
the dashboard (qebis_dashboard.html) reads to display results.
=============================================================
"""

# ── Standard library ──────────────────────────────────────
import json
import math
import random

# ── Third-party libraries ─────────────────────────────────
import numpy as np
import pybamm
from sklearn.ensemble import IsolationForest

# ── Reproducibility ───────────────────────────────────────
# Setting a seed means every run gives the same random numbers,
# so your demo is consistent and reproducible.
np.random.seed(42)
random.seed(42)


# =============================================================
# STEP 1: SIMULATE BATTERY AGING WITH PyBaMM
# =============================================================
# PyBaMM (Python Battery Mathematical Modelling) is an open-source
# library that solves the actual electrochemical physics equations
# of a lithium-ion battery. We use the SPM (Single Particle Model)
# with SEI (Solid Electrolyte Interphase) growth enabled.
#
# SEI is a thin film that forms on the battery anode over time.
# As it grows, it:
#   1. Consumes lithium (reducing capacity)
#   2. Increases internal resistance
#   3. Slows down charging/discharging
# This is a major real-world degradation mechanism.

print("=" * 60)
print("QEBIS Battery Degradation Simulation")
print("=" * 60)
print()
print("STEP 1: Setting up PyBaMM battery model...")

# Create the battery physics model
# 'ec reaction limited' = a realistic SEI growth mechanism
model = pybamm.lithium_ion.SPM(options={"SEI": "ec reaction limited"})

# Chen2020 is a well-validated parameter set for a 5Ah LFP pouch cell
# (commonly used in Indian EVs - e-rickshaws, two-wheelers)
param = pybamm.ParameterValues("Chen2020")

# ── Define the charge/discharge cycle ─────────────────────
# We simulate N_CYCLES charge-discharge cycles.
# Each cycle = Discharge → Charge → Rest
# This mimics a real EV user's daily usage pattern.
N_CYCLES = 200
print(f"   Simulating {N_CYCLES} charge/discharge cycles...")
print("   (This may take 1-2 minutes. Please wait.)")

# Build the experiment: list of steps repeated N_CYCLES times
cycle_steps = [
    "Discharge at 1C until 2.5V",   # Discharge fully at moderate rate
    "Charge at 1C until 4.2V",      # Recharge fully
    "Rest for 5 minutes",           # Short rest (simulates parking time)
]
experiment = pybamm.Experiment(cycle_steps * N_CYCLES)

# Run the simulation
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
solution = sim.solve()

# Extract only the discharge steps (every 3rd cycle starting from index 0)
# The solution contains all steps: discharge, charge, rest, discharge, charge...
# We only care about the discharge because that's when the car is in use
discharge_cycles = [solution.cycles[i] for i in range(0, len(solution.cycles), 3)]
actual_cycles = len(discharge_cycles)

print(f"   ✓ Simulation complete. {actual_cycles} discharge cycles extracted.")
print()


# =============================================================
# STEP 2: EXTRACT DEGRADATION DATA (QUANTUM PROXY LABELS)
# =============================================================
# In real QEBIS, Phase 1 uses NMR to label each EIS measurement
# with the exact degradation mechanism and severity.
#
# Here, we use PyBaMM's internal physics variables as a "proxy"
# for what NMR would confirm:
#   - SEI loss capacity = proxy for SEI growth severity
#   - Plating loss      = proxy for lithium plating severity
#   - Capacity          = total usable battery capacity
#
# We then assign degradation STAGES 1-5 based on thresholds,
# exactly as QEBIS would do with real NMR labels.

print("STEP 2: Extracting degradation data (quantum-proxy labels)...")

# Storage lists
cycle_numbers    = []   # Cycle index
capacity_fade    = []   # Capacity as % of initial (100% = brand new)
sei_loss_pct     = []   # SEI capacity loss as % of initial capacity
plating_loss_pct = []   # Lithium plating loss as %
total_deg_pct    = []   # Combined degradation percentage
voltage_end      = []   # Terminal voltage at end of discharge

# Get initial (fresh battery) capacity for % calculations
initial_cap = discharge_cycles[0]["Discharge capacity [A.h]"].entries[-1]

for i, cyc in enumerate(discharge_cycles):
    # ── Extract raw values from simulation ────────────────
    cap = cyc["Discharge capacity [A.h]"].entries[-1]
    v   = cyc["Terminal voltage [V]"].entries[-1]
    sei = cyc["Loss of capacity to negative SEI [A.h]"].entries[-1]
    
    # Lithium plating - another degradation mechanism
    try:
        plating = cyc["Loss of capacity to negative lithium plating [A.h]"].entries[-1]
    except:
        plating = 0.0
    
    # ── Convert to percentages ────────────────────────────
    cap_pct        = (cap / initial_cap) * 100.0
    sei_pct        = (sei / initial_cap) * 100.0
    plating_pct    = (plating / initial_cap) * 100.0
    total_deg      = sei_pct + plating_pct
    
    cycle_numbers.append(i + 1)
    capacity_fade.append(round(cap_pct, 4))
    sei_loss_pct.append(round(sei_pct, 4))
    plating_loss_pct.append(round(plating_pct, 4))
    total_deg_pct.append(round(total_deg, 4))
    voltage_end.append(round(float(v), 4))

print(f"   ✓ Capacity fade: {capacity_fade[0]:.1f}% → {capacity_fade[-1]:.1f}%")
print(f"   ✓ SEI loss at final cycle: {sei_loss_pct[-1]:.2f}%")
print()


# =============================================================
# STEP 3: GENERATE SYNTHETIC EIS IMPEDANCE FEATURES
# =============================================================
# EIS (Electrochemical Impedance Spectroscopy) is the core
# sensing technique in QEBIS. It applies tiny AC signals at
# many frequencies and measures the battery's electrical response.
#
# The result is a "Nyquist plot" - the battery's electrochemical
# fingerprint. As the battery degrades, this fingerprint changes
# in predictable ways that classical BMS voltage sensors MISS.
#
# Key EIS parameters (Randles circuit model):
#   Rs   = Series resistance (electrolyte + current collector)
#          Increases with SEI growth and electrolyte degradation
#   Rct  = Charge transfer resistance (how easily Li+ moves)
#          Increases dramatically with SEI and plating
#   W    = Warburg coefficient (solid-state diffusion speed)
#          Changes with electrode structural changes
#
# Since we don't have real EIS hardware yet, we DERIVE these
# parameters mathematically from PyBaMM's physics variables.
# This is physically motivated - real EIS values follow the
# same trends, just with real sensor noise on top.

print("STEP 3: Generating synthetic EIS impedance features...")

eis_rs   = []  # Series resistance [Ohm]
eis_rct  = []  # Charge transfer resistance [Ohm]
eis_w    = []  # Warburg coefficient [Ohm·s^-0.5]

# Baseline values for a fresh Chen2020 battery
RS_BASE  = 0.008   # Ohm - typical for a fresh lithium-ion cell
RCT_BASE = 0.025   # Ohm
W_BASE   = 0.045   # Ohm·s^-0.5

for i in range(actual_cycles):
    sei   = sei_loss_pct[i] / 100.0    # Convert % to fraction
    plat  = plating_loss_pct[i] / 100.0
    cap   = capacity_fade[i] / 100.0   # Current capacity fraction

    # Rs grows with SEI (SEI is an electrically resistive layer)
    # Formula: physically motivated, with small realistic noise
    rs  = RS_BASE  + 0.18 * sei + 0.05 * plat
    rs  += np.random.normal(0, 0.0003)   # Sensor noise

    # Rct grows faster - charge transfer is more sensitive to surface changes
    rct = RCT_BASE + 0.65 * sei + 0.30 * plat
    rct += np.random.normal(0, 0.0008)

    # Warburg changes more slowly (structural electrode changes are gradual)
    w   = W_BASE   + 0.10 * sei + 0.04 * plat + 0.02 * (1 - cap)
    w   += np.random.normal(0, 0.0005)

    eis_rs.append(round(max(rs,  0.001), 6))
    eis_rct.append(round(max(rct, 0.001), 6))
    eis_w.append(round(max(w,   0.001), 6))

print(f"   ✓ EIS Rs:  {eis_rs[0]:.5f} → {eis_rs[-1]:.5f} Ω")
print(f"   ✓ EIS Rct: {eis_rct[0]:.5f} → {eis_rct[-1]:.5f} Ω")
print(f"   ✓ EIS W:   {eis_w[0]:.5f} → {eis_w[-1]:.5f} Ω·s^-0.5")
print()


# =============================================================
# STEP 4: ASSIGN QUANTUM-CONFIRMED DEGRADATION STAGES (1–5)
# =============================================================
# In real QEBIS Phase 1, NMR directly observes the battery's
# atomic state and assigns a degradation stage label.
#
# Here we use PyBaMM's SEI loss as the labeling oracle.
# This is the "indirect quantum inference" in simulation form.
#
# Stage definitions (based on SEI capacity loss %):
#   Stage 1: 0–2%     → Fresh, healthy battery
#   Stage 2: 2–5%     → Early SEI formation, minor capacity loss
#   Stage 3: 5–10%    → Noticeable degradation, user may notice range drop
#   Stage 4: 10–18%   → Significant degradation, service recommended
#   Stage 5: >18%     → Advanced degradation, replacement likely needed

print("STEP 4: Assigning degradation stage labels (quantum proxy)...")

degradation_stages = []

for sei in sei_loss_pct:
    if sei < 2.0:
        stage = 1
    elif sei < 5.0:
        stage = 2
    elif sei < 10.0:
        stage = 3
    elif sei < 18.0:
        stage = 4
    else:
        stage = 5
    degradation_stages.append(stage)

# Find when each stage first appears
stage_transitions = {}
for i, stage in enumerate(degradation_stages):
    if stage not in stage_transitions:
        stage_transitions[stage] = cycle_numbers[i]

print("   ✓ Stage transitions:")
for s, c in sorted(stage_transitions.items()):
    print(f"      Stage {s} reached at cycle {c}")
print()


# =============================================================
# STEP 5: CLASSICAL BMS DETECTION (VOLTAGE THRESHOLD)
# =============================================================
# A classical Battery Management System only monitors voltage.
# It raises an alert when voltage drops below a threshold:
#   - End-of-discharge voltage consistently below 2.55V
#   - Capacity fade below 80% of original (standard industry metric)
#
# We simulate this threshold-based detection here.
# This is what QEBIS is COMPARED AGAINST in the demo.

print("STEP 5: Simulating Classical BMS detection...")

BMS_VOLTAGE_THRESHOLD = 2.55   # Volts - typical BMS alert level
BMS_CAPACITY_THRESHOLD = 85.0  # % of original - trigger for EV BMS alert

classical_bms_detection_cycle = None
for i in range(actual_cycles):
    # Classical BMS detects only when capacity is significantly gone
    if capacity_fade[i] < BMS_CAPACITY_THRESHOLD:
        classical_bms_detection_cycle = cycle_numbers[i]
        break

if classical_bms_detection_cycle is None:
    classical_bms_detection_cycle = actual_cycles  # Never detected in range

print(f"   ✓ Classical BMS would alert at cycle: {classical_bms_detection_cycle}")
print()


# =============================================================
# STEP 6: QEBIS ANOMALY DETECTION (ISOLATION FOREST)
# =============================================================
# QEBIS runs an AI anomaly detection model on the EIS features.
# We use Isolation Forest - a lightweight algorithm ideal for
# edge devices (Raspberry Pi) that doesn't need labels to work.
#
# How Isolation Forest works:
#   - "Normal" data points are harder to isolate (need more splits)
#   - "Anomalous" data points are easy to isolate (need fewer splits)
#   - It returns an anomaly score: lower = more anomalous
#
# We train on the first 20% of cycles (fresh battery behavior),
# then run the trained model on all cycles to get anomaly scores.
# The model learns what "healthy" impedance looks like, then
# flags deviations - even subtle ones that voltage can't catch.

print("STEP 6: Running QEBIS anomaly detection (Isolation Forest)...")

# Feature matrix: stack all three EIS parameters as columns
# Shape: (N_CYCLES, 3)
X = np.column_stack([eis_rs, eis_rct, eis_w])

# Train on the first 20% of cycles (fresh battery = "normal" behavior)
train_size = max(10, actual_cycles // 5)
X_train = X[:train_size]

# Fit the Isolation Forest
# contamination=0.05 means we expect about 5% of training data might
# have slight anomalies (conservative assumption)
iso_forest = IsolationForest(
    n_estimators=200,       # More trees = more stable predictions
    contamination=0.05,     # Expected fraction of anomalies in training
    random_state=42         # Reproducibility
)
iso_forest.fit(X_train)

# Get anomaly scores for ALL cycles
# score_samples returns log-likelihood; more negative = more anomalous
raw_scores = iso_forest.score_samples(X)

# Normalize scores to a 0-100 "health score" scale
# 100 = perfectly healthy, 0 = severely anomalous
score_min = raw_scores.min()
score_max = raw_scores.max()
health_scores = ((raw_scores - score_min) / (score_max - score_min)) * 100.0
health_scores = health_scores.tolist()

# QEBIS alert threshold: health score drops below 65
# (This means the EIS pattern deviates meaningfully from healthy baseline)
QEBIS_ALERT_THRESHOLD = 65.0

qebis_detection_cycle = None
for i in range(actual_cycles):
    if health_scores[i] < QEBIS_ALERT_THRESHOLD:
        qebis_detection_cycle = cycle_numbers[i]
        break

if qebis_detection_cycle is None:
    qebis_detection_cycle = actual_cycles

print(f"   ✓ QEBIS would alert at cycle: {qebis_detection_cycle}")
print(f"   ✓ QEBIS detects {classical_bms_detection_cycle - qebis_detection_cycle} cycles EARLIER than classical BMS")
print()


# =============================================================
# STEP 7: GENERATE NYQUIST PLOT DATA
# =============================================================
# The Nyquist plot is the key visualisation for EIS.
# X-axis = Real part of impedance (Z_real)
# Y-axis = -Imaginary part of impedance (Z_imag)
#
# A healthy battery shows a small semicircle (Rct) followed
# by a straight line (Warburg diffusion tail).
# A degraded battery shows a LARGER semicircle and steeper tail.
#
# We generate Nyquist data for 4 representative cycles:
# fresh (cycle 1), early (25%), mid (50%), late (75%)

print("STEP 7: Generating Nyquist plot data...")

# Frequency range for EIS sweep: 100kHz down to 0.1Hz (log scale)
# This is the same range a real EIS instrument would use
freqs = np.logspace(5, -1, 60)  # 60 frequency points

def compute_nyquist(Rs, Rct, W_coeff, frequencies):
    """
    Compute Randles circuit impedance at each frequency.
    
    The Randles circuit is the standard model for battery impedance:
      Z_total = Rs + (Rct || Zw)   [parallel combination]
    
    Where:
      Rs   = series (electrolyte) resistance
      Rct  = charge transfer resistance
      Zw   = Warburg diffusion impedance = W / sqrt(jω)
      j    = imaginary unit, ω = angular frequency = 2π*f
    """
    z_real_list = []
    z_imag_list = []
    
    for f in frequencies:
        omega = 2 * math.pi * f
        
        # Warburg impedance: Z_W = W * (1-j) / sqrt(2*omega)
        z_w_real =  W_coeff / math.sqrt(2 * omega)
        z_w_imag = -W_coeff / math.sqrt(2 * omega)
        
        # Parallel combination of Rct and Zw
        # Z_parallel = (Rct * Zw) / (Rct + Zw)  [complex arithmetic]
        num_real = Rct * z_w_real - 0 * z_w_imag   # numerator real part
        num_imag = Rct * z_w_imag + 0 * z_w_real   # numerator imag part
        den_real = (Rct + z_w_real)
        den_imag = z_w_imag
        den_sq   = den_real**2 + den_imag**2
        
        z_par_real = (num_real * den_real + num_imag * den_imag) / den_sq
        z_par_imag = (num_imag * den_real - num_real * den_imag) / den_sq
        
        # Total impedance Z = Rs + Z_parallel
        z_total_real = Rs + z_par_real
        z_total_imag = z_par_imag
        
        # Nyquist convention: plot -Z_imag vs Z_real
        z_real_list.append(round(z_total_real, 7))
        z_imag_list.append(round(-z_total_imag, 7))
    
    return z_real_list, z_imag_list

# Representative cycle indices: 0%, 33%, 66%, 100% through the simulation
nyquist_indices = [
    0,
    actual_cycles // 3,
    (2 * actual_cycles) // 3,
    actual_cycles - 1,
]

nyquist_plots = []
for idx in nyquist_indices:
    z_r, z_i = compute_nyquist(
        eis_rs[idx], eis_rct[idx], eis_w[idx], freqs
    )
    nyquist_plots.append({
        "cycle":  cycle_numbers[idx],
        "stage":  degradation_stages[idx],
        "z_real": z_r,
        "z_imag": z_i,
        "label":  f"Cycle {cycle_numbers[idx]} (Stage {degradation_stages[idx]})"
    })

print(f"   ✓ Nyquist plots generated for cycles: {[p['cycle'] for p in nyquist_plots]}")
print()


# =============================================================
# STEP 8: PACKAGE DATA FOR DASHBOARD
# =============================================================
# We save all simulation results as a JSON file.
# The dashboard (qebis_dashboard.html) reads this file and
# renders interactive charts - no Python needed to view it.

print("STEP 8: Packaging results for dashboard...")

output = {
    # ── Metadata ──────────────────────────────────────────
    "metadata": {
        "title": "QEBIS Battery Degradation Simulation",
        "model": "PyBaMM SPM + SEI (ec reaction limited)",
        "parameters": "Chen2020 (5Ah LFP pouch cell)",
        "n_cycles": actual_cycles,
        "initial_capacity_ah": round(initial_cap, 4),
        "description": (
            "Simulation of EV battery degradation over charge-discharge cycles. "
            "EIS parameters are physics-derived proxies for real sensor data. "
            "Degradation stage labels are quantum-proxy labels from PyBaMM "
            "internal variables, standing in for NMR/NV-center measurements."
        )
    },

    # ── Per-cycle time series data ─────────────────────────
    "cycles": {
        "cycle_number":        cycle_numbers,
        "capacity_pct":        capacity_fade,
        "sei_loss_pct":        sei_loss_pct,
        "plating_loss_pct":    plating_loss_pct,
        "total_degradation":   total_deg_pct,
        "voltage_end":         voltage_end,
        "eis_rs":              eis_rs,
        "eis_rct":             eis_rct,
        "eis_warburg":         eis_w,
        "health_score":        [round(h, 2) for h in health_scores],
        "degradation_stage":   degradation_stages,
    },

    # ── Detection comparison ───────────────────────────────
    "detection": {
        "qebis_alert_cycle":   qebis_detection_cycle,
        "bms_alert_cycle":     classical_bms_detection_cycle,
        "cycles_earlier":      classical_bms_detection_cycle - qebis_detection_cycle,
        "qebis_threshold":     QEBIS_ALERT_THRESHOLD,
        "bms_cap_threshold":   BMS_CAPACITY_THRESHOLD,
        "qebis_capacity_at_alert": round(
            capacity_fade[qebis_detection_cycle - 1], 2
        ) if qebis_detection_cycle <= actual_cycles else None,
        "bms_capacity_at_alert": round(
            capacity_fade[min(classical_bms_detection_cycle - 1, actual_cycles - 1)], 2
        ),
    },

    # ── Stage transition info ──────────────────────────────
    "stage_transitions": {str(k): v for k, v in sorted(stage_transitions.items())},

    # ── Nyquist plot data ──────────────────────────────────
    "nyquist": nyquist_plots,
}

# Write to JSON file
output_path = "qebis_data.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"   ✓ Data saved to '{output_path}'")
print()


# =============================================================
# FINAL SUMMARY
# =============================================================
print("=" * 60)
print("SIMULATION COMPLETE — RESULTS SUMMARY")
print("=" * 60)
print()
print(f"  Battery modeled : 5Ah LFP pouch cell (Chen2020)")
print(f"  Cycles simulated: {actual_cycles}")
print(f"  Initial capacity: {initial_cap:.3f} Ah")
print(f"  Final capacity  : {discharge_cycles[-1]['Discharge capacity [A.h]'].entries[-1]:.3f} Ah")
print(f"  Capacity fade   : {100 - capacity_fade[-1]:.1f}% lost over {actual_cycles} cycles")
print()
print(f"  ┌─────────────────────────────────────────────────┐")
print(f"  │ DETECTION COMPARISON                           │")
print(f"  │                                                │")
print(f"  │  Classical BMS alerts at : Cycle {classical_bms_detection_cycle:<5}          │")
print(f"  │  QEBIS alerts at         : Cycle {qebis_detection_cycle:<5}          │")
print(f"  │  QEBIS advantage         : {classical_bms_detection_cycle - qebis_detection_cycle} cycles earlier  │")
print(f"  │                                                │")
print(f"  │  At QEBIS alert → battery still at            │")
det_cap = output['detection']['qebis_capacity_at_alert']
if det_cap:
    print(f"  │  {det_cap:.1f}% capacity (preventive maintenance window) │")
print(f"  └─────────────────────────────────────────────────┘")
print()
print(f"  ✓ Open 'qebis_dashboard.html' to view the demo")
print("=" * 60)
