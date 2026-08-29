# JIS-SSI-FL: Self-Sovereign Identity for Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hardhat](https://img.shields.io/badge/built%20with-Hardhat-FFDB1C.svg)](https://hardhat.org/)
[![JIS](https://img.shields.io/badge/submitted-JIS-orange)](https://journals-sol.sbc.org.br/index.php/jisa)

> Official implementation accompanying: **"An Identity-First Software Architecture for Secure Federated Learning in Healthcare: Integrating Self-Sovereign Identity with Blockchain-Based Access Control"** — submitted to the *Journal on Interactive Systems (JIS)*.

## Overview

This repository implements a decentralized identity verification protocol for Federated Learning (FL) in Internet-distributed environments. By integrating W3C Self-Sovereign Identity (SSI) standards (DIDs, VCs) with blockchain smart contracts, the protocol provides deterministic Sybil-resistance through proactive institutional identity verification rather than reactive reputation scoring.

### Key Features

- **100% external Sybil attack neutralization**: Only DIDs with valid Verifiable Credentials can participate
- **Domain-agnostic protocol**: Transferable beyond healthcare to any Internet service requiring authenticated collaborative ML
- **Non-IID federation**: 10 hospital clients partitioned via a Dirichlet distribution (alpha=0.5), each fold independently rebalanced with SMOTETomek
- **Four-track evaluation**: every run compares `proposed` (on-chain identity verification), `pki` (equivalent off-chain allowlist, no blockchain), `reputation` (reactive trust-score defense, the class critiqued as prior art), and `baseline` (no defense) — all from real simulation runs, not synthetic baselines, so both the Sybil-resistance claim and what specifically blockchain contributes on top of "identity-first" are directly measurable
- **< 1% protocol overhead**: blockchain verification adds negligible latency relative to local training
- **LGPD compliance mapping**: First structured analysis of Brazilian data protection law across FL trust models

### Dataset setup notice

**No dataset ships with this repository.** MIMIC-IV is credentialed-access data under a PhysioNet Data Use Agreement and must never be committed to a public repo. `data/` is empty except for instructions; generate `data/mortalidade_features.csv` yourself with [`sql/extract_cohort.sql`](sql/extract_cohort.sql) — see [`data/README.md`](data/README.md) for the exact steps. That query includes `admittime` (needed for the paper's chronological 90/10 split), Charlson Comorbidity Index, length of stay, and ICD-10 sepsis/heart-failure flags — the full feature set described in Sec. 4.1.

## Repository Structure

```
JIS-SSI-FL/
├── contracts/                   # Ethereum Smart Contracts
│   └── FLRegistry.sol          # DID/VC verification + access control
├── scripts/                     # Deployment and utilities
│   └── deploy.js               # Contract deployment script
├── src/                         # Federated Learning Core
│   ├── main_tbfl_simulation.py # Main execution script (100 rounds)
│   ├── blockchain_manager.py   # Web3 interface
│   ├── data_loader.py          # Chronological split, Dirichlet partition, SMOTETomek
│   └── cliente_fl.py           # FL client (hospital) with FedProx
├── sql/
│   └── extract_cohort.sql      # MIMIC-IV v3.1 cohort/feature extraction query
├── data/                        # Empty — see data/README.md to generate the cohort yourself
│   └── README.md
├── hardhat.config.js            # Hardhat configuration
├── package.json                 # Node.js dependencies
├── requirements.txt             # Python dependencies
└── README.md
```

## Step-by-Step Implementation

### Prerequisites

- **Python 3.8+**
- **Node.js 14+** and npm
- **MIMIC-IV dataset access** — obtain credentialed access via [PhysioNet](https://physionet.org/content/mimiciv/)
- **Ethereum wallet** — not required for the local workflow below, which runs entirely against a local Hardhat node (`npx hardhat node`). Only needed if you adapt `hardhat.config.js` to deploy to a public testnet such as Sepolia (e.g., via [MetaMask](https://metamask.io) or [Alchemy faucet](https://sepoliafaucet.com))

---

### Step 1: Clone and install dependencies

```bash
git clone https://github.com/rodrigoronner/JIS-SSI-FL.git
cd JIS-SSI-FL

# Install blockchain dependencies (Hardhat, Ethers.js)
npm install

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

### Step 2: Launch local blockchain node

Open **Terminal 1** and keep it running throughout the experiment:

```bash
npx hardhat node
```

Expected output:
```
Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/

Accounts
========
Account #0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 (10000 ETH)
Account #1: 0x70997970C51812dc3A010C7d01b50e0d17dc79C8 (10000 ETH)
...
```

The first account (Account #0) acts as the **Trusted Issuer** — the credential authority that signs Verifiable Credentials for legitimate participants.

---

### Step 3: Deploy the FLRegistry smart contract

Open **Terminal 2**:

```bash
npx hardhat run scripts/deploy.js --network localhost
```

Expected output:
```
FLRegistry deployed to: 0x5FbDB2315678afecb367f032d93F642f64180aa3
```

Copy the contract address and update it in `src/main_tbfl_simulation.py`:

```python
# Line ~20 in main_tbfl_simulation.py
CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3'
```

---

### Step 4: Extract the MIMIC-IV cohort

No dataset ships with this repository — MIMIC-IV requires credentialed PhysioNet access and its Data Use Agreement prohibits redistribution. You must extract the cohort yourself:

1. Get credentialed access to MIMIC-IV v3.1 via [PhysioNet](https://physionet.org/content/mimiciv/) and load it into a local PostgreSQL instance following the official [mimic-code](https://github.com/MIT-LCP/mimic-code) build scripts.
2. Build the community "concepts" layer (`mimic-iv/concepts_postgres`), which materializes `mimiciv_derived.charlson` (Charlson Comorbidity Index) used by the query below.
3. Run [`sql/extract_cohort.sql`](sql/extract_cohort.sql) against your database and export the result to `data/mortalidade_features.csv` — see [`data/README.md`](data/README.md) for the exact `\copy` command.

This query implements the paper's inclusion criteria (adult patients, first admission per subject, non-null `hospital_expire_flag`) and produces the `admittime` column that `src/data_loader.py` requires for a true chronological 90/10 split, plus the full clinical feature set (demographics, admission context, Charlson index, length of stay, ICD-10 sepsis/heart-failure flags) described in Sec. 4.1 of the paper.

Verify extraction:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/mortalidade_features.csv'); print(f'{len(df)} admissions loaded, columns: {list(df.columns)}')"
# Expected: 546028 admissions loaded (may differ slightly depending on your MIMIC-IV build/version)
```

---

### Step 5: Run the FL simulation with SSI authentication

In **Terminal 2** (with Python virtual environment activated):

```bash
python src/main_tbfl_simulation.py
```

**What happens at each round ($R = 100$ total, $K = 10$ honest hospitals + 5 Sybil nodes from Round 10 onward):**

| Phase | Action | Component |
|-------|--------|-----------|
| **1. Credential issuance** | Issuer authorizes each of the 10 honest hospital DIDs; the 5 Sybil DIDs are never authorized | `main_tbfl_simulation.py` via `blockchain_manager.py` |
| **2. On-chain registration** | `FLRegistry.sol` deployed once, holding the authorization registry and current-round state | `scripts/deploy.js` (one-time) |
| **3. Authenticated participation** | Every node (honest and Sybil) calls `submitUpdate`; the contract accepts only calls from authorized addresses — Sybil calls revert | `FLRegistry.sol`, `blockchain_manager.py` |
| **4. Local training** | Each honest hospital trains the MLP on its Dirichlet-partitioned, SMOTETomek-balanced local fold using FedProx ($\mu = 0.01$, SGD lr=0.01/momentum=0.9); Sybil nodes submit i.i.d. $\mathcal{N}(0,1)$ noise instead of training | `cliente_fl.py`, `main_tbfl_simulation.py` |
| **5. Four-track aggregation** | The script runs `proposed`, `pki`, `reputation`, and `baseline` from the same data/model init (see below), so the Sybil-resistance gap — and what specifically blockchain adds versus a simpler identity check — is measured directly rather than assumed | `main_tbfl_simulation.py::run_simulation` |

**The four tracks**, all evaluated under the identical attack (5 Sybil nodes join at Round 10):

| Track | Identity check | Enforcement | Expected behavior |
|-------|-----------------|-------------|--------------------|
| `proposed` | Yes (VC-gated) | On-chain (`FLRegistry.sol`) | Sybils rejected before any computation reaches aggregation — zero degradation |
| `pki` | Yes (VC-gated) | Off-chain, in-memory allowlist | **Identical** trajectory to `proposed` — isolates that Sybil-resistance comes from checking identity first, not from the ledger itself; blockchain's contribution is auditability/decentralization (Sec. 6.1), not raw Sybil defense |
| `reputation` | No — any node may submit | Reactive trust score (cosine similarity to the round's median update; cold-start neutral, asymmetric reward/penalty) | Sybils are included for the first 1-2 rounds after joining (a real degradation "window of vulnerability") before trust drops below the participation threshold and they are excluded |
| `baseline` | No | None | Sybils are aggregated every round indefinitely; the reference worst case |

The `reputation` track is a reference implementation of the class of defenses critiqued in Sec. 2.1 (trust accumulated from historical contribution quality) — it is not a reproduction of any specific named system (e.g. FoolsGold, Krum), and its thresholds (`reputation_*` in `ARGS`) are simple, documented choices rather than a tuned/optimal design.

---

### Step 6: Run the Sybil attack demonstration

To test the SSI protection, the simulation includes a **Sybil adversary** with 5 fake identities:

```bash
python src/demo_security_mechanism.py
```

The adversary:
- Creates 5 DID identities but **lacks valid VCs** from the Trusted Issuer
- Attempts to submit updates to `FLRegistry.sol` — **100% rejected** (transaction reverts)

This script only demonstrates the registration/rejection mechanism in isolation. The actual accuracy-degradation comparison (baseline vs. proposed under a live Sybil noise-injection attack) is produced by running the full dual-track simulation in `main_tbfl_simulation.py`, which writes both tracks to CSV for direct comparison.

**Expected result:**
```
Sybil attack simulation complete:
  Attempted registrations: 5
  Blocked (no VC):         5  (100%)
```

---

### Step 7: Verify results

After 100 rounds, `main_tbfl_simulation.py` writes one CSV per track (`tbfl_results_proposed.csv`, `tbfl_results_pki.csv`, `tbfl_results_reputation.csv`, `tbfl_results_baseline.csv`) and prints:
- pairwise statistical comparisons of accuracy over the final 40 rounds (Welch's t-test and Cohen's d) between `proposed` and each of the other three tracks;
- the **window of vulnerability** for each track — how many rounds after Round 10 the AUC stays degraded before durably recovering (0 for `proposed`/`pki`, a small positive number for `reputation`, and typically "never recovers" for `baseline`);
- the cumulative on-chain gas cost and its USD conversion at 20 Gwei / \$3,000-ETH, computed from the receipts actually returned by the local Hardhat node (only the `proposed` track touches the blockchain; `pki` and `reputation` are deliberately off-chain).

The exact AUC/Recall/F1 values reported in the paper (AUC=0.954, Recall=0.890, F1=0.883) were obtained on the full 25-feature, chronologically-ordered cohort described in Sec. 4.1 — reproducing them requires extracting your own cohort via `sql/extract_cohort.sql` (Step 4) rather than a reduced ad-hoc feature set. What this repository's code faithfully reproduces regardless of the exact feature set is the *qualitative* security result: `proposed` and `pki` keep a stable, monotonically-improving accuracy/AUC trajectory after Round 10 with zero window of vulnerability; `reputation` dips briefly then recovers once trust drops below threshold; `baseline` degrades and never recovers.

## Acknowledgments

- MIT Laboratory for Computational Physiology (MIMIC-IV)
- PhysioNet for credentialed clinical data access
- Federal Institute of Rio Grande do Norte (IFRN) for computational resources

## License

MIT License. See [LICENSE](LICENSE) for details.
