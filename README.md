# JISA-SSI-FL: Self-Sovereign Identity for Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hardhat](https://img.shields.io/badge/built%20with-Hardhat-FFDB1C.svg)](https://hardhat.org/)
[![JISA](https://img.shields.io/badge/submitted-JISA-orange)](https://journals-sol.sbc.org.br/index.php/jisa)

> Official implementation accompanying: **"A Decentralized Identity Protocol for Trustworthy Federated Learning: Comparative Analysis of Trust Models for Internet-Based Health Services"** — submitted to the *Journal of Internet Services and Applications (JISA)*.

## Overview

This repository implements a decentralized identity verification protocol for Federated Learning (FL) in Internet-distributed environments. By integrating W3C Self-Sovereign Identity (SSI) standards (DIDs, VCs) with blockchain smart contracts, the protocol provides deterministic Sybil-resistance through proactive institutional identity verification rather than reactive reputation scoring.

### Key Features

- **100% external Sybil attack neutralization**: Only DIDs with valid Verifiable Credentials can participate
- **Domain-agnostic protocol**: Transferable beyond healthcare to any Internet service requiring authenticated collaborative ML
- **Clinical-grade performance**: AUC-ROC = 0.954, Recall = 0.890 on MIMIC-IV mortality prediction
- **< 1% protocol overhead**: Blockchain verification + IPFS storage incurs negligible latency relative to local training
- **Analytical cost model**: Enables pre-deployment economic planning (~$15.67 for N=10 over 100 rounds)
- **LGPD compliance mapping**: First structured analysis of Brazilian data protection law across FL trust models

## Repository Structure

```
JISA-SSI-FL/
├── contracts/                   # Ethereum Smart Contracts
│   └── FLRegistry.sol          # DID/VC verification + access control
├── scripts/                     # Deployment and utilities
│   └── deploy.js               # Contract deployment script
├── src/                         # Federated Learning Core
│   ├── main_tbfl_simulation.py # Main execution script (100 rounds)
│   ├── blockchain_manager.py   # Web3 interface
│   └── cliente_fl.py           # FL client (hospital) with FedProx
├── data/                        # Preprocessed MIMIC-IV cohort
│   └── mortalidade_features.csv.zip
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
- **Ethereum wallet** — any funded account with Sepolia testnet ETH (e.g., via [MetaMask](https://metamask.io) or [Alchemy faucet](https://sepoliafaucet.com))

---

### Step 1: Clone and install dependencies

```bash
git clone https://github.com/rodrigoronner/JISA-SSI-FL.git
cd JISA-SSI-FL

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

### Step 4: Prepare the MIMIC-IV dataset

The repository includes a preprocessed, de-identified cohort derived from MIMIC-IV (546,028 ICU admissions):

```bash
cd data/
unzip mortalidade_features.csv.zip     # creates mortalidade_features.csv
cd ..
```

Verify extraction:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/mortalidade_features.csv'); print(f'{len(df)} admissions loaded')"
# Expected: 546028 admissions loaded
```

**Option B — generate from raw MIMIC-IV:** If you have credentialed access to MIMIC-IV v2.2+ via a PostgreSQL instance, the cohort selection logic is embedded in `src/data_loader.py`. Run the SQL query against your MIMIC-IV database and save the result as `data/mortalidade_features.csv`.

---

### Step 5: Run the FL simulation with SSI authentication

In **Terminal 2** (with Python virtual environment activated):

```bash
python src/main_tbfl_simulation.py
```

**What happens at each round ($R = 100$ total):**

| Phase | Action | Component |
|-------|--------|-----------|
| **1. Credential issuance** | Issuer signs a Verifiable Credential (VC) for each legitimate hospital DID | Off-chain (simulated in `main_tbfl_simulation.py`) |
| **2. On-chain registration** | Each hospital registers its DID + public key in `FLRegistry.sol` | `scripts/deploy.js` (one-time) |
| **3. Authenticated participation** | Hospital submits VC as Verifiable Presentation → contract verifies signature → adds to authorized set $\mathcal{A}_r$ | `FLRegistry.sol`, `blockchain_manager.py` |
| **4. Local training** | Hospital trains MLP classifier on its non-IID partition using FedProx ($\mu = 0.01$) + SMOTETomek | `cliente_fl.py` |
| **5. Verified aggregation** | Authorized hospitals upload model hash to IPFS → submit CID signed with DID private key → aggregator retrieves only verified updates | `main_tbfl_simulation.py` |

---

### Step 6: Run the Sybil attack demonstration

To test the SSI protection, the simulation includes a **Sybil adversary** with 5 fake identities:

```bash
python src/demo_security_mechanism.py
```

The adversary:
- Creates 5 DID identities but **lacks valid VCs** from the Trusted Issuer
- Attempts to register in `FLRegistry.sol` — **100% rejected**
- Simulates a label-flipping attack (minority class → majority class) — **gradients never reach aggregation**

**Expected result:**
```
Sybil attack simulation complete:
  - Attempted registrations: 5
  - Blocked (no VC):         5  (100%)
  - Model degradation:       0  (T0 vs T3: AUC-ROC gap = 14 points)
```

---

### Step 7: Verify results

After 100 rounds, the simulation outputs final metrics to console. Representative results:

| Metric | T3 (SSI-FL, under attack) | T0 (no auth, under attack) |
|--------|--------------------------|---------------------------|
| AUC-ROC | **0.954** | 0.813 |
| Accuracy | **88.5%** | 75.8% |
| Recall | **0.890** | 0.700 |
| FNR | **11.0%** | 30.0% |
| Blockchain overhead | **< 0.12%** | — |
| Protocol overhead (total) | **< 1%** | — |

The cost model $C(N,R)$ can be projected from console gas logs or the analytical formula $C(N,R) = 0.067N + 0.009NR + 0.06R$.


## Acknowledgments

- MIT Laboratory for Computational Physiology (MIMIC-IV)
- PhysioNet for credentialed clinical data access
- Federal Institute of Rio Grande do Norte (IFRN) for computational resources

## License

MIT License. See [LICENSE](LICENSE) for details.
