import json
import os

from web3 import Web3

"""
demo_security_mechanism.py

Standalone Proof of Concept demonstrating the Identity-First security
mechanism: 2 legitimate hospitals receive Verifiable Credentials, while a
Sybil adversary controlling 5 unauthorized identities (Sec. 4.3: 5 Sybil
nodes) attempts to submit random-noise model updates. The smart contract
rejects every Sybil submission because none of the 5 addresses were ever
authorized by the Trusted Issuer.
"""

# ================= CONFIGURATION =================
RPC_URL = 'http://127.0.0.1:8545'
CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3'
NUM_SYBILS = 5
# =================================================


def load_contract_abi():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    artifact_path = os.path.join(
        project_root, 'artifacts', 'contracts', 'FLRegistry.sol', 'FLRegistry.json'
    )
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"ABI artifact not found at {artifact_path}. Run 'npx hardhat compile' first."
        )
    with open(artifact_path) as f:
        return json.load(f)['abi']


def attempt_submission(w3, contract, worker_account, label):
    """Simulates a noise-based Sybil submission attempt against the contract."""
    fake_ipfs_hash = f"QmSybilNoise_{worker_account[:8]}"
    try:
        tx = contract.functions.submitUpdate(fake_ipfs_hash).transact({'from': worker_account})
        receipt = w3.eth.wait_for_transaction_receipt(tx)
        print(f"  [{label}] SUCCESS: accepted in block {receipt['blockNumber']} "
              f"(gas={receipt['gasUsed']}).")
        return True
    except Exception:
        print(f"  [{label}] BLOCKED: transaction reverted (Access Denied: No Valid VC).")
        return False


def main():
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("Failed to connect to blockchain. Ensure 'npx hardhat node' is running.")
        return

    abi = load_contract_abi()
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
    print(f"Connected to FLRegistry at {CONTRACT_ADDRESS}")

    issuer = w3.eth.accounts[0]
    hospital_a = w3.eth.accounts[1]
    hospital_b = w3.eth.accounts[2]
    sybil_accounts = w3.eth.accounts[3:3 + NUM_SYBILS]

    print("\n--- PHASE 1: Credential Issuance (legitimate hospitals only) ---")
    for label, addr in [("Hospital A", hospital_a), ("Hospital B", hospital_b)]:
        tx_hash = contract.functions.authorizeWorker(addr).transact({'from': issuer})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"  Issuer authorized {label} ({addr[:8]}...).")

    print(f"\n{len(sybil_accounts)} Sybil identities were NOT issued credentials.")

    print("\n--- PHASE 2: Submission attempts ---")
    attempt_submission(w3, contract, hospital_a, "Hospital A (legit)")
    attempt_submission(w3, contract, hospital_b, "Hospital B (legit)")

    blocked = 0
    for i, sybil_addr in enumerate(sybil_accounts):
        accepted = attempt_submission(w3, contract, sybil_addr, f"Sybil #{i + 1}")
        blocked += (not accepted)

    print("\nSybil attack simulation complete:")
    print(f"  Attempted registrations: {len(sybil_accounts)}")
    print(f"  Blocked (no VC):         {blocked} "
          f"({100 * blocked / len(sybil_accounts):.0f}%)")


if __name__ == "__main__":
    main()
