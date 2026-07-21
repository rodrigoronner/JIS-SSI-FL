import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader

from blockchain_manager import BlockchainManager
from cliente_fl import MLP, MimicDataset, train_client_fedprox
from data_loader import load_and_process_mimic

# ================= CONFIGURATIONS =================
CONTRACT_ADDRESS = '0x5FbDB2315678afecb367f032d93F642f64180aa3'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
CSV_PATH = os.path.join(_PROJECT_ROOT, 'data', 'mortalidade_features.csv')

# Gas -> USD conversion assumptions used throughout the paper's cost analysis
# (Sec. 4.4): 20 Gwei gas price, ETH at $3,000.
GWEI_PRICE = 20
ETH_USD = 3000
USD_PER_GAS_UNIT = GWEI_PRICE * 1e-9 * ETH_USD

ARGS = {
    'rounds': 100,          # R = 100 global training rounds
    'num_honest': 10,       # K = 10 legitimate federated clients (hospitals)
    'num_sybils': 5,        # 5 Sybil nodes -> 5/(10+5) = 33% of the network after injection
    'attack_start_round': 10,  # Sybils join at Round 10 (1-indexed)
    'local_ep': 3,          # E = 3 local epochs per round
    'bs': 32,               # Batch size
    'lr': 0.01,             # SGD learning rate
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'mu': 0.01,             # FedProx proximal term coefficient
    'dirichlet_alpha': 0.5,  # Non-IID partitioning across honest clients
    'seed': 42,
}
# =================================================


def average_weights(state_dicts, sample_counts):
    """
    Performs FedAvg weighted by each contributor's local sample count
    (Sec. 3.5: "computes a weighted average of validated updates using
    sample counts as weights"), i.e. w_avg = sum_k (n_k / sum(n)) * w_k.
    """
    total = float(sum(sample_counts))
    weights = [n / total for n in sample_counts]

    w_avg = copy.deepcopy(state_dicts[0])
    for key in w_avg.keys():
        w_avg[key] = state_dicts[0][key] * weights[0]
        for i in range(1, len(state_dicts)):
            w_avg[key] += state_dicts[i][key] * weights[i]
    return w_avg


def sybil_noise_update(reference_state_dict):
    """
    Produces a fake "model update" for a Sybil node: every tensor is sampled
    i.i.d. from N(0, 1), matching the shapes of a legitimate update (Sec. 4.3).
    """
    return {k: torch.randn_like(v) for k, v in reference_state_dict.items()}


def evaluate_model(model, X_test, y_test):
    """Evaluates the global model on the held-out test set."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    model.eval()
    criterion = nn.BCELoss()
    loader = DataLoader(MimicDataset(X_test, y_test), batch_size=64, shuffle=False)

    y_true, y_pred_probs, total_loss = [], [], 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            y_pred_probs.extend(outputs.numpy())
            y_true.extend(labels.numpy())

    avg_loss = total_loss / len(loader)
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_cls = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_cls)
    prec = precision_score(y_true, y_pred_cls, zero_division=0)
    rec = recall_score(y_true, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true, y_pred_cls, zero_division=0)
    try:
        auc_val = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc_val = 0.5

    return avg_loss, acc, prec, rec, f1, auc_val


def run_simulation(track_name, enforce_identity, bc, honest_workers, sybil_workers,
                    client_data, X_test, y_test, input_dim, args):
    """
    Runs one full R-round federated simulation.

    enforce_identity=True  -> "proposed" system: every submission goes through
        FLRegistry.submitUpdate(); only credentialed honest workers succeed,
        Sybil submissions revert and are excluded from aggregation.
    enforce_identity=False -> "baseline" system: standard FedAvg with no
        on-chain gate at all; Sybil noise updates (from attack_start_round
        onward) are aggregated alongside honest updates, exactly like a
        deployment that never added identity verification.
    """
    torch.manual_seed(args['seed'])
    global_model = MLP(input_dim)
    global_model.train()

    # Sample counts used as FedAvg weights (Sec. 3.5). Honest clients weigh
    # by the size of their local (post-SMOTETomek) training fold. A Sybil
    # node has no genuine dataset; we assume it claims a size equal to the
    # mean honest client, i.e. a plausible-looking institution rather than
    # a trivially detectable outlier — this is a modeling assumption, not
    # something the paper specifies explicitly.
    honest_sample_counts = {cid: len(client_data[cid][0]) for cid in range(len(honest_workers))}
    sybil_claimed_count = int(np.mean(list(honest_sample_counts.values())))

    history = []
    cumulative_gas = 0

    for round_idx in range(1, args['rounds'] + 1):
        accepted_weights, accepted_counts = [], []
        training_times, blockchain_times = [], []
        attack_active = round_idx >= args['attack_start_round']

        # --- Honest clients: real local training on their Dirichlet/SMOTETomek fold ---
        for cid, worker_addr in enumerate(honest_workers):
            X_c, y_c = client_data[cid]
            t0 = time.time()
            w, _ = train_client_fedprox(
                copy.deepcopy(global_model), X_c, y_c, args, global_model
            )
            training_times.append(time.time() - t0)

            if enforce_identity:
                t0_bc = time.time()
                fake_cid = f"Qm{round_idx}_{worker_addr[:8]}"
                ok, gas_used = bc.submit_hash(worker_addr, fake_cid)
                blockchain_times.append(time.time() - t0_bc)
                cumulative_gas += gas_used
                if ok:
                    accepted_weights.append(w)
                    accepted_counts.append(honest_sample_counts[cid])
            else:
                accepted_weights.append(w)
                accepted_counts.append(honest_sample_counts[cid])

        # --- Sybil nodes: random-noise updates injected from attack_start_round ---
        if attack_active:
            for worker_addr in sybil_workers:
                noisy_update = sybil_noise_update(global_model.state_dict())
                if enforce_identity:
                    fake_cid = f"Qm{round_idx}_sybil_{worker_addr[:8]}"
                    ok, gas_used = bc.submit_hash(worker_addr, fake_cid)
                    cumulative_gas += gas_used  # 0 for reverted calls in blockchain_manager
                    if ok:
                        accepted_weights.append(noisy_update)  # never happens: no VC
                        accepted_counts.append(sybil_claimed_count)
                else:
                    accepted_weights.append(noisy_update)
                    accepted_counts.append(sybil_claimed_count)

        if not accepted_weights:
            continue

        global_weights = average_weights(accepted_weights, accepted_counts)
        global_model.load_state_dict(global_weights)

        loss, acc, prec, rec, f1, auc_val = evaluate_model(global_model, X_test, y_test)

        if round_idx % 10 == 0:
            print(f"  [{track_name}] R{round_idx}: Loss={loss:.4f} Acc={acc:.4f} "
                  f"AUC={auc_val:.4f} Recall={rec:.4f}")

        history.append({
            'round': round_idx,
            'loss': loss,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc_val,
            'avg_train_time': np.mean(training_times) if training_times else np.nan,
            'avg_blockchain_time': np.mean(blockchain_times) if blockchain_times else np.nan,
            'cumulative_gas': cumulative_gas,
            'cumulative_cost_usd': cumulative_gas * USD_PER_GAS_UNIT,
        })

    return pd.DataFrame(history)


def compare_tracks(df_proposed, df_baseline, last_n_rounds=40):
    """
    Statistical comparison of the two tracks' accuracy over the final
    `last_n_rounds` rounds (Sec. 5.2 reports this over the final 40 rounds).
    Both series come from real simulation runs — no synthetic baseline.
    """
    print("\n--- STATISTICAL ANALYSIS (real baseline vs. real proposed run) ---")

    acc_proposed = df_proposed['accuracy'].values[-last_n_rounds:]
    acc_baseline = df_baseline['accuracy'].values[-last_n_rounds:]

    t_stat, p_val = stats.ttest_ind(acc_proposed, acc_baseline, equal_var=False)

    n1, n2 = len(acc_proposed), len(acc_baseline)
    pooled_std = np.sqrt(
        ((n1 - 1) * acc_proposed.std(ddof=1) ** 2 + (n2 - 1) * acc_baseline.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    cohens_d = (acc_proposed.mean() - acc_baseline.mean()) / pooled_std if pooled_std > 0 else np.nan

    print(f"  Proposed mean accuracy (last {last_n_rounds} rounds): {acc_proposed.mean():.4f}")
    print(f"  Baseline mean accuracy (last {last_n_rounds} rounds): {acc_baseline.mean():.4f}")
    print(f"  t = {t_stat:.4f}, p = {p_val:.3e}, Cohen's d = {cohens_d:.4f}")

    return t_stat, p_val, cohens_d


def main():
    print(f"Starting TBFL simulation ({ARGS['rounds']} rounds, "
          f"{ARGS['num_honest']} honest + {ARGS['num_sybils']} Sybil nodes)...")

    bc = BlockchainManager(CONTRACT_ADDRESS)

    total_workers_needed = 1 + ARGS['num_honest'] + ARGS['num_sybils']
    if len(bc.accounts) < total_workers_needed:
        raise RuntimeError(
            f"Need {total_workers_needed} funded accounts (1 issuer + "
            f"{ARGS['num_honest']} honest + {ARGS['num_sybils']} Sybil), "
            f"but the connected node only exposes {len(bc.accounts)}."
        )

    honest_workers = [bc.get_account(i) for i in range(1, ARGS['num_honest'] + 1)]
    sybil_workers = [
        bc.get_account(i)
        for i in range(ARGS['num_honest'] + 1, ARGS['num_honest'] + ARGS['num_sybils'] + 1)
    ]

    print("\nOnboarding: issuing credentials to honest hospitals only.")
    for worker in honest_workers:
        bc.issue_credential(worker)
    print(f"{len(sybil_workers)} Sybil nodes were NOT issued credentials.")

    X_train, y_train, X_test, y_test, client_data = load_and_process_mimic(
        CSV_PATH, ARGS['num_honest'], dirichlet_alpha=ARGS['dirichlet_alpha'], seed=ARGS['seed']
    )
    input_dim = X_train.shape[1]

    print("\n=== Track 1/2: PROPOSED (identity-verified) ===")
    df_proposed = run_simulation(
        'proposed', True, bc, honest_workers, sybil_workers,
        client_data, X_test, y_test, input_dim, ARGS
    )

    print("\n=== Track 2/2: BASELINE (standard FedAvg, no identity verification) ===")
    df_baseline = run_simulation(
        'baseline', False, bc, honest_workers, sybil_workers,
        client_data, X_test, y_test, input_dim, ARGS
    )

    df_proposed.to_csv('tbfl_results_proposed.csv', index=False)
    df_baseline.to_csv('tbfl_results_baseline.csv', index=False)
    print("\nResults saved to tbfl_results_proposed.csv / tbfl_results_baseline.csv")

    compare_tracks(df_proposed, df_baseline, last_n_rounds=40)

    total_cost = df_proposed['cumulative_cost_usd'].iloc[-1]
    print(f"\nTotal on-chain cost over {ARGS['rounds']} rounds (proposed system): "
          f"${total_cost:.2f} (20 Gwei, ETH=${ETH_USD})")


if __name__ == '__main__':
    main()
