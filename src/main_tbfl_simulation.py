import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # Reputation-based baseline (Sec. 2.1's critiqued class of defenses):
    # a reference implementation using cosine-similarity-to-median as the
    # trust signal, NOT a reproduction of any specific named system
    # (e.g. FoolsGold, Krum) — it exists to demonstrate the *reactive,
    # detection-lag* property common to this class, not to claim it is the
    # strongest possible reputation defense.
    'reputation_participation_threshold': 0.5,   # trust >= this to be aggregated
    'reputation_similarity_threshold': 0.5,      # cosine sim >= this counts as "good"
    'reputation_reward': 0.1,                    # trust gained per good round
    'reputation_penalty': 0.3,                   # trust lost per bad round (asymmetric: slow to build, fast to lose)
}
# =================================================


def flatten_state_dict(state_dict):
    """Flattens a model state_dict into a single 1-D tensor for similarity comparisons."""
    return torch.cat([v.flatten() for v in state_dict.values()])


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


def run_simulation(track_name, mode, bc, honest_workers, sybil_workers,
                    client_data, X_test, y_test, input_dim, args):
    """
    Runs one full R-round federated simulation under one of four gating modes:

    mode='proposed'   -> identity-first, on-chain: every submission goes
        through FLRegistry.submitUpdate(); only credentialed honest workers
        succeed, Sybil submissions revert and are excluded from aggregation.
        Real gas cost is measured.
    mode='pki'        -> identity-first, off-chain: the SAME deterministic
        allowlist semantics as 'proposed', but enforced with a plain
        in-memory set instead of a smart contract. Exists to isolate what
        blockchain specifically adds on top of "identity-first" as a
        paradigm (Sec. 3.1/6.1): if 'pki' and 'proposed' produce identical
        security trajectories, the Sybil-resistance comes from checking
        identity before aggregation, not from the ledger itself.
    mode='reputation' -> the class of defense critiqued in Sec. 2.1: no
        identity check at all; every node may submit, and a trust score
        accumulated from historical update similarity gates future
        participation. Cold-start neutral (all nodes start fully trusted),
        reactive (a node is only excluded after enough bad rounds are
        observed), unlike identity-first's preemptive exclusion.
    mode='baseline'   -> standard FedAvg, no defense of any kind.
    """
    torch.manual_seed(args['seed'])
    global_model = MLP(input_dim)
    global_model.train()

    honest_sample_counts = {cid: len(client_data[cid][0]) for cid in range(len(honest_workers))}
    sybil_claimed_count = int(np.mean(list(honest_sample_counts.values())))

    # Static allowlist for the 'pki' mode (no blockchain involved at all).
    pki_authorized = set(honest_workers)

    # Trust scores for the 'reputation' mode: every node starts fully
    # trusted (cold-start neutrality — matches the literature's description
    # of open FL deployments where "any node presenting a valid network
    # address and key pair may request participation", Sec. 2.1).
    trust_scores = {addr: 1.0 for addr in list(honest_workers) + list(sybil_workers)}

    history = []
    cumulative_gas = 0

    for round_idx in range(1, args['rounds'] + 1):
        training_times, blockchain_times = [], []
        attack_active = round_idx >= args['attack_start_round']

        # --- Honest clients always train locally, regardless of mode ---
        honest_updates = {}
        for cid, worker_addr in enumerate(honest_workers):
            X_c, y_c = client_data[cid]
            t0 = time.time()
            w, _ = train_client_fedprox(
                copy.deepcopy(global_model), X_c, y_c, args, global_model
            )
            training_times.append(time.time() - t0)
            honest_updates[worker_addr] = w

        # --- Sybil nodes submit N(0,1) noise once the attack is active ---
        sybil_updates = {}
        if attack_active:
            for worker_addr in sybil_workers:
                sybil_updates[worker_addr] = sybil_noise_update(global_model.state_dict())

        accepted_weights, accepted_counts = [], []

        if mode == 'proposed':
            for cid, worker_addr in enumerate(honest_workers):
                t0_bc = time.time()
                fake_cid = f"Qm{round_idx}_{worker_addr[:8]}"
                ok, gas_used = bc.submit_hash(worker_addr, fake_cid)
                blockchain_times.append(time.time() - t0_bc)
                cumulative_gas += gas_used
                if ok:
                    accepted_weights.append(honest_updates[worker_addr])
                    accepted_counts.append(honest_sample_counts[cid])
            for worker_addr, update in sybil_updates.items():
                fake_cid = f"Qm{round_idx}_sybil_{worker_addr[:8]}"
                ok, gas_used = bc.submit_hash(worker_addr, fake_cid)
                cumulative_gas += gas_used  # 0 for reverted calls in blockchain_manager
                if ok:
                    accepted_weights.append(update)  # never happens: no VC
                    accepted_counts.append(sybil_claimed_count)

        elif mode == 'pki':
            for cid, worker_addr in enumerate(honest_workers):
                if worker_addr in pki_authorized:
                    accepted_weights.append(honest_updates[worker_addr])
                    accepted_counts.append(honest_sample_counts[cid])
            for worker_addr, update in sybil_updates.items():
                if worker_addr in pki_authorized:
                    accepted_weights.append(update)  # never happens: not in allowlist
                    accepted_counts.append(sybil_claimed_count)

        elif mode == 'reputation':
            candidates = [
                (worker_addr, honest_updates[worker_addr], honest_sample_counts[cid])
                for cid, worker_addr in enumerate(honest_workers)
            ] + [
                (worker_addr, update, sybil_claimed_count)
                for worker_addr, update in sybil_updates.items()
            ]
            flats = {addr: flatten_state_dict(upd) for addr, upd, _ in candidates}
            reference = torch.median(torch.stack(list(flats.values())), dim=0).values

            for addr, upd, count in candidates:
                if trust_scores[addr] >= args['reputation_participation_threshold']:
                    accepted_weights.append(upd)
                    accepted_counts.append(count)
                # Trust is updated AFTER this round's gating decision, so a
                # newly-attacking node is still included in the round it
                # first misbehaves — the detection lag being measured.
                sim = F.cosine_similarity(
                    flats[addr].unsqueeze(0), reference.unsqueeze(0)
                ).item()
                if sim >= args['reputation_similarity_threshold']:
                    trust_scores[addr] = min(1.0, trust_scores[addr] + args['reputation_reward'])
                else:
                    trust_scores[addr] = max(0.0, trust_scores[addr] - args['reputation_penalty'])

        elif mode == 'baseline':
            for cid, worker_addr in enumerate(honest_workers):
                accepted_weights.append(honest_updates[worker_addr])
                accepted_counts.append(honest_sample_counts[cid])
            for update in sybil_updates.values():
                accepted_weights.append(update)
                accepted_counts.append(sybil_claimed_count)

        else:
            raise ValueError(f"Unknown mode: {mode}")

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
            'n_accepted': len(accepted_weights),
            'avg_train_time': np.mean(training_times) if training_times else np.nan,
            'avg_blockchain_time': np.mean(blockchain_times) if blockchain_times else np.nan,
            'cumulative_gas': cumulative_gas,
            'cumulative_cost_usd': cumulative_gas * USD_PER_GAS_UNIT,
        })

    return pd.DataFrame(history)


def compare_tracks(df_a, df_b, name_a, name_b, last_n_rounds=40):
    """
    Welch's t-test + Cohen's d comparing two tracks' accuracy over the final
    `last_n_rounds` rounds. Both series come from real simulation runs.
    """
    acc_a = df_a['accuracy'].values[-last_n_rounds:]
    acc_b = df_b['accuracy'].values[-last_n_rounds:]

    t_stat, p_val = stats.ttest_ind(acc_a, acc_b, equal_var=False)

    n1, n2 = len(acc_a), len(acc_b)
    pooled_std = np.sqrt(
        ((n1 - 1) * acc_a.std(ddof=1) ** 2 + (n2 - 1) * acc_b.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    cohens_d = (acc_a.mean() - acc_b.mean()) / pooled_std if pooled_std > 0 else np.nan

    print(f"  {name_a} vs {name_b} (last {last_n_rounds} rounds): "
          f"mean_a={acc_a.mean():.4f}, mean_b={acc_b.mean():.4f}, "
          f"t={t_stat:.4f}, p={p_val:.3e}, d={cohens_d:.4f}")

    return t_stat, p_val, cohens_d


def detection_window(df, attack_start_round, pre_attack_value, metric='auc', tolerance=0.02):
    """
    Counts how many rounds after `attack_start_round` a track's `metric`
    stays below (pre_attack_value - tolerance) before recovering and
    remaining recovered through the end of the run. Used to quantify the
    "window of vulnerability" that separates reactive defenses
    (reputation, baseline) from preemptive ones (proposed, pki).

    Returns None if the metric never recovers within the observed rounds.
    """
    post = df[df['round'] >= attack_start_round].reset_index(drop=True)
    threshold = pre_attack_value - tolerance

    below = post[metric] < threshold
    if not below.any():
        return 0  # never dipped below threshold at all

    last_bad_idx = below[below].index.max()
    # Confirm it stays recovered afterward (no relapse to the end of the run).
    if (post[metric].iloc[last_bad_idx + 1:] < threshold).any():
        return None  # never durably recovers
    return int(post['round'].iloc[last_bad_idx] - attack_start_round + 1)


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

    print("\nOnboarding: issuing on-chain credentials to honest hospitals only "
          "(used by the 'proposed' track; 'pki' uses an equivalent off-chain "
          "allowlist; 'reputation' and 'baseline' perform no identity check).")
    for worker in honest_workers:
        bc.issue_credential(worker)
    print(f"{len(sybil_workers)} Sybil nodes were NOT issued credentials.")

    X_train, y_train, X_test, y_test, client_data = load_and_process_mimic(
        CSV_PATH, ARGS['num_honest'], dirichlet_alpha=ARGS['dirichlet_alpha'], seed=ARGS['seed']
    )
    input_dim = X_train.shape[1]

    tracks = [
        ('proposed', 'proposed'),
        ('pki', 'pki'),
        ('reputation', 'reputation'),
        ('baseline', 'baseline'),
    ]
    results = {}
    for i, (label, mode) in enumerate(tracks, 1):
        print(f"\n=== Track {i}/{len(tracks)}: {label.upper()} ===")
        df = run_simulation(label, mode, bc, honest_workers, sybil_workers,
                             client_data, X_test, y_test, input_dim, ARGS)
        df.to_csv(f'tbfl_results_{label}.csv', index=False)
        results[label] = df
    print("\nResults saved to tbfl_results_{proposed,pki,reputation,baseline}.csv")

    print("\n--- STATISTICAL ANALYSIS (final 40 rounds, real simulation runs) ---")
    compare_tracks(results['proposed'], results['baseline'], 'proposed', 'baseline', last_n_rounds=40)
    compare_tracks(results['proposed'], results['pki'], 'proposed', 'pki', last_n_rounds=40)
    compare_tracks(results['proposed'], results['reputation'], 'proposed', 'reputation', last_n_rounds=40)

    print("\n--- WINDOW OF VULNERABILITY (rounds of degraded AUC after attack starts) ---")
    pre_attack_auc = results['proposed'][
        results['proposed']['round'] < ARGS['attack_start_round']
    ]['auc'].iloc[-1]
    for label in ['proposed', 'pki', 'reputation', 'baseline']:
        window = detection_window(results[label], ARGS['attack_start_round'], pre_attack_auc)
        window_str = 'never recovers' if window is None else f'{window} round(s)'
        print(f"  {label}: {window_str}")

    total_cost = results['proposed']['cumulative_cost_usd'].iloc[-1]
    print(f"\nTotal on-chain cost over {ARGS['rounds']} rounds (proposed system): "
          f"${total_cost:.2f} (20 Gwei, ETH=${ETH_USD})")


if __name__ == '__main__':
    main()
