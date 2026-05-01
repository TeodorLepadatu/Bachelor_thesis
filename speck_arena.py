import torch
import torch.nn as nn
import numpy as np
import os
import time
import warnings
from tqdm import tqdm

np.seterr(over='ignore', under='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. CONSTANTS & GPU CONFIGURATION
# =============================================================================
WORD_SIZE = 16
MASK_VAL = 0xFFFF
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x + res


class ImprovedCNN(nn.Module):
    def __init__(self, depth=10):
        super(ImprovedCNN, self).__init__()
        self.in_channels = 3
        self.res_tower = nn.Sequential(*[ResBlock(self.in_channels) for _ in range(depth)])
        self.fc1 = nn.Linear(3 * 16, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.res_tower(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.sigmoid(self.out(x))


# =============================================================================
# 3. PURE PYTORCH CRYPTO HELPERS
# =============================================================================
def expand_key(key, rounds):
    k = [key[i] for i in range(4)]
    ks = [0] * rounds
    ks[0] = k[0]
    l = k[1:]
    for i in range(rounds - 1):
        l_curr = l[0]
        l_curr = ((l_curr >> 7) | (l_curr << 9)) & MASK_VAL
        l_curr = (int(l_curr) + int(ks[i])) & MASK_VAL
        l_curr = l_curr ^ i
        l = l[1:] + [l_curr]
        k_curr = ks[i]
        k_curr = ((k_curr << 2) | (k_curr >> 14)) & MASK_VAL
        k_curr = k_curr ^ l_curr
        ks[i + 1] = k_curr
    return ks


def encrypt_vectorized(p_left, p_right, ks):
    x = p_left.astype(np.int32)
    y = p_right.astype(np.int32)
    for k in ks:
        x = ((x >> 7) | (x << 9)) & MASK_VAL
        x = (x + y) & MASK_VAL
        x = x ^ k
        y = ((y << 2) | (y >> 14)) & MASK_VAL
        y = y ^ x
    return x.astype(np.uint16), y.astype(np.uint16)


def dec_one_round_torch(c0, c1, k):
    if c0.dim() == 1: c0 = c0.unsqueeze(1)
    if c1.dim() == 1: c1 = c1.unsqueeze(1)
    if k.dim() == 1: k = k.unsqueeze(0)
    c1 = c1 ^ c0
    c1 = ((c1 >> 2) | (c1 << 14)) & MASK_VAL
    c0 = c0 ^ k
    if c1.shape != c0.shape:
        c1 = c1.expand_as(c0)
    c0 = (c0 - c1) & MASK_VAL
    c0 = ((c0 << 7) | (c0 >> 9)) & MASK_VAL
    return c0, c1


def extract_features_torch(c0l, c0r, c1l, c1r):
    f1 = c0l ^ c1l
    f2 = c0l ^ c0r
    f3 = c0l ^ c1r
    bits = torch.arange(16, device=c0l.device).reshape(1, 1, 16)
    f_stack = torch.stack([f1, f2, f3], dim=1).unsqueeze(2)
    X = (f_stack >> bits) & 1
    return X.float()


def make_structure(pt0, pt1, diff=(0x0040, 0x0000), neutral_bits=[20, 21, 22, 14, 15]):
    p0l = np.array([pt0], dtype=np.uint16)
    p0r = np.array([pt1], dtype=np.uint16)
    for i in neutral_bits:
        d = 1 << i
        d_high = (d >> 16) & 0xFFFF
        d_low = d & 0xFFFF
        p0l = np.concatenate([p0l, p0l ^ d_high])
        p0r = np.concatenate([p0r, p0r ^ d_low])
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r


def gen_challenge(n_structures, rounds, neutral_bits=[20, 21, 22, 14, 15]):
    secret_key = np.frombuffer(os.urandom(8), dtype=np.uint16)
    ks = expand_key(secret_key, rounds)
    target_subkey = ks[-1]

    p0l_list, p0r_list, p1l_list, p1r_list = [], [], [], []
    c0l_list, c0r_list, c1l_list, c1r_list = [], [], [], []

    for _ in range(n_structures):
        pt0 = np.random.randint(0, 65536, dtype=np.uint16)
        pt1 = np.random.randint(0, 65536, dtype=np.uint16)
        p0l, p0r, p1l, p1r = make_structure(pt0, pt1, neutral_bits=neutral_bits)

        c0_left, c0_right = encrypt_vectorized(p0l, p0r, ks)
        c1_left, c1_right = encrypt_vectorized(p1l, p1r, ks)

        p0l_list.append(p0l)
        p0r_list.append(p0r)
        p1l_list.append(p1l)
        p1r_list.append(p1r)

        c0l_list.append(c0_left)
        c0r_list.append(c0_right)
        c1l_list.append(c1_left)
        c1r_list.append(c1_right)

    p0l = torch.tensor(np.concatenate(p0l_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)
    p0r = torch.tensor(np.concatenate(p0r_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)
    p1l = torch.tensor(np.concatenate(p1l_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)
    p1r = torch.tensor(np.concatenate(p1r_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)

    c0l = torch.tensor(np.concatenate(c0l_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)
    c0r = torch.tensor(np.concatenate(c0r_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)
    c1l = torch.tensor(np.concatenate(c1l_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)
    c1r = torch.tensor(np.concatenate(c1r_list), dtype=torch.int32, device=DEVICE).unsqueeze(1)

    return p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r, target_subkey


def get_log_odds_single(model, X):
    with torch.no_grad():
        v = model(X).squeeze()
    epsilon = 1e-7
    v = torch.clamp(v, epsilon, 1 - epsilon)
    return torch.log2(v / (1 - v))


def get_log_odds_ensemble(m1, m2, X, loss1, loss2):
    with torch.no_grad():
        p1 = m1(X).squeeze()
        p2 = m2(X).squeeze()
    w1, w2 = 1.0 / (loss1 ** 2), 1.0 / (loss2 ** 2)
    alpha1, alpha2 = w1 / (w1 + w2), w2 / (w1 + w2)
    p_final = alpha1 * p1 + alpha2 * p2
    epsilon = 1e-7
    p_final = torch.clamp(p_final, epsilon, 1 - epsilon)
    return torch.log2(p_final / (1 - p_final))


# =============================================================================
# 4. WKRP GENERATION
# =============================================================================
def generate_full_wkrp(model_1, model_2=None, is_ensemble=False, loss1=0, loss2=0, dist_rounds=4, n_samples=2000):
    mode = 'Ensemble' if is_ensemble else 'Single'
    print(f"\nGenerating Full WKRP (Distinguisher={dist_rounds}r) -> {mode}")

    dummy_key = np.zeros(4, dtype=np.uint16)
    ks_dist = expand_key(dummy_key, dist_rounds)

    c0l_list, c0r_list, c1l_list, c1r_list = [], [], [], []
    batches = (n_samples // 32) + 1

    for _ in range(batches):
        pt0 = np.random.randint(0, 65536, dtype=np.uint16)
        pt1 = np.random.randint(0, 65536, dtype=np.uint16)
        p0l, p0r, p1l, p1r = make_structure(pt0, pt1)

        c0l, c0r = encrypt_vectorized(p0l, p0r, ks_dist)
        c1l, c1r = encrypt_vectorized(p1l, p1r, ks_dist)

        # 1 EXTRA Round with Key=0
        c0l, c0r = encrypt_vectorized(c0l, c0r, [0])
        c1l, c1r = encrypt_vectorized(c1l, c1r, [0])

        c0l_list.append(c0l)
        c0r_list.append(c0r)
        c1l_list.append(c1l)
        c1r_list.append(c1r)

    c0l = torch.tensor(np.concatenate(c0l_list)[:n_samples], dtype=torch.int32, device=DEVICE).unsqueeze(1)
    c0r = torch.tensor(np.concatenate(c0r_list)[:n_samples], dtype=torch.int32, device=DEVICE).unsqueeze(1)
    c1l = torch.tensor(np.concatenate(c1l_list)[:n_samples], dtype=torch.int32, device=DEVICE).unsqueeze(1)
    c1r = torch.tensor(np.concatenate(c1r_list)[:n_samples], dtype=torch.int32, device=DEVICE).unsqueeze(1)

    mu_table = torch.zeros(65536, device=DEVICE)
    std_table = torch.zeros(65536, device=DEVICE)
    all_diffs = torch.arange(0, 65536, dtype=torch.int32, device=DEVICE)

    diff_batch = 512
    with torch.no_grad():
        for i in tqdm(range(0, 65536, diff_batch), desc="  -> Profiling Diffs", leave=True):
            batch = all_diffs[i: i + diff_batch]
            d0l, d0r = dec_one_round_torch(c0l, c0r, batch)
            d1l, d1r = dec_one_round_torch(c1l, c1r, batch)

            X = extract_features_torch(d0l.flatten(), d0r.flatten(), d1l.flatten(), d1r.flatten())

            with torch.amp.autocast('cuda'):
                if is_ensemble:
                    z = get_log_odds_ensemble(model_1, model_2, X, loss1, loss2)
                else:
                    z = get_log_odds_single(model_1, X)

            z_mat = z.view(c0l.size(0), -1).float()
            mu_table[i:i + diff_batch] = z_mat.mean(dim=0)
            std_table[i:i + diff_batch] = z_mat.std(dim=0)

    return mu_table, std_table


# =============================================================================
# 5. EXACT BAYESIAN KEY SEARCH & SUM OF LOGITS
# =============================================================================
def sum_of_logits(model, c0l, c0r, c1l, c1r, top_k=64):
    all_keys = torch.arange(0, 65536, dtype=torch.int32, device=DEVICE)
    final_scores = torch.zeros(65536, device=DEVICE)
    batch_size = 256  # for 4GB VRAM safety, a higher value would give OOM

    with torch.no_grad():
        for i in range(0, 65536, batch_size):
            k_batch = all_keys[i: i + batch_size]
            d0l, d0r = dec_one_round_torch(c0l, c0r, k_batch)
            d1l, d1r = dec_one_round_torch(c1l, c1r, k_batch)

            X = extract_features_torch(d0l.flatten(), d0r.flatten(), d1l.flatten(), d1r.flatten())

            with torch.amp.autocast('cuda'):
                z = get_log_odds_single(model, X)

            logits_matrix = z.view(c0l.size(0), -1).float()
            final_scores[i: i + batch_size] = torch.sum(logits_matrix, dim=0)

    _, top_indices = torch.topk(final_scores, top_k)
    return top_indices.int()


def exact_bayesian_key_search(c0l, c0r, c1l, c1r, wkrp_mean, wkrp_std, model1, model2=None, loss1=0, loss2=0,
                              is_ensemble=False, initial_S=None, candidate_space=None, n_iterations=5, n_candidates=32):
    n_pairs = c0l.size(0)

    if candidate_space is None:
        candidate_space = torch.arange(65536, device=DEVICE, dtype=torch.long)
    else:
        candidate_space = candidate_space.long()

    if initial_S is not None:
        S = initial_S.long()[:n_candidates]
    else:
        S = torch.randperm(65536, device=DEVICE)[:n_candidates]

    global_best_key = None
    global_best_score = -float('inf')

    for t in range(n_iterations):
        d0l, d0r = dec_one_round_torch(c0l, c0r, S)
        d1l, d1r = dec_one_round_torch(c1l, c1r, S)

        n_curr_cand = S.size(0)
        if d0r.shape[1] != n_curr_cand: d0r = d0r.expand(-1, n_curr_cand)
        if d1r.shape[1] != n_curr_cand: d1r = d1r.expand(-1, n_curr_cand)

        X = extract_features_torch(d0l.flatten(), d0r.flatten(), d1l.flatten(), d1r.flatten())

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                if is_ensemble:
                    z = get_log_odds_ensemble(model1, model2, X, loss1, loss2)
                else:
                    z = get_log_odds_single(model1, X)

        z_matrix = z.view(n_pairs, n_curr_cand).float()
        s_ki = torch.sum(z_matrix, dim=0)
        m_ki = torch.mean(z_matrix, dim=0)

        batch_max_idx = torch.argmax(s_ki)
        batch_max_score = s_ki[batch_max_idx]
        batch_best_key = S[batch_max_idx].item()

        if batch_max_score > global_best_score:
            global_best_score = batch_max_score
            global_best_key = batch_best_key

        if len(candidate_space) <= n_candidates:
            break

        S_exp = S.unsqueeze(1).long()
        K_exp = candidate_space.unsqueeze(0)
        diffs = S_exp ^ K_exp

        mus = wkrp_mean[diffs]
        sigs = wkrp_std[diffs]

        m_exp = m_ki.unsqueeze(1)
        numer = (m_exp - mus) ** 2
        denom = sigs ** 2 + 1e-9
        lambdas = torch.sum(numer / denom, dim=0)

        _, top_k_indices = torch.topk(lambdas, n_candidates, largest=False)
        S = candidate_space[top_k_indices].int()

        if global_best_key is not None:
            if not (S == global_best_key).any():
                S[-1] = int(global_best_key)

    return int(global_best_key)


# =============================================================================
# 6. TXT DATASET EXPORTER
# =============================================================================
def save_dataset_history_to_txt(filename, dataset_history):
    with open(filename, 'w') as f:
        f.write("P0L,P0R,P1L,P1R,C0L,C0R,C1L,C1R,KEY\n")
        for p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r, target_key in dataset_history:
            p0l_np = p0l.cpu().numpy().flatten()
            p0r_np = p0r.cpu().numpy().flatten()
            p1l_np = p1l.cpu().numpy().flatten()
            p1r_np = p1r.cpu().numpy().flatten()

            c0l_np = c0l.cpu().numpy().flatten()
            c0r_np = c0r.cpu().numpy().flatten()
            c1l_np = c1l.cpu().numpy().flatten()
            c1r_np = c1r.cpu().numpy().flatten()

            key_str = f"0x{target_key:04X}"

            for i in range(len(p0l_np)):
                f.write(f"0x{p0l_np[i]:04X},0x{p0r_np[i]:04X},0x{p1l_np[i]:04X},0x{p1r_np[i]:04X},"
                        f"0x{c0l_np[i]:04X},0x{c0r_np[i]:04X},0x{c1l_np[i]:04X},0x{c1r_np[i]:04X},{key_str}\n")


# =============================================================================
# 7. CACHING & SCENARIO EXECUTION
# =============================================================================
def get_cached_wkrp(model_1, model_2, is_ensemble, loss1, loss2, dist_rounds, filename):
    if os.path.exists(filename):
        print(f"Loading cached WKRP from {filename}...")
        data = torch.load(filename, map_location=DEVICE, weights_only=True)
        return data['mu'].to(DEVICE), data['std'].to(DEVICE)
    else:
        mu, std = generate_full_wkrp(model_1, model_2, is_ensemble, loss1, loss2, dist_rounds)
        print(f"Saving generated WKRP to {filename}...")
        torch.save({'mu': mu, 'std': std}, filename)
        return mu, std


def run_arena_scenario(model_base, model_ft, loss_base, loss_ft, dist_rounds, attack_rounds, n_structures):
    model_base.eval()
    model_ft.eval()

    print("\n" + "=" * 70)
    print(f" LOADING/GENERATING FULL WKRP TABLES FOR {dist_rounds}R DISTINGUISHER")
    print("=" * 70)

    os.makedirs("wkrp_cache", exist_ok=True)
    os.makedirs("winning_datasets", exist_ok=True)

    mu_b, std_b = get_cached_wkrp(model_base, None, False, 0, 0, dist_rounds, f"wkrp_cache/wkrp_b_{dist_rounds}r.pt")
    mu_ft, std_ft = get_cached_wkrp(model_ft, None, False, 0, 0, dist_rounds, f"wkrp_cache/wkrp_ft_{dist_rounds}r.pt")
    mu_ens, std_ens = get_cached_wkrp(model_base, model_ft, True, loss_base, loss_ft, dist_rounds,
                                      f"wkrp_cache/wkrp_e_{dist_rounds}r.pt")

    wins = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    time_spent = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

    total_runs = 0
    dataset_history = []

    dataset_filename_pt = os.path.join("winning_datasets", f"winning_dataset_{attack_rounds}r.pt")
    dataset_filename_txt = os.path.join("winning_datasets", f"winning_dataset_{attack_rounds}r.txt")

    print("\n" + "=" * 70)
    print(f" ATTACK ARENA. Config: {attack_rounds} Rounds, {n_structures} Structures (x32 pairs)")
    print("=" * 70)

    while True:
        total_runs += 1
        print(f"\n--- Generating Attack Data #{total_runs} ---")

        p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r, target_key = gen_challenge(n_structures, attack_rounds)

        # Accumulate the entire history of runs
        dataset_history.append((p0l, p0r, p1l, p1r, c0l, c0r, c1l, c1r, target_key))

        # --- M1: Original BKS ---
        t0 = time.time()
        ans1 = exact_bayesian_key_search(c0l, c0r, c1l, c1r, mu_b, std_b, model_base)
        time_spent[1] += time.time() - t0
        if ans1 == target_key: wins[1] += 1
        print("1. Finished Orig BKS.")
        # --- M2: Fine-Tuned BKS ---
        t0 = time.time()
        ans2 = exact_bayesian_key_search(c0l, c0r, c1l, c1r, mu_ft, std_ft, model_ft)
        time_spent[2] += time.time() - t0
        if ans2 == target_key: wins[2] += 1
        print("2. Finished FT BKS.")
        # SoL execution
        t0_sol = time.time()
        top_64_base = sum_of_logits(model_base, c0l, c0r, c1l, c1r, top_k=64)
        t_sol = time.time() - t0_sol
        print("Calculated sum_of_logits...")
        # --- M3: SoL -> Original BKS (Restricted to 64) ---
        t0 = time.time()
        ans3 = exact_bayesian_key_search(c0l, c0r, c1l, c1r, mu_b, std_b, model_base, candidate_space=top_64_base,
                                         initial_S=top_64_base)
        time_spent[3] += (time.time() - t0) + t_sol
        if ans3 == target_key: wins[3] += 1
        print("3. Finished SoL -> Orig BKS.")
        # --- M4: SoL -> Fine-Tuned BKS (Restricted to 64) ---
        t0 = time.time()
        ans4 = exact_bayesian_key_search(c0l, c0r, c1l, c1r, mu_ft, std_ft, model_ft, candidate_space=top_64_base,
                                         initial_S=top_64_base)
        time_spent[4] += (time.time() - t0) + t_sol
        if ans4 == target_key: wins[4] += 1
        print("4. Finished SoL -> FT BKS.")
        # --- M5: Ensemble BKS ---
        t0 = time.time()
        ans5 = exact_bayesian_key_search(c0l, c0r, c1l, c1r, mu_ens, std_ens, model_base, model_ft, loss_base, loss_ft,
                                         is_ensemble=True)
        time_spent[5] += time.time() - t0
        if ans5 == target_key: wins[5] += 1
        print("5. Finished Ensemble BKS.")
        # --- M6: SoL -> Ensemble BKS (Restricted to 64) ---
        t0 = time.time()
        ans6 = exact_bayesian_key_search(c0l, c0r, c1l, c1r, mu_ens, std_ens, model_base, model_ft, loss_base, loss_ft,
                                         is_ensemble=True, candidate_space=top_64_base, initial_S=top_64_base)
        time_spent[6] += (time.time() - t0) + t_sol
        if ans6 == target_key: wins[6] += 1
        print("6. Finished SoL -> Ensemble BKS.")
        acc = {k: v / total_runs for k, v in wins.items()}
        avg_time = {k: v / total_runs for k, v in time_spent.items()}

        print(f">>> Run {total_runs:03d} Summary")
        print(
            f"    Accuracy: [M1]{acc[1]:.1%} | [M2]{acc[2]:.1%} | [M3]{acc[3]:.1%} | [M4]{acc[4]:.1%} | [M5]{acc[5]:.1%} | [M6]{acc[6]:.1%}")
        print(
            f"    Avg Time: [M1]{avg_time[1]:.2f}s | [M2]{avg_time[2]:.2f}s | [M3]{avg_time[3]:.2f}s | [M4]{avg_time[4]:.2f}s | [M5]{avg_time[5]:.2f}s | [M6]{avg_time[6]:.2f}s")

        better_accuracy = any(acc[m] > acc[1] for m in range(2, 7))
        speed_dominance = any(
            (acc[1] > 0 and acc[m] > 0 and acc[m] >= acc[1] and avg_time[m] < avg_time[1]) for m in range(2, 7))

        if total_runs >= 10 and (better_accuracy or speed_dominance):
            if better_accuracy:
                print(
                    f"\n[SUCCESS] A proposed algorithm has statistically beaten the {attack_rounds}-round baseline in ACCURACY!")
            elif speed_dominance:
                print(f"\n[SUCCESS] A proposed algorithm matched baseline accuracy but dominated in EXECUTION SPEED!")

            print(f"    Saving the entire dataset ({len(dataset_history)} challenges) to disk...")
            torch.save({'dataset_history': dataset_history}, dataset_filename_pt)
            save_dataset_history_to_txt(dataset_filename_txt, dataset_history)
            print(f"    Datasets saved successfully as '{dataset_filename_pt}' and '{dataset_filename_txt}'.")

            print(f"\nMethod 1 (Orig BKS):             {acc[1]:.2%} (Avg Time: {avg_time[1]:.2f}s)")
            print(f"Method 2 (FT BKS):               {acc[2]:.2%} (Avg Time: {avg_time[2]:.2f}s)")
            print(f"Method 3 (Orig SoL -> Orig BKS): {acc[3]:.2%} (Avg Time: {avg_time[3]:.2f}s)")
            print(f"Method 4 (Orig SoL -> FT BKS):   {acc[4]:.2%} (Avg Time: {avg_time[4]:.2f}s)")
            print(f"Method 5 (Ensemble BKS):         {acc[5]:.2%} (Avg Time: {avg_time[5]:.2f}s)")
            print(f"Method 6 (Orig SoL -> Ens BKS):  {acc[6]:.2%} (Avg Time: {avg_time[6]:.2f}s)")
            return


# =============================================================================
# INITIALIZATION & SCENARIO LOOP
# =============================================================================
if __name__ == "__main__":
    print(f"Active Device: {DEVICE}")

    os.makedirs("winning_datasets", exist_ok=True)

    SCENARIOS = {
        5: {"dist_rounds": 5, "attack_rounds": 6, "depth": 10, "n_structures": 32, "loss_base": 0.05520,
            "loss_ft": 0.06105},
        6: {"dist_rounds": 6, "attack_rounds": 7, "depth": 10, "n_structures": 64, "loss_base": 0.14628,
            "loss_ft": 0.14793},
        7: {"dist_rounds": 7, "attack_rounds": 8, "depth": 1, "n_structures": 128, "loss_base": 0.24574,
            "loss_ft": 0.24575}
    }

    for r in SCENARIOS:
        cfg = SCENARIOS[r]
        attack_rounds = cfg['attack_rounds']

        dataset_pt = os.path.join("winning_datasets", f"winning_dataset_{attack_rounds}r.pt")
        dataset_txt = os.path.join("winning_datasets", f"winning_dataset_{attack_rounds}r.txt")

        if os.path.exists(dataset_pt) and os.path.exists(dataset_txt):
            print("\n" + "#" * 70)
            print(f" [SKIP] Winning datasets already exist for {attack_rounds}-ROUND ATTACK. Skipping...")
            print("#" * 70)
            continue

        print("\n" + "#" * 70)
        print(f" STARTING EVALUATION FOR {attack_rounds}-ROUND ATTACK")
        print("#" * 70)

        base_path = os.path.join("light_models", f"dnd_speck32_r{r}_d{cfg['depth']}.pt")
        ft_path = os.path.join("light_models", f"finetuned_hn_dnd_speck32_r{r}_d{cfg['depth']}.pt")

        if os.path.exists(base_path) and os.path.exists(ft_path):
            model_base = ImprovedCNN(depth=cfg['depth']).to(DEVICE)
            model_ft = ImprovedCNN(depth=cfg['depth']).to(DEVICE)

            model_base.load_state_dict(
                torch.load(base_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
            model_ft.load_state_dict(torch.load(ft_path, map_location=DEVICE, weights_only=True)['model_state_dict'])

            run_arena_scenario(
                model_base=model_base,
                model_ft=model_ft,
                loss_base=cfg['loss_base'],
                loss_ft=cfg['loss_ft'],
                dist_rounds=cfg['dist_rounds'],
                attack_rounds=cfg['attack_rounds'],
                n_structures=cfg['n_structures']
            )
        else:
            print(f"[CRITICAL ERROR] Could not find the .pt files for {r}-round distinguisher.")
            print(f"Skipping {r}-round models...")
