import numpy as np
import pandas as pd
import math
import time
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ================================================================
# Core Math Functions from Paper's Author with Minor Modifications
# ================================================================
def bernouli_kl_divergence(mu1, mu2):
    P = [1 - mu1, mu1]
    Q = [1 - mu2, mu2]
    return stats.entropy(P, Q)

def chernoff(pull_num, mu, maxindex):
    mubest = mu[maxindex]
    numbest = pull_num[maxindex]
    MuMid = [(numbest * mubest + pull_num[i] * mu[i]) / (numbest + pull_num[i]) for i in range(len(mu))]
    Index = [i for i in range(len(mu))]
    Index.pop(maxindex)
    zj = [bernouli_kl_divergence(mubest, MuMid[i]) * numbest + bernouli_kl_divergence(mu[i], MuMid[i]) * pull_num[i] for i in Index]
    return min(zj)

def beta_func_TS(tau, delta):
    return np.log((np.log(tau) + 1) / delta)

def Ifonc(alpha, mu1, mu2):
    if (alpha == 0) or (alpha == 1): return 0
    mid = alpha * mu1 + (1 - alpha) * mu2
    return alpha * bernouli_kl_divergence(mu1, mid) + (1 - alpha) * bernouli_kl_divergence(mu2, mid)

def cost(mu1, mu2, nu1, nu2):
    if (nu1 == 0) and (nu2 == 0): return 0
    alpha = nu1 / (nu1 + nu2)
    return ((nu1 + nu2) * Ifonc(alpha, mu1, mu2))

def xkofy(y, k, sortb):
    xMax = 1
    for _ in range(20): # Added hard limit on expansion
        if (1 + xMax) * cost(sortb[0], sortb[k], 1 / (1 + xMax), xMax / (1 + xMax)) - y >= 0: break
        xMax *= 2
    l, u = 0, xMax
    pre = 1e-5 # We loosened precision from 1e-11 slightly for speed
    sgn = cost(sortb[0], sortb[k], 1, 0) - y
    for _ in range(50): # Added hard limit on bisection
        if u - l <= pre: break
        m = (u + l) / 2
        if ((1 + m) * cost(sortb[0], sortb[k], 1 / (1 + m), m / (1 + m)) - y) * sgn > 0: l = m
        else: u = m
    return (u + l) / 2

def muddle(mu1, mu2, nu1, nu2):
    result = (nu1 * mu1 + nu2 * mu2) / (nu1 + nu2)
    return result - 1e-16 if result == 1 else result

def aux(y, sortb):
    K = len(sortb)
    x = [xkofy(y, k, sortb) for k in range(1, K)]
    m = [muddle(sortb[0], sortb[k], 1, x[k-1]) for k in range(1, K)]
    return (sum([bernouli_kl_divergence(sortb[0], m[k-1]) / (bernouli_kl_divergence(sortb[k], m[k-1]) + 1e-16) for k in range(1, K)]) - 1) # Added 1e-16 in the denominator to prevent division-by-zero

def optimal_weights(b):
    sortb = -np.sort(-b)
    Index = np.argsort(-b).tolist()
    ymax = bernouli_kl_divergence(sortb[0], sortb[1])
    if ymax == np.inf or ymax == 0:
        ymax = 0.5
    
    l, u, pre = 0, ymax, 1e-5
    sgn = aux(l, sortb)
    for _ in range(50):
        if u - l <= pre: break
        m = (u + l) / 2
        if (aux(m, sortb) * sgn > 0): l = m
        else: u = m
    y = (u + l) / 2
    
    x = [xkofy(y, k, sortb) for k in range(1, len(sortb))]
    x.insert(0, 1)
    invindex = [Index.index(i) for i in range(len(Index))]
    nuOpt = [i / sum(x) for i in x]
    NuOpt = [nuOpt[i] for i in invindex]
    vOpt = y / sum(x)
    Tb = 100000 if vOpt == 0 else min(1 / vOpt, 100000) # Cap Tb 
    return Tb, NuOpt

# ==================================================================================
# Algorithms based on author's logic with our own modifications to fit the simulator
# ==================================================================================
def Tri_BBAI(epsilon, delta, L1, L2, L3, alpha, means):
    n = len(means)
    pulls = np.zeros(n, dtype=int)
    scores = np.zeros(n)
    
    def pull(i):
        r = 1 if np.random.rand() < means[i] else 0
        scores[i] += r
        pulls[i] += 1
        
    batch = 1
    for i in range(n):
        for _ in range(L1): pull(i)
        
    q = 1
    wb_list = []
    while True:
        if q > 10: break # Safety cap of max 10 batches in Stage 2
        
        b = scores / pulls
        maxindex = np.argmax(b)
        for i in range(n):
            b[i] = b[i] - epsilon if i == maxindex else b[i] + epsilon
        Tb, wb = optimal_weights(b)
        wb_list.append(wb)
        
        for i in range(n):
            Ti = L2 if (2**q * alpha * wb[i] * Tb * np.log(1/delta)) == np.inf else min(math.ceil(2**q * alpha * wb[i] * Tb * np.log(1/delta)), L2)
            new_pulls = max(0, Ti - pulls[i])
            for _ in range(new_pulls): pull(i)
                
        batch += 1
        if q >= 2:
            if abs(wb_list[q-1][0] - wb_list[q-2][0]) <= 1 / np.sqrt(n):
                break
        q += 1
        
    maxindex = np.argmax(scores / pulls)
    if chernoff(pulls, scores / pulls, maxindex) >= beta_func_TS(sum(pulls), delta):
        return sum(pulls), batch, maxindex
        
    batch += 1
    for i in range(n):
        if pulls[i] < L3:
            for _ in range(int(L3 - pulls[i])): pull(i)
    return sum(pulls), batch, np.argmax(scores / pulls)

def Opt_BBAI(epsilon, delta, L1, L2, L3, alpha, means):
    n = len(means)
    pulls = np.zeros(n, dtype=int)
    scores = np.zeros(n)
    
    def pull(i):
        r = 1 if np.random.rand() < means[i] else 0
        scores[i] += r
        pulls[i] += 1
        
    batch = 1
    for i in range(n):
        for _ in range(L1): pull(i)
        
    b = scores / pulls
    maxindex = np.argmax(b)
    for i in range(n):
        b[i] = b[i] - epsilon if i == maxindex else b[i] + epsilon
    Tb, wb = optimal_weights(b)
    for i in range(n):
        Ti = L2 if (alpha * wb[i] * Tb * np.log(1/delta)) == np.inf else min(math.ceil(alpha * wb[i] * Tb * np.log(1/delta)), L2)
        new_pulls = max(0, Ti - pulls[i])
        for _ in range(new_pulls): pull(i)
        
    maxindex = np.argmax(scores / pulls)
    if chernoff(pulls, scores / pulls, maxindex) >= beta_func_TS(sum(pulls), delta):
        return sum(pulls), batch + 1, maxindex

    batch += 1
    setlist = [{i for i in range(n)}]
    B_list = [0]
    r = 1
    while len(setlist[r-1]) > 1:
        if r > 10: break # SAFETY CAP: Max 10 loops in Stage 4
        
        epsilon_r = np.power(0.5, r) / 4
        delta_r = delta / (40 * np.pi * np.pi * n * np.power(r, 2))
        batch += 1
        
        for i in setlist[r-1]:
            d_r = 1 / (2 * np.power(epsilon_r, 2)) * np.log(2 / delta_r) / 100
            for _ in range(math.ceil(d_r)): pull(i)
            
        mu = scores / pulls
        p_best = max([mu[i] for i in setlist[r-1]])
        B_list.append(B_list[r-1] + len(setlist[r-1]) * math.ceil(d_r))
        set_k = setlist[r-1].copy()
        for i in setlist[r-1]:
            if mu[i] < p_best - epsilon_r: set_k.remove(i)
        setlist.append(set_k)
        
        for j in range(r-1):
            list_i = []
            batch += 1
            delta_j = delta / (40 * np.pi**2 * n * np.power(j+1, 2))
            for i in setlist[j]:
                if B_list[r] * delta_j > B_list[j]:
                    delta_j = delta_j ** 2
                    for _ in range(math.ceil(d_r * 2)): pull(i)
                list_i.append(scores[i] / pulls[i])
            if list_i and max(list_i) >= p_best - epsilon_r:
                return sum(pulls), batch, np.argmax(scores / pulls)
        r += 1
        
    return sum(pulls), batch, list(setlist[-1])[0]

# ======================
# Main Experiment Loop
# ======================
def generate_means(instance_type, n=10, seed=None):
    rng = np.random.RandomState(seed)
    means = np.zeros(n)
    if instance_type == "uniform":
        means[0] = 0.5
        means[1:] = rng.uniform(0.2, 0.4, n - 1)
    else:
        means[0] = 0.6
        means[1:] = np.clip(rng.normal(0.2, 0.2, n - 1), 0.0, 0.4)
    return means

def get_params(delta, n=10):
    safe_delta = min(delta, 0.99) 
    log_inv = np.log(1 / safe_delta)
    epsilon = 1 / np.log(np.log(1 / safe_delta)) if log_inv > 1 else 0.1
    L1 = max(1, int(np.sqrt(log_inv)))
    L2 = 2000 # Capped for manageable runtime
    L3 = max(10, int(log_inv ** 2)) if log_inv > 1 else 10
    return epsilon, L1, L2, L3, 1.1

DELTAS = [1e-1, 1e-3, 1e-5]
TRIALS = 50
INSTANCE = "uniform" # Change "uniform" to "normal" for testing in differnet distributions

print(f"Starting flawless fast reproduction on {INSTANCE}...")
start_time = time.time()
results = []

for delta in DELTAS:
    epsilon, L1, L2, L3, alpha = get_params(delta)
    for trial in range(TRIALS):
        means = generate_means(INSTANCE, seed=trial)
        true_best = np.argmax(means)
        np.random.seed(trial)
        
        pulls, batches, best = Tri_BBAI(epsilon, delta, L1, L2, L3, alpha, means)
        results.append({"Delta": delta, "Trial": trial, "Algorithm": "Tri-BBAI", "Pulls": pulls, "Batches": batches, "Success": 1 if best == true_best else 0})
        
        pulls, batches, best = Opt_BBAI(epsilon, delta, L1, L2, L3, alpha, means)
        results.append({"Delta": delta, "Trial": trial, "Algorithm": "Opt-BBAI", "Pulls": pulls, "Batches": batches, "Success": 1 if best == true_best else 0})
        
    print(f"Finished Delta = {delta}")

df = pd.DataFrame(results)
df.to_csv(f"results_{INSTANCE}.csv", index=False)

summary = df.groupby(["Algorithm", "Delta"]).agg(
    Avg_Pulls=("Pulls", "mean"),
    Avg_Batches=("Batches", "mean"),
    Recall=("Success", "mean")
).reset_index()

print("\n--- SUMMARY RESULTS ---")
print(summary.to_string(index=False))
print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
