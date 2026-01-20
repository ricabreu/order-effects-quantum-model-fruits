import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

# -------------------------
# CONFIG
# -------------------------
XLSX_PATH = "Order_Effects_paper.xlsx"   # adjust if needed
BETA = 1.0

# In your Excel, columns are:
# Apple:  SGM1, SGM2  (S->G),  GSM1, GSM2  (G->S)
# Banana: SGB1, SGB2,         GSB1, GSB2
# Lemon:  SGL1, SGL2,         GSL1, GSL2

CONDITIONS = {
    "apple":  {"S->G": ("SGM1", "SGM2"), "G->S": ("GSM1", "GSM2")},
    "banana": {"S->G": ("SGB1", "SGB2"), "G->S": ("GSB1", "GSB2")},
    "lemon":  {"S->G": ("SGL1", "SGL2"), "G->S": ("GSL1", "GSL2")},
}

YES = {"SIM", "YES", "Y", "1", 1, True}
NO  = {"NÃO", "NAO", "NO", "N", "0", 0, False}

# -------------------------
# HELPERS: data
# -------------------------
def to_binary(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().upper()
    if x in YES:
        return 1.0
    if x in NO:
        return 0.0
    return np.nan

def load_data(path=XLSX_PATH):
    df = pd.read_excel(path)

    # Your file includes a first row with "IDADE", "Género" etc. as labels -> drop it
    if "SOC.1" in df.columns and str(df.loc[0, "SOC.1"]).strip().upper() == "IDADE":
        df = df.iloc[1:].copy()

    # Convert response columns
    response_cols = [c for c in df.columns if c not in ["SOC", "SOC.1"]]
    for c in response_cols:
        df[c] = df[c].apply(to_binary)

    return df

def empirical_conditional(df, q1, q2):
    """
    Compute: P(q2=1 | q1=1) from a between-subjects condition (q1 asked first).
    Returns numerator, denom, estimate.
    """
    sub = df[[q1, q2]].dropna(subset=[q1, q2]).copy()
    denom = int((sub[q1] == 1).sum())
    numer = int(((sub[q1] == 1) & (sub[q2] == 1)).sum())
    p = numer / denom if denom > 0 else np.nan
    return numer, denom, p

def compute_empirical(df):
    """
    For each fruit:
      P_hat(G|S) from S->G columns
      P_hat(S|G) from G->S columns
    """
    results = {}
    for fruit, conds in CONDITIONS.items():
        qS_then_G = conds["S->G"]
        qG_then_S = conds["G->S"]

        # In S->G condition, q1=S, q2=G  => P_hat(G|S)
        n1, d1, pG_given_S = empirical_conditional(df, qS_then_G[0], qS_then_G[1])

        # In G->S condition, q1=G, q2=S  => P_hat(S|G)
        n2, d2, pS_given_G = empirical_conditional(df, qG_then_S[0], qG_then_S[1])

        results[fruit] = {
            "pG_given_S": pG_given_S,
            "pS_given_G": pS_given_G,
            "counts_G_given_S": (n1, d1),
            "counts_S_given_G": (n2, d2),
        }
    return results

# -------------------------
# MODEL: quantum-like
# -------------------------
DIM = 4  # |SG>,|SB>,|RG>,|RB>

def P_S():
    # Projector for "Sweet=Yes": |SG><SG| + |SB><SB|
    P = np.zeros((DIM, DIM))
    P[0,0] = 1.0
    P[1,1] = 1.0
    return P

def P_G(theta):
    # |g> = cosθ|SG> + sinθ|RG>
    g = np.zeros((DIM, 1))
    g[0,0] = np.cos(theta)
    g[2,0] = np.sin(theta)
    return g @ g.T

def A_coupling():
    """
    Minimal symmetric coupling.
    You can edit this to match your paper's exact A if you fixed one.
    """
    A = np.zeros((DIM, DIM))
    A[0,2] = A[2,0] = 1.0  # SG <-> RG
    A[1,3] = A[3,1] = 1.0  # SB <-> RB
    return A

def gibbs_state(H, beta=BETA):
    X = expm(-beta * H)
    return X / np.trace(X)

def model_conditionals(params):
    """
    params = [e2, e3, e4, Gamma, theta]
    H0 = diag(0, e2, e3, e4)
    H  = H0 - Gamma * A
    Returns P_model(G|S), P_model(S|G)
    """
    e2, e3, e4, Gamma, theta = params
    H0 = np.diag([0.0, e2, e3, e4])
    H = H0 - Gamma * A_coupling()
    rho = gibbs_state(H)

    PS = P_S()
    PG = P_G(theta)

    # Lüders conditional probabilities
    pS = np.trace(PS @ rho)
    pG = np.trace(PG @ rho)

    pG_given_S = np.trace(PG @ PS @ rho @ PS) / pS if pS > 1e-12 else np.nan
    pS_given_G = np.trace(PS @ PG @ rho @ PG) / pG if pG > 1e-12 else np.nan

    # real parts (numerical noise)
    return float(np.real(pG_given_S)), float(np.real(pS_given_G))

def fit_gamma_theta(pG_given_S_emp, pS_given_G_emp):
    """
    Fit [e2,e3,e4,Gamma,theta] by least squares to match the two empirical conditionals.
    Returns best params and fit error.
    """
    def loss(x):
        pG_S, pS_G = model_conditionals(x)
        return (pG_S - pG_given_S_emp)**2 + (pS_G - pS_given_G_emp)**2

    x0 = np.array([0.2, 0.2, 0.2, 0.05, 0.7])  # initial guess
    bounds = [(-5, 5), (-5, 5), (-5, 5), (0, 2), (0, np.pi/2)]
    res = minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")
    return res

# -------------------------
# RUN: full mapping
# -------------------------
def main():
    df = load_data(XLSX_PATH)
    emp = compute_empirical(df)

    print("\n=== EMPIRICAL -> MODEL -> FIT (per fruit) ===\n")

    for fruit, vals in emp.items():
        pG_S_emp = vals["pG_given_S"]
        pS_G_emp = vals["pS_given_G"]
        (n1, d1) = vals["counts_G_given_S"]
        (n2, d2) = vals["counts_S_given_G"]

        # Fit
        res = fit_gamma_theta(pG_S_emp, pS_G_emp)
        e2, e3, e4, Gamma_hat, theta_hat = res.x

        # Model prediction at fitted params
        pG_S_mod, pS_G_mod = model_conditionals(res.x)

        print(f"--- {fruit.upper()} ---")
        print(f"Empirical  P^(G|S) = {n1}/{d1} = {pG_S_emp:.3f}")
        print(f"Empirical  P^(S|G) = {n2}/{d2} = {pS_G_emp:.3f}")
        print(f"Model      P_model(G|S) = {pG_S_mod:.3f}")
        print(f"Model      P_model(S|G) = {pS_G_mod:.3f}")
        print(f"Fit params Gamma = {Gamma_hat:.4f}, theta = {theta_hat:.4f} rad")
        print(f"Optional energies: e2={e2:.3f}, e3={e3:.3f}, e4={e4:.3f}")
        print(f"Fit loss = {res.fun:.6f}\n")

if __name__ == "__main__":
    main()
