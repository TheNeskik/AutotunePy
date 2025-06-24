# DEPRECATED: Ce module n'est plus utilisé dans le pipeline principal. Gardé pour référence et recherche uniquement.
# DEPRECATED: This module is no longer used in the main pipeline. Kept for reference and research purposes only.
#
# Conseils sur le nombre de jours optimal :
# - Pour la stabilité, 14 à 30 jours est un bon compromis (assez de données, pas trop vieux)
# - Si le patient a changé de schéma récemment, privilégier 7 à 14 jours
# - Pour un profil très robuste, 30 jours (si pas de changement majeur)
# - Pour tester la réactivité, 7 jours
#
# Tips for optimal number of days:
# - For stability, 14 to 30 days is a good compromise (enough data, not too old)
# - If the patient recently changed their regimen, prefer 7 to 14 days
# - For a very robust profile, use 30 days (if no major change)
# - To test reactivity, use 7 days
#
# Utilisation / Usage:
# transitions = build_rl_dataset('features_debug.csv', days=7)
# Q = fitted_q_iteration(transitions, use_gpu=True)
# Pour prédire la meilleure action pour un état : Q.predict(np.concatenate([state, action]).reshape(1, -1))
# To predict the best action for a state: Q.predict(np.concatenate([state, action]).reshape(1, -1))

import pandas as pd
import numpy as np
from collections import namedtuple

# Pour RL batch simple : Q-learning tabulaire ou fitted Q-iteration (FQI)
# Ici, on prépare un pipeline RL batch sur les données features_debug.csv
# Reward = -abs(glucose-100) - alpha*|variation|
# Action = (basal, isf, csf) arrondis réglementairement
# State = toutes les features historiques/contextuelles

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

def discretize_action(basal, isf, csf):
    # Arrondi réglementaire
    basal_ = round(basal / 0.05) * 0.05
    isf_ = int(round(isf))
    csf_ = int(round(csf / 10) * 10)
    return (basal_, isf_, csf_)

def compute_reward(glucose, target=100, alpha=0.1, prev_glucose=None):
    # Reward = -|gly - 100| - alpha*|variation|
    var = abs(glucose - prev_glucose) if prev_glucose is not None else 0
    return -abs(glucose - target) - alpha * var

def build_rl_dataset(csv_path, days=None):
    print(f"[RL] Chargement du dataset depuis {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['datetime_local'])
    # Correction : parsing robuste des datetimes (timezone, ISO8601, etc.)
    df['datetime_local'] = pd.to_datetime(df['datetime_local'], format='mixed', errors='coerce')
    if days is not None:
        max_date = df['datetime_local'].max()
        min_date = max_date - pd.Timedelta(days=days)
        df = df[df['datetime_local'] >= min_date]
    print(f"[RL] {len(df)} lignes après filtrage sur {days} jours.")
    # Exclut les colonnes non numériques et cibles, mais garde toutes les features temporelles dérivées
    exclude_cols = ['timestamp_ms','datetime_local','basal_profile','isf','csf','glucose','timestamp','date']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    states = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    actions = [discretize_action(b, i, c) for b, i, c in zip(df['basal_profile'], df['isf'], df['csf'])]
    rewards = []
    for i in range(len(df)):
        prev_g = df['sgv'].iloc[i-1] if i > 0 else None
        rewards.append(compute_reward(df['sgv'].iloc[i], prev_glucose=prev_g))
    # --- Diagnostics avancés ---
    print("[RL][Diag] Stats states (première ligne):", states[0] if len(states) else 'Empty')
    print("[RL][Diag] States min/max:", np.min(states) if len(states) else 'Empty', np.max(states) if len(states) else 'Empty')
    print("[RL][Diag] Actions uniques:", set(actions))
    print("[RL][Diag] Rewards min/max:", np.min(rewards) if len(rewards) else 'Empty', np.max(rewards) if len(rewards) else 'Empty')
    print("[RL][Diag] sgv min/max:", df['sgv'].min() if 'sgv' in df else 'NA', df['sgv'].max() if 'sgv' in df else 'NA')
    # Variance des features d'état
    print("[RL][Diag] Variance des features d'état:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: var={np.var(states[:,i]):.2f}")
    # Moyenne de la reward par action
    from collections import defaultdict
    reward_by_action = defaultdict(list)
    for a, r in zip(actions, rewards):
        reward_by_action[a].append(r)
    print("[RL][Diag] Moyenne reward par action:")
    for a, vals in reward_by_action.items():
        print(f"  {a}: mean={np.mean(vals):.2f}, n={len(vals)}")
    # ---
    print(f"[RL] Calcul des rewards terminé.")
    next_states = np.roll(states, -1, axis=0)
    done = np.zeros(len(df), dtype=bool)
    done[-1] = True
    transitions = [Transition(states[i], actions[i], rewards[i], next_states[i], done[i]) for i in range(len(df))]
    print(f"[RL] Dataset RL prêt : {len(transitions)} transitions.")
    return transitions

# Fitted Q Iteration avec XGBoost (GPU si dispo)
def fitted_q_iteration(transitions, n_iter=20, use_gpu=True):
    import xgboost as xgb
    print(f"[RL] Démarrage Fitted Q Iteration (XGBoost) sur {len(transitions)} transitions...")
    X = []
    y = []
    for t in transitions:
        X.append(np.concatenate([t.state, np.array(t.action)]))
        y.append(t.reward)
    X = np.array(X)
    y = np.array(y)
    # Grille d'actions basée sur les valeurs du dataset
    all_actions = np.array([t.action for t in transitions])
    basal_vals = np.unique(all_actions[:,0])
    isf_vals = np.unique(all_actions[:,1])
    csf_vals = np.unique(all_actions[:,2])
    print(f"[RL][Diag] Grille d'actions FQI: basal={basal_vals}, isf={isf_vals}, csf={csf_vals}")
    if use_gpu:
        Q = xgb.XGBRegressor(n_estimators=50, device='cuda', n_jobs=-1, verbosity=1)
    else:
        Q = xgb.XGBRegressor(n_estimators=50, tree_method='hist', n_jobs=-1, verbosity=1)
    Q.fit(X, y)
    print("[RL] Initial fit terminé.")
    for it in range(n_iter):
        print(f"[RL] Itération FQI {it+1}/{n_iter}...")
        y_new = []
        for idx, t in enumerate(transitions):
            if idx % 500 == 0:
                print(f"[RL]  Transition {idx}/{len(transitions)}...")
            actions_grid = [(b, i, c)
                for b in basal_vals
                for i in isf_vals
                for c in csf_vals]
            actions_arr = np.array(actions_grid)
            states_arr = np.repeat(t.next_state.reshape(1, -1), len(actions_grid), axis=0)
            X_next = np.hstack([states_arr, actions_arr])
            Q_next = Q.predict(X_next)
            max_Q = np.max(Q_next) if len(Q_next) > 0 else 0
            y_new.append(t.reward + (0 if t.done else 0.99 * max_Q))
        Q.fit(X, y_new)
        print(f"[RL] Fit Q terminé pour itération {it+1}.")
    print("[RL] Fitted Q Iteration terminé.")
    return Q

def extract_best_profile(Q, transitions):
    """
    Pour chaque heure (0-23), extrait l'action (basal, isf, csf) qui maximise Q sur les états de cette heure.
    Retourne un dict {hour: (basal, isf, csf)} et un texte formaté.
    """
    import numpy as np
    # Extraire les valeurs uniques d'action du dataset
    all_actions = np.array([t.action for t in transitions])
    basal_vals = np.unique(all_actions[:,0])
    isf_vals = np.unique(all_actions[:,1])
    csf_vals = np.unique(all_actions[:,2])
    print(f"[RL][Diag] Grille d'actions utilisée: basal={basal_vals}, isf={isf_vals}, csf={csf_vals}")
    hours = np.arange(len(transitions)) % 24
    best_profile = {}
    result = ["<b>Profil RL optimal (Q-fitted, 24h) :</b>"]
    for h in range(24):
        idxs = np.where(hours == h)[0]
        if len(idxs) == 0:
            best_profile[h] = (np.nan, np.nan, np.nan)
            result.append(f"{h:02d}h  Basal: NA  ISF: NA  CSF: NA")
            continue
        # Moyenne de l'état pour cette heure
        state = np.mean([transitions[i].state for i in idxs], axis=0)
        # Grille d'actions basée sur les valeurs du dataset
        actions_grid = [(b, i, c)
            for b in basal_vals
            for i in isf_vals
            for c in csf_vals]
        actions_arr = np.array(actions_grid)
        states_arr = np.repeat(state.reshape(1, -1), len(actions_grid), axis=0)
        X = np.hstack([states_arr, actions_arr])
        Q_vals = Q.predict(X)
        print(f"[RL][Diag] Heure {h}: Q min={np.min(Q_vals):.2f}, max={np.max(Q_vals):.2f}, argmax={np.argmax(Q_vals)}")
        best_idx = np.argmax(Q_vals)
        best_action = actions_grid[best_idx]
        best_profile[h] = best_action
        result.append(f"{h:02d}h  Basal: {best_action[0]:.2f}  ISF: {int(best_action[1])}  CSF: {int(best_action[2])}")
    return best_profile, "<br>".join(result)

def export_optimal_profile_by_hour(csv_path, days=None, output_csv="profil_optimal_par_heure.csv"): 
    """
    Entraîne un modèle de reward (XGBoost) sur reward ~ état+action, puis pour chaque heure,
    balaye toutes les actions possibles et exporte l'action qui maximise la reward prédite.
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    df = pd.read_csv(csv_path, parse_dates=["datetime_local"])
    # Correction : forcer le parsing datetime
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors='coerce')
    if days is not None:
        max_date = df["datetime_local"].max()
        min_date = max_date - pd.Timedelta(days=days)
        df = df[df["datetime_local"] >= min_date]
    df["hour"] = pd.to_datetime(df["datetime_local"]).dt.hour
    # Features d'état (hors action)
    exclude_cols = ['timestamp_ms','datetime_local','basal_profile','isf','csf','glucose','timestamp','date','reward']
    state_cols = [c for c in df.columns if c not in exclude_cols and c not in ["hour"]]
    # Actions
    action_cols = ["basal_profile", "isf", "csf"]
    # Reward
    y = df["sgv"].values  # ou une autre reward si besoin
    # X = état + action
    X = np.hstack([
        df[state_cols].fillna(0).to_numpy(dtype=np.float32),
        df[action_cols].fillna(0).to_numpy(dtype=np.float32)
    ])
    model = xgb.XGBRegressor(n_estimators=50, tree_method='hist', n_jobs=-1, verbosity=1)
    model.fit(X, y)
    # Grille d'actions possibles (uniques dans le dataset)
    basal_vals = np.unique(df["basal_profile"].values)
    isf_vals = np.unique(df["isf"].values)
    csf_vals = np.unique(df["csf"].values)
    result = []
    for h in range(24):
        dfh = df[df["hour"] == h]
        if len(dfh) == 0:
            result.append([h, np.nan, np.nan, np.nan])
            continue
        state_mean = dfh[state_cols].mean().values.astype(np.float32)
        best_reward = -np.inf
        best_action = (np.nan, np.nan, np.nan)
        for b in basal_vals:
            for i in isf_vals:
                for c in csf_vals:
                    x = np.concatenate([state_mean, [b, i, c]])
                    pred = model.predict(x.reshape(1, -1))[0]
                    if pred > best_reward:
                        best_reward = pred
                        best_action = (b, i, c)
        result.append([h, *best_action])
    outdf = pd.DataFrame(result, columns=["hour", "basal_opt", "isf_opt", "csf_opt"])
    outdf.to_csv(output_csv, index=False)
    print(f"[Profil] Profil optimal par heure exporté dans {output_csv}")

def pipeline_optimal_profile(csv_path, days=14, output_csv="profil_optimal_par_heure.csv"): 
    """
    Pipeline complet :
    - Génère le dataset RL (filtrage days, features, actions, rewards)
    - Entraîne le modèle reward~état+action
    - Exporte le profil optimal par heure selon le modèle
    """
    print(f"[PIPELINE] Génération du dataset RL sur {days} jours...")
    transitions = build_rl_dataset(csv_path, days=days)
    print(f"[PIPELINE] Extraction du profil optimal par heure...")
    export_optimal_profile_by_hour(csv_path, days=days, output_csv=output_csv)
    print(f"[PIPELINE] Terminé. Profil exporté dans {output_csv}")
