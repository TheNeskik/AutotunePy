import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.ndimage import gaussian_filter1d

# 1. Préparation des données

def prepare_data(csv_path, dropna=True, filter_perturbations=True):
    """
    Charge les données, crée les cibles multi-horizon et les features dérivées.
    Optionnel : filtre les perturbations (repas, bolus, variations rapides).
    """
    df = pd.read_csv(csv_path, parse_dates=['datetime_local'])
    df = df.sort_values('datetime_local')

    # Création des cibles à t+30, t+60, t+90 min (suppose 5 min/ligne)
    df['sgv_t+30'] = df['sgv'].shift(-6)
    df['sgv_t+60'] = df['sgv'].shift(-12)
    df['sgv_t+90'] = df['sgv'].shift(-18)

    # Dérivées glycémie
    df['sgv_delta'] = df['sgv'].diff().fillna(0)
    df['sgv_rolling_mean'] = df['sgv'].rolling(window=6, min_periods=1).mean()
    df['sgv_rolling_std'] = df['sgv'].rolling(window=6, min_periods=1).std().fillna(0)

    # Filtrage perturbations
    df['perturbé'] = (
        (df['cob_val'] > 10) |
        (df['iob_val'] > 2) |
        (df['sgv_delta'].abs() > 10)
    )
    if filter_perturbations:
        df = df[~df['perturbé']]
    df["tail_exposure"] = (
        df["iob_val"].rolling(window=12, min_periods=1).mean()
    )

    if dropna:
        df = df.dropna(subset=['sgv_t+30', 'sgv_t+60', 'sgv_t+90'])

    feature_cols = [
        'sgv', 'iob_val', 'cob_val', 'hour', 'weekday',
        'basal_profile', 'isf', 'csf',
        'sgv_delta', 'sgv_rolling_mean', 'sgv_rolling_std', 'tail_exposure',
        'Delta', 'ShortAvgDelta', 'LongAvgDelta', 'glucose_mean_30min', 'glucose_std_30min', 'glucose_min_30min', 'glucose_max_30min', 'glucose_mean_60min', 'glucose_std_60min', 'glucose_min_60min', 'glucose_max_60min', 'glucose_slope', 'glucose_slope_5min_ago', 'glucose_accel', 'glucose_rolling_min_30min', 'glucose_rolling_max_30min', 'glucose_rolling_min_60min', 'glucose_rolling_max_60min', 'meal_bolus', 'meal_carbs', 'iob_val_5min_ago', 'iob_val_15min_ago', 'iob_val_30min_ago', 'iob_val_60min_ago', 'cob_val_5min_ago', 'cob_val_15min_ago', 'cob_val_30min_ago', 'cob_val_60min_ago'
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype(float)

    X = df[feature_cols].fillna(0)
    y = df[['sgv_t+30', 'sgv_t+60', 'sgv_t+90']]
    return X, y, df, feature_cols

# 2. Entraînement du simulateur multi-horizon (CatBoost)

def train_catboost_multioutput(X, y, save_path='model_catboost_multi.cbm', n_splits=10):
    """
    Entraîne un CatBoostRegressor multi-sortie avec validation croisée temporelle.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_maes = []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostRegressor(loss_function='MultiRMSE', verbose=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = np.mean(np.abs(y_pred - y_test.values), axis=0)
        print(f"[Fold {i+1}] MAE t+30: {mae[0]:.2f}, t+60: {mae[1]:.2f}, t+90: {mae[2]:.2f}")
        all_maes.append(mae)
        log_model_quality(y_test.values, y_pred)

    avg_mae = np.mean(all_maes, axis=0)
    print(f"\n✅ Moyenne MAE — t+30: {avg_mae[0]:.2f}, t+60: {avg_mae[1]:.2f}, t+90: {avg_mae[2]:.2f}")

    final_model = CatBoostRegressor(loss_function='MultiRMSE', verbose=0)
    final_model.fit(X, y)
    final_model.save_model(save_path)
    return final_model

# 3. Génération de profils aléatoires (inchangé)

def generate_random_profiles(n_profiles=400, basal_range=(0.3, 1.5), isf_range=(20, 80), csf_range=(5, 20)):
    """
    Génère n profils journaliers aléatoires (24h) pour basal, ISF, CSF.
    """
    profiles = []
    for _ in range(n_profiles):
        basal = np.round(np.random.uniform(*basal_range, 24) / 0.05) * 0.05
        isf = np.round(np.random.uniform(*isf_range, 24))
        csf = np.round(np.random.uniform(*csf_range, 24) / 10) * 10
        profiles.append({'basal': basal, 'isf': isf, 'csf': csf})
    return profiles

# 4. Application d’un profil

def apply_profile_to_data(df, profile):
    """
    Applique un profil (basal, isf, csf) heure par heure sur une copie du DataFrame.
    """
    df_mod = df.copy()
    for h in range(24):
        mask = df_mod['hour'] == h
        df_mod.loc[mask, 'basal_profile'] = profile['basal'][h]
        df_mod.loc[mask, 'isf'] = profile['isf'][h]
        df_mod.loc[mask, 'csf'] = profile['csf'][h]
    return df_mod

# 5. Scoring multi-horizon (sur t+30 uniquement pour stabilité, mais on peut étendre)

def custom_score_multi(y_pred, y_true, cob_val, tail_exposure, target_range=(80, 180)):
    """
    Calcule un score composite (hypo, in range, déviation) sur la prédiction t+30.
    """
    y_pred_30 = y_pred[:, 0]
    hypo = np.mean(y_pred_30 < 80)
    in_range = np.mean((y_pred_30 >= target_range[0]) & (y_pred_30 <= target_range[1]))
    stable_zone = (cob_val < 1e-2) & (tail_exposure < 0.1)
    if np.sum(stable_zone) > 10:
        y_pred_stable = y_pred_30[stable_zone]
        deviation = np.std(y_pred_stable)
    else:
        deviation = 0.0
    score = 5 * hypo - 2 * in_range + 3 * deviation
    return {
        'hypo': hypo,
        'in_range': in_range,
        'deviation': deviation,
        'score': score
    }

def evaluate_profile(model, df, profile, feature_cols):
    """
    Applique un profil, prédit la glycémie et calcule le score associé.
    """
    df_mod = apply_profile_to_data(df, profile)
    X = df_mod[feature_cols].fillna(0)
    y_true = df_mod[['sgv_t+30', 'sgv_t+60', 'sgv_t+90']].values
    y_pred = model.predict(X)
    cob_val = df_mod["cob_val"].values
    tail_exposure = df_mod["tail_exposure"].values
    score_dict = custom_score_multi(y_pred, y_true, cob_val, tail_exposure)
    score_dict["mae"] = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))
    score_dict["profile"] = profile
    return score_dict

# 6. Optimisation

def optimize_profiles(model, df, feature_cols, baseline_profile, n_profiles=50):
    """
    Optimise les profils mutants autour d'un profil de référence.
    """
    results = []
    for _ in range(n_profiles):
        profile = mutate_around_baseline(baseline_profile)
        profile = smooth_profile(profile)
        res = evaluate_profile(model, df, profile, feature_cols)
        results.append(res)
    results = sorted(results, key=lambda x: x['score'])
    return results

# 7. Affichage

def summarize_top_profiles(results, top_k=5):
    """
    Affiche un résumé des top profils optimaux.
    """
    rows = []
    for i, r in enumerate(results[:top_k]):
        profile = r['profile']
        row = {
            'Rank': i + 1,
            'Score': round(r['score'], 2),
            'MAE': round(r['mae'], 2),
            'In Range (%)': f"{r['in_range'] * 100:.1f}",
            'Hypo (%)': f"{r['hypo'] * 100:.1f}",
            'Déviation': round(r['deviation'], 2),
            'Basal (moy)': round(np.mean(profile['basal']), 2),
            'ISF (moy)': round(np.mean(profile['isf']), 1),
            'CSF (moy)': round(np.mean(profile['csf']), 1)
        }
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    print("\n📋 Résumé des profils optimaux :")
    print(df_summary.to_markdown(index=False))

def export_hourly_profiles(results, top_k=3, save_csv=True):
    """
    Exporte les profils top_k heure par heure en CSV.
    """
    for i, r in enumerate(results[:top_k]):
        p = r["profile"]
        df = pd.DataFrame({
            "Heure": [f"{h:02d}h" for h in range(24)],
            "Basal": p["basal"],
            "ISF": p["isf"],
            "CSF": p["csf"]
        })
        if save_csv:
            df.to_csv(f"profile_{i+1}_hourly.csv", index=False)


def smooth_profile(profile, sigma=3):
    for key in ['basal', 'isf', 'csf']:
        values = profile[key]
        extended = np.concatenate([values[-3:], values, values[:3]])  # padding circulaire
        smoothed = gaussian_filter1d(extended, sigma=sigma)[3:-3]
        profile[key] = smoothed
    return profile


def mutate_around_baseline(baseline_profile, amp_basal=0.5, amp_isf=30, amp_csf=5):
    """
    Génère un profil mutant réaliste autour d’un profil de référence.
    """
    def mutate_array(arr, amp, mode='relative'):
        if mode == 'relative':
            noise = np.random.normal(0, amp, len(arr))
            return np.clip(arr * (1 + noise), 0.3, None)
        elif mode == 'absolute':
            noise = np.random.normal(0, amp, len(arr))
            return np.clip(arr + noise, 1, None)

    return {
        'basal': mutate_array(baseline_profile['basal'], amp_basal, mode='relative'),
        'isf': mutate_array(baseline_profile['isf'], amp_isf, mode='absolute'),
        'csf': mutate_array(baseline_profile['csf'], amp_csf, mode='absolute')
    }

def log_model_quality(y_true, y_pred):
    horizons = ['t+30', 't+60', 't+90']
    maes = np.mean(np.abs(y_pred - y_true), axis=0)
    for i, h in enumerate(horizons):
        print(f"📈 MAE {h}: {maes[i]:.2f}")
    print(f"🎯 MAE moyenne globale : {np.mean(maes):.2f}")


def load_profile_from_ini(path):
    """
    Charge un profil basal / ISF / CSF depuis un fichier .ini AAPS-like.
    Retourne un dict avec 3 arrays de taille 24.
    """
    import configparser
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profil .ini introuvable : {path}")
    config = configparser.ConfigParser()
    config.read(path)

    def parse_section(section):
        return np.array([float(config[section][f"{h:02d}"]) for h in range(24)])

    profile = {
        "basal": parse_section("basal"),
        "isf": parse_section("isf"),
        "csf": parse_section("csf")
    }
    return profile


def plot_mutant_vs_baseline(results, baseline_profile, top_k=3):
    """
    Affiche et sauvegarde la comparaison entre profils mutants et baseline.
    """
    heures = np.arange(24)
    for i, r in enumerate(results[:top_k]):
        mutant = r["profile"]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(heures, baseline_profile["basal"], label="Basal actuel", color="gray", linestyle='--')
        plt.plot(heures, mutant["basal"], label=f"Basal mutant #{i+1}", color="tab:blue")
        plt.title("Basal (U/h)")
        plt.xlabel("Heure")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(heures, baseline_profile["isf"], label="ISF actuel", color="gray", linestyle='--')
        plt.plot(heures, mutant["isf"], label=f"ISF mutant #{i+1}", color="tab:green")
        plt.title("ISF (mg/dL/U)")
        plt.xlabel("Heure")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(heures, baseline_profile["csf"], label="CSF actuel", color="gray", linestyle='--')
        plt.plot(heures, mutant["csf"], label=f"CSF mutant #{i+1}", color="tab:orange")
        plt.title("CSF (g/U)")
        plt.xlabel("Heure")
        plt.grid(True)

        plt.suptitle(f"🧬 Comparaison Profil Mutant #{i+1} — Score {r['score']:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"profile_mutant_{i+1}_vs_baseline.png")
        plt.close()

# 8. Entrée point

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    outputs_dir = os.path.join(os.path.dirname(__file__), '../outputs')
    plots_dir = os.path.join(os.path.dirname(__file__), '../plots')

    X, y, df, feature_cols = prepare_data(os.path.join(data_dir, 'features_debug.csv'))
    baseline_profile = load_profile_from_ini(os.path.join(data_dir, "profil_base.ini"))
    model = train_catboost_multioutput(X, y, save_path=os.path.join(models_dir, 'model_catboost_multi.cbm'))
    results = optimize_profiles(model, df, feature_cols, baseline_profile, n_profiles=100)
    summarize_top_profiles(results, top_k=3)
    export_hourly_profiles(results, top_k=3, save_csv=True)
    plot_mutant_vs_baseline(results, baseline_profile, top_k=3)

