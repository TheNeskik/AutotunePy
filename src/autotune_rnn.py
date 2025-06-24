# DEPRECATED: Ce module n'est plus utilisé dans le pipeline principal. Gardé pour référence et recherche uniquement.
# DEPRECATED: This module is no longer used in the main pipeline. Kept for reference and research purposes only.
#
# Utilisation typique :
# X, y_basal, y_isf, y_csf = load_features_from_csv('features_debug.csv', days=nb_jours)
# model = train_rnn_model(X, y_basal, y_isf, y_csf)
# y_pred = predict_with_rnn(model, X)
#
# Typical usage:
# X, y_basal, y_isf, y_csf = load_features_from_csv('features_debug.csv', days=nb_days)
# model = train_rnn_model(X, y_basal, y_isf, y_csf)
# y_pred = predict_with_rnn(model, X)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Préparation des données pour le RNN

def get_devicestatus_features(devicestatus, ts):
    """
    Pour un timestamp donné, retourne les features IOB, basaliob, activity, COB les plus proches dans devicestatus.
    Corrige le bug offset-naive/offset-aware en forçant UTC.
    """
    import datetime
    from datetime import timezone
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    min_dt = None
    best = None
    for d in devicestatus:
        dts = None
        # Cherche le timestamp dans openaps.iob.time ou created_at
        if 'openaps' in d and 'iob' in d['openaps'] and 'time' in d['openaps']['iob']:
            try:
                dts = datetime.datetime.fromisoformat(d['openaps']['iob']['time'].replace('Z', '+00:00'))
            except Exception:
                continue
        elif 'created_at' in d:
            try:
                dts = datetime.datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
            except Exception:
                continue
        if not dts:
            continue
        if dts.tzinfo is None:
            dts = dts.replace(tzinfo=timezone.utc)
        dt = abs((ts - dts).total_seconds())
        if min_dt is None or dt < min_dt:
            min_dt = dt
            best = d
    # Extrait les features
    iob = best.get('openaps', {}).get('iob', {}).get('iob') if best else None
    basaliob = best.get('openaps', {}).get('iob', {}).get('basaliob') if best else None
    activity = best.get('openaps', {}).get('iob', {}).get('activity') if best else None
    cob = best.get('openaps', {}).get('cob') if best and 'openaps' in best and 'cob' in best['openaps'] else best.get('COB') if best else None
    # fallback sur racine
    if iob is None and best and 'IOB' in best:
        iob = best['IOB']
    if cob is None and best and 'COB' in best:
        cob = best['COB']
    return iob or 0, basaliob or 0, activity or 0, cob or 0


def prepare_rnn_sequences(entries, treatments, profiles, window_hours=24, target_glucose=100, devicestatus=None):
    """
    Prépare les séquences d'entrée pour le RNN à partir des données Nightscout brutes.
    Ajoute toutes les features précédentes + IOB, basaliob, activity, COB depuis devicestatus.
    """
    from nightscout_fetcher import get_profile_for_timestamp
    import datetime
    import numpy as np
    # Liste des noms de profils pour one-hot
    profile_names = list({p['defaultProfile'] for p in profiles if 'defaultProfile' in p})
    # Construction d'un DataFrame horaire à partir des données brutes
    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if 'date' not in e or 'sgv' not in e:
            continue
        ts = datetime.datetime.utcfromtimestamp(e['date'] / 1000)
        prof = get_profile_for_timestamp(profiles, ts)
        if not prof:
            continue
        # Jour de la semaine
        weekday = ts.weekday()
        # Profil one-hot
        prof_name = prof['defaultProfile']
        prof_onehot = [1 if prof_name == n else 0 for n in profile_names]
        # Historique des cibles (basal, isf, csf à cette heure)
        store = prof['store'][prof['defaultProfile']]
        def get_val(schedule):
            h = ts.hour
            for i in range(len(schedule)-1, -1, -1):
                seg = schedule[i]
                start = seg.get('timeAsSeconds', 0)//3600 if 'timeAsSeconds' in seg else int(seg['time'].split(':')[0])
                if h >= start:
                    return float(seg['value'])
            return float(schedule[0]['value']) if schedule else np.nan
        basal = get_val(store.get('basal', []))
        isf = get_val(store.get('sens', []))
        csf = get_val(store.get('carbratio', []))
        # Traitements : bolus, carbs, délai depuis dernier bolus/repas
        last_bolus = 0
        last_carb = 0
        min_bolus_delay = 24*60  # minutes
        min_carb_delay = 24*60
        for t in treatments:
            tts = None
            if 'date' in t and isinstance(t['date'], (int, float)):
                tts = datetime.datetime.utcfromtimestamp(t['date'] / 1000)
            elif 'created_at' in t:
                try:
                    tts = datetime.datetime.fromisoformat(t['created_at'].replace('Z', '+00:00'))
                except Exception:
                    continue
            if not tts: continue
            dt_min = (ts - tts).total_seconds() / 60
            if 0 <= dt_min < min_bolus_delay and t.get('insulin', 0):
                min_bolus_delay = dt_min
                last_bolus = t.get('insulin', 0) or 0
            if 0 <= dt_min < min_carb_delay and t.get('carbs', 0):
                min_carb_delay = dt_min
                last_carb = t.get('carbs', 0) or 0
        hour = ts.hour
        # Ajout des features devicestatus
        iob, basaliob, activity, cob = get_devicestatus_features(devicestatus or [], ts)
        rows.append({
            'timestamp': ts,
            'glucose': e['sgv'],
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'weekday_sin': np.sin(2 * np.pi * weekday / 7),
            'weekday_cos': np.cos(2 * np.pi * weekday / 7),
            'profile_onehot': prof_onehot,
            'basal_prev': basal,
            'isf_prev': isf,
            'csf_prev': csf,
            'last_bolus': last_bolus,
            'last_carb': last_carb,
            'bolus_delay': min_bolus_delay,
            'carb_delay': min_carb_delay,
            'target_glucose': target_glucose,
            'iob': iob,
            'basaliob': basaliob,
            'activity': activity,
            'cob': cob,
            'profile': prof
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Aucune entrée exploitable pour le RNN (profil ou timestamp manquant)")
    # Colonnes numériques et one-hot
    numeric_cols = ['glucose', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
                   'basal_prev', 'isf_prev', 'csf_prev', 'last_bolus', 'last_carb', 'bolus_delay', 'carb_delay',
                   'target_glucose', 'iob', 'basaliob', 'activity', 'cob']
    onehot_cols = [f'profile_{n}' for n in profile_names]
    # Déplier le one-hot
    for i, n in enumerate(profile_names):
        df[f'profile_{n}'] = df['profile_onehot'].apply(lambda x: x[i])
    df_num = df[['timestamp'] + numeric_cols + onehot_cols] if 'timestamp' in df else df[numeric_cols + onehot_cols]
    df_num = df_num.set_index('timestamp')
    # Pas de resample, on garde la granularité native des entries
    df_profiles = df[['timestamp', 'profile']].set_index('timestamp')
    df = df_num.join(df_profiles, how='left')
    # Calcul des dérivées du glucose sur la granularité native
    df['glucose_delta'] = df['glucose'].diff().fillna(0)
    df['glucose_mean'] = df['glucose'].rolling(window=window_hours, min_periods=1).mean()
    df['glucose_min'] = df['glucose'].rolling(window=window_hours, min_periods=1).min()
    df['glucose_max'] = df['glucose'].rolling(window=window_hours, min_periods=1).max()
    df['glucose_std'] = df['glucose'].rolling(window=window_hours, min_periods=1).std().fillna(0)
    def get_hourly_targets(ts, prof):
        store = prof['store'][prof['defaultProfile']]
        def get_val(schedule):
            h = ts.hour
            for i in range(len(schedule)-1, -1, -1):
                seg = schedule[i]
                start = seg.get('timeAsSeconds', 0)//3600 if 'timeAsSeconds' in seg else int(seg['time'].split(':')[0])
                if h >= start:
                    return float(seg['value'])
            return float(schedule[0]['value']) if schedule else np.nan
        basal = get_val(store.get('basal', []))
        isf = get_val(store.get('sens', []))
        csf = get_val(store.get('carbratio', []))
        return basal, isf, csf
    features = numeric_cols + onehot_cols + ['glucose_delta', 'glucose_mean', 'glucose_min', 'glucose_max', 'glucose_std']
    X, y_basal, y_isf, y_csf = [], [], [], []
    for i in range(len(df) - window_hours):
        seq = df.iloc[i:i+window_hours][features].values
        X.append(seq)
        ts_target = df.index[i+window_hours]
        prof_target = df.iloc[i+window_hours].get('profile')
        # Correction : si prof_target n'est pas un dict, on le récupère via get_profile_for_timestamp
        if not isinstance(prof_target, dict):
            prof_target = get_profile_for_timestamp(profiles, ts_target)
        basal, isf, csf = get_hourly_targets(ts_target, prof_target)
        y_basal.append(basal)
        y_isf.append(isf)
        y_csf.append(csf)
    X = np.array(X)
    y_basal = np.array(y_basal)
    y_isf = np.array(y_isf)
    y_csf = np.array(y_csf)
    return X, y_basal, y_isf, y_csf

# 2. Modèle RNN multi-sorties (basal, isf, csf)
def build_rnn_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inp)
    out_basal = Dense(1, name='basal')(x)
    out_isf = Dense(1, name='isf')(x)
    out_csf = Dense(1, name='csf')(x)
    model = Model(inputs=inp, outputs=[out_basal, out_isf, out_csf])
    model.compile(optimizer='adam', loss='mse')
    return model

# 3. Entraînement du modèle
def train_rnn_model(X, y_basal, y_isf, y_csf, epochs=500, batch_size=64, early_stopping_patience=5, verbose=1):
    model = build_rnn_model(X.shape[1:])
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True, verbose=verbose)
    history = model.fit(
        X, [y_basal, y_isf, y_csf],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[es],
        verbose=verbose
    )
    if es.stopped_epoch > 0:
        print(f"[INFO] Early stopping: entraînement arrêté à l'epoch {es.stopped_epoch+1} (convergence détectée)")
    return model

# 4. Prédiction
def predict_with_rnn(model, X):
    y_basal, y_isf, y_csf = model.predict(X)
    return y_basal, y_isf, y_csf

# 5. Sauvegarde/chargement du modèle
def save_rnn_model(model, path):
    model.save(path)

def load_rnn_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path)

def format_rnn_output(y_basal_pred, y_isf_pred, y_csf_pred):
    """
    Formate l'output pour n'afficher que le dernier bloc de 24h prédit.
    """
    n = 24
    if len(y_basal_pred) < n:
        n = len(y_basal_pred)
    start = len(y_basal_pred) - n
    result = ["✅ Profil RNN prédit (24h finale) :\n"]
    for i in range(start, start + n):
        h = i % 24
        result.append(f"Heure {h:02d}h  Basal: {y_basal_pred[i][0]:.3f}  ISF: {y_isf_pred[i][0]:.2f}  CSF: {y_csf_pred[i][0]:.2f}")
    return "\n".join(result)

def get_basal_for_entry(entry):
    prof = entry.get('profile')
    if not prof: return None
    store = prof['store'][prof['defaultProfile']]
    basal_list = store.get('basal', [])
    minutes = entry['timestamp'].hour * 60 + entry['timestamp'].minute
    for i in range(len(basal_list)-1, -1, -1):
        seg = basal_list[i]
        seg_min = seg.get('timeAsSeconds', 0)//60 if 'timeAsSeconds' in seg else int(seg['time'].split(':')[0])*60+int(seg['time'].split(':')[1])
        if minutes >= seg_min:
            return float(seg['value'])
    return float(basal_list[0]['value']) if basal_list else None

def get_isf_for_entry(entry):
    prof = entry.get('profile')
    if not prof: return None
    store = prof['store'][prof['defaultProfile']]
    sens_list = store.get('sens', [])
    minutes = entry['timestamp'].hour * 60 + entry['timestamp'].minute
    for i in range(len(sens_list)-1, -1, -1):
        seg = sens_list[i]
        seg_min = seg.get('timeAsSeconds', 0)//60 if 'timeAsSeconds' in seg else int(seg['time'].split(':')[0])*60+int(seg['time'].split(':')[1])
        if minutes >= seg_min:
            return float(seg['value'])
    return float(sens_list[0]['value']) if sens_list else None

def get_csf_for_entry(entry):
    prof = entry.get('profile')
    if not prof: return None
    store = prof['store'][prof['defaultProfile']]
    csf_list = store.get('carbratio', [])
    minutes = entry['timestamp'].hour * 60 + entry['timestamp'].minute
    for i in range(len(csf_list)-1, -1, -1):
        seg = csf_list[i]
        seg_min = seg.get('timeAsSeconds', 0)//60 if 'timeAsSeconds' in seg else int(seg['time'].split(':')[0])*60+int(seg['time'].split(':')[1])
        if minutes >= seg_min:
            return float(seg['value'])
    return float(csf_list[0]['value']) if csf_list else None

# Utilisation :
# X, y_basal, y_isf, y_csf = prepare_rnn_sequences(entries, treatments, profiles)
# model = train_rnn_model(X, y_basal, y_isf, y_csf)
# y_basal_pred, y_isf_pred, y_csf_pred = predict_with_rnn(model, X)
# print(format_rnn_output(y_basal_pred, y_isf_pred, y_csf_pred))
