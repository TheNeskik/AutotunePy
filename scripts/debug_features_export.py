import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import numpy as np
from nightscout_fetcher import fetch_nightscout_data, get_profile_for_timestamp
from autotune_rnn import get_basal_for_entry, get_isf_for_entry, get_csf_for_entry, get_devicestatus_features

def generate_features_csv(days=2, output_path=None, status_callback=None):
    import os
    import sys
    try:
        from dotenv import load_dotenv
    except ImportError:
        print('python-dotenv non trouvé, tentative d\'installation...')
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-dotenv'])
        from dotenv import load_dotenv
    from datetime import datetime, timezone
    if status_callback:
        status_callback("Récupération des données Nightscout...")
    load_dotenv()
    url = os.getenv("NIGHTSCOUT_URL", "")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), '../data/features_debug.csv')
    token = os.getenv("NIGHTSCOUT_TOKEN", "")
    data = fetch_nightscout_data(url, token, days)
    entries = sorted(data.get("entries", []), key=lambda e: e.get("date", 0))
    devicestatus = sorted(data.get("devicestatus", []), key=lambda d: d.get("date", 0))
    treatments = data.get("treatments", [])
    profiles = data.get("profile", [])
    print(f"Entries: {len(entries)}, Treatments: {len(treatments)}, Profiles: {len(profiles)}, Devicestatus: {len(devicestatus)}")
    if status_callback:
        status_callback("Agglomération et calcul des features avancées...")
    entries_df = pd.DataFrame(entries)
    devicestatus_df = pd.DataFrame(devicestatus)
    entries_df['timestamp'] = pd.to_datetime(entries_df['date'], unit='ms', utc=True)
    devicestatus_df['timestamp'] = pd.to_datetime(devicestatus_df['date'], unit='ms', utc=True)
    def extract_iob_cob_target(row):
        iob_val = None
        if 'iob' in row and row['iob'] is not None:
            iob = row['iob']
            if isinstance(iob, dict):
                iob_val = iob.get('iob', None)
            elif isinstance(iob, list) and len(iob) > 0 and isinstance(iob[0], dict):
                iob_val = iob[0].get('iob', None)
        elif 'openaps' in row and row['openaps'] is not None:
            openaps = row['openaps']
            if isinstance(openaps, dict) and 'iob' in openaps and isinstance(openaps['iob'], dict):
                iob_val = openaps['iob'].get('iob', None)
        cob_val = None
        target_val = None
        if 'openaps' in row and row['openaps'] is not None:
            openaps = row['openaps']
            if isinstance(openaps, dict) and 'suggested' in openaps and isinstance(openaps['suggested'], dict):
                cob_val = openaps['suggested'].get('COB', None)
                target_val = openaps['suggested'].get('targetBG', None)
        return iob_val, cob_val, target_val
    devicestatus_df['iob_val'], devicestatus_df['cob_val'], devicestatus_df['target_val'] = zip(*devicestatus_df.apply(extract_iob_cob_target, axis=1))
    entries_df = entries_df.sort_values('timestamp')
    devicestatus_df = devicestatus_df.sort_values('timestamp')
    merged = pd.merge_asof(
        entries_df,
        devicestatus_df[['timestamp', 'iob_val', 'cob_val', 'target_val']],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('2min'),
        allow_exact_matches=True
    )
    def find_nearest_devicestatus_ts(entry_ts):
        idx = devicestatus_df['timestamp'].searchsorted(entry_ts, side='left')
        if idx == 0:
            nearest = devicestatus_df.iloc[0]['timestamp']
        elif idx == len(devicestatus_df):
            nearest = devicestatus_df.iloc[-1]['timestamp']
        else:
            before = devicestatus_df.iloc[idx-1]['timestamp']
            after = devicestatus_df.iloc[idx]['timestamp']
            nearest = before if abs((entry_ts - before).total_seconds()) <= abs((after - entry_ts).total_seconds()) else after
        return nearest
    merged['devicestatus_ts'] = merged['timestamp'].apply(find_nearest_devicestatus_ts)
    merged['delta_entry_devicestatus_s'] = (merged['timestamp'] - merged['devicestatus_ts']).abs().dt.total_seconds()
    merged = merged[(merged['delta_entry_devicestatus_s'] <= 300) & (~merged['iob_val'].isna()) & (~merged['cob_val'].isna())]
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    merged['Delta'] = merged['sgv'].diff()
    merged['ShortAvgDelta'] = merged['sgv'].rolling(window=3, min_periods=1).mean().diff()
    merged['LongAvgDelta'] = merged['sgv'].rolling(window=9, min_periods=1).mean().diff()
    merged['Delta'] = merged['Delta'].round(2)
    merged['ShortAvgDelta'] = merged['ShortAvgDelta'].round(2)
    merged['LongAvgDelta'] = merged['LongAvgDelta'].round(2)
    merged['datetime_local'] = merged['timestamp'].dt.tz_convert('Europe/Paris')
    merged['hour'] = merged['datetime_local'].dt.hour
    merged['minute'] = merged['datetime_local'].dt.minute
    merged['weekday'] = merged['datetime_local'].dt.weekday
    merged['minutes_since_midnight'] = merged['hour'] * 60 + merged['minute']
    merged['hour_sin'] = np.sin(2 * np.pi * merged['minutes_since_midnight'] / 1440)
    merged['hour_cos'] = np.cos(2 * np.pi * merged['minutes_since_midnight'] / 1440)
    merged['weekday_sin'] = np.sin(2 * np.pi * merged['weekday'] / 7)
    merged['weekday_cos'] = np.cos(2 * np.pi * merged['weekday'] / 7)
    for feat, col in [('glucose', 'sgv'), ('iob', 'iob_val'), ('cob', 'cob_val')]:
        for mins in [5, 15, 30, 60]:
            merged[f'{col}_{mins}min_ago'] = merged[col].shift(int(mins/5))
    for window in [6, 12]:
        merged[f'glucose_mean_{window*5}min'] = merged['sgv'].rolling(window=window, min_periods=1).mean()
        merged[f'glucose_std_{window*5}min'] = merged['sgv'].rolling(window=window, min_periods=1).std()
        merged[f'glucose_min_{window*5}min'] = merged['sgv'].rolling(window=window, min_periods=1).min()
        merged[f'glucose_max_{window*5}min'] = merged['sgv'].rolling(window=window, min_periods=1).max()
    merged['glucose_slope'] = (merged['sgv'] - merged['sgv'].shift(1)) / 5
    merged['glucose_slope_5min_ago'] = merged['glucose_slope'].shift(1)
    merged['glucose_accel'] = (merged['glucose_slope'] - merged['glucose_slope_5min_ago']) / 5
    merged['glucose_rolling_min_30min'] = merged['sgv'].rolling(window=6, min_periods=1).min()
    merged['glucose_rolling_max_30min'] = merged['sgv'].rolling(window=6, min_periods=1).max()
    merged['glucose_rolling_min_60min'] = merged['sgv'].rolling(window=12, min_periods=1).min()
    merged['glucose_rolling_max_60min'] = merged['sgv'].rolling(window=12, min_periods=1).max()
    # Mapping meal_bolus/meal_carbs vectorisé
    from bisect import bisect_left
    entry_times = merged['timestamp'].tolist()
    meal_bolus_events = []
    meal_carbs_events = []
    for t in treatments:
        if t.get('eventType', '').lower() == 'meal bolus':
            if 'created_at' in t:
                try:
                    meal_time = datetime.fromisoformat(t['created_at'].replace('Z', '+00:00'))
                except Exception:
                    continue
            elif 'timestamp' in t:
                meal_time = pd.to_datetime(t['timestamp'], utc=True)
            elif 'date' in t:
                meal_time = datetime.fromtimestamp(t['date']/1000, tz=timezone.utc)
            else:
                continue
            if t.get('insulin', 0):
                meal_bolus_events.append({'time': meal_time, 'amount': t.get('insulin', 0)})
            if t.get('carbs', 0):
                meal_carbs_events.append({'time': meal_time, 'amount': t.get('carbs', 0)})
    meal_bolus_events = sorted(meal_bolus_events, key=lambda s: s['time'])
    meal_carbs_events = sorted(meal_carbs_events, key=lambda s: s['time'])
    meal_bolus_idx = np.zeros(len(merged))
    meal_carbs_idx = np.zeros(len(merged))
    for mb in meal_bolus_events:
        times = entry_times
        pos = bisect_left(times, mb['time'])
        candidates = []
        for i in [pos-1, pos]:
            if 0 <= i < len(times):
                delta = abs((times[i] - mb['time']).total_seconds())
                if delta <= 300:
                    candidates.append((delta, i))
        if candidates:
            _, best_idx = min(candidates)
            meal_bolus_idx[best_idx] = mb['amount']
    for mc in meal_carbs_events:
        times = entry_times
        pos = bisect_left(times, mc['time'])
        candidates = []
        for i in [pos-1, pos]:
            if 0 <= i < len(times):
                delta = abs((times[i] - mc['time']).total_seconds())
                if delta <= 300:
                    candidates.append((delta, i))
        if candidates:
            _, best_idx = min(candidates)
            meal_carbs_idx[best_idx] = mc['amount']
    merged['meal_bolus'] = meal_bolus_idx
    merged['meal_carbs'] = meal_carbs_idx
    # Vectorisation des masques historiques
    for feat, col in [('glucose', 'sgv'), ('iob', 'iob_val'), ('cob', 'cob_val')]:
        for mins in [5, 15, 30, 60]:
            mask_col = f'{feat}_{mins}min_ago_mask'
            value_col = f'{col}_{mins}min_ago'
            merged[mask_col] = (~merged[value_col].isna()).astype(int)
    # Calcul des features classiques nécessitant un accès externe (profil, basal, isf, csf)
    # On ne garde la boucle que pour ces features
    if status_callback:
        status_callback("Calcul des features classiques (profil, basal, isf, csf)...")
    try:
        import swifter
    except ImportError:
        print('swifter non trouvé, tentative d\'installation...')
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'swifter'])
        import swifter
    from tqdm import tqdm
    tqdm.pandas(desc="Features classiques")
    def compute_profile_features(row):
        ts = row['timestamp'].to_pydatetime()
        prof = get_profile_for_timestamp(profiles, ts)
        basal_profile = get_basal_for_entry({'timestamp': ts, 'profile': prof}) if prof else None
        isf = get_isf_for_entry({'timestamp': ts, 'profile': prof}) if prof else None
        csf = get_csf_for_entry({'timestamp': ts, 'profile': prof}) if prof else None
        return pd.Series({'basal_profile': basal_profile, 'isf': isf, 'csf': csf})
    merged[['basal_profile', 'isf', 'csf']] = merged.swifter.apply(compute_profile_features, axis=1)
    if status_callback:
        status_callback("Génération du CSV features_debug.csv...")
    keep_cols = [
        'date','timestamp','datetime_local','sgv','basal_profile','isf','csf','iob_val','cob_val','target_val',
        'delta_entry_devicestatus_s','Delta','ShortAvgDelta','LongAvgDelta','basal_profile','isf','csf',
        'hour','minute','weekday','minutes_since_midnight','hour_sin','hour_cos','weekday_sin','weekday_cos',
        'glucose_mean_30min','glucose_std_30min','glucose_min_30min','glucose_max_30min',
        'glucose_mean_60min','glucose_std_60min','glucose_min_60min','glucose_max_60min',
        'glucose_slope','glucose_slope_5min_ago','glucose_accel',
        'glucose_rolling_min_30min','glucose_rolling_max_30min','glucose_rolling_min_60min','glucose_rolling_max_60min',
        'meal_bolus','meal_carbs',
        'glucose_5min_ago','glucose_15min_ago','glucose_30min_ago','glucose_60min_ago',
        'iob_val_5min_ago','iob_val_15min_ago','iob_val_30min_ago','iob_val_60min_ago',
        'cob_val_5min_ago','cob_val_15min_ago','cob_val_30min_ago','cob_val_60min_ago',
        'glucose_5min_ago_mask','glucose_15min_ago_mask','glucose_30min_ago_mask','glucose_60min_ago_mask',
        'iob_5min_ago_mask','iob_15min_ago_mask','iob_30min_ago_mask','iob_60min_ago_mask',
        'cob_5min_ago_mask','cob_15min_ago_mask','cob_30min_ago_mask','cob_60min_ago_mask'
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged[keep_cols].to_csv(output_path, index=False)
    if status_callback:
        status_callback(f"CSV généré avec succès ({output_path})")
    print(f'{output_path} généré avec succès.')

def main(days=2):
    generate_features_csv(days=days, output_path='features_debug.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=2, help='Nombre de jours à exporter')
    args = parser.parse_args()
    main(days=args.days)
