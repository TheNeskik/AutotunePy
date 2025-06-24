# DEPRECATED: Ce module n'est plus utilisé dans le pipeline principal. Gardé pour référence et recherche uniquement.
# DEPRECATED: This module is no longer used in the main pipeline. Kept for reference and research purposes only.
# ATTENTION : Les résultats de ce script sont différents de ceux du vrai Autotune (OpenAPS/oref0).
# WARNING: The results of this script differ from the real Autotune (OpenAPS/oref0).
#
# Utilisation typique :
# from autotune import run_autotune
# result = run_autotune(data)
#
# Typical usage:
# from autotune import run_autotune
# result = run_autotune(data)

from datetime import datetime
from datetime import timedelta
from collections import defaultdict
from statistics import mean
from nightscout_fetcher import get_profile_for_timestamp

def prepare_data(profiles, treatments, entries):
    print("[DEBUG] Entrée dans prepare_data (multi-profils)")
    # 1. Nettoyage des entries (glycémies)
    clean_entries = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if "sgv" not in e or not isinstance(e["sgv"], (int, float)):
            continue
        ts = None
        if "date" in e and isinstance(e["date"], (int, float)):
            ts = datetime.utcfromtimestamp(e["date"] / 1000)
        elif "created_at" in e:
            try:
                ts = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
            except Exception:
                continue
        if ts:
            prof = get_profile_for_timestamp(profiles, ts)
            clean_entries.append({
                "glucose": e["sgv"],
                "timestamp": ts,
                "profile": prof
            })

    # 2. Nettoyage des traitements (bolus/glucides)
    filtered_treatments = []
    for t in treatments:
        if not isinstance(t, dict):
            continue
        has_carbs = "carbs" in t and isinstance(t["carbs"], (int, float)) and t["carbs"] is not None
        has_insulin = "insulin" in t and isinstance(t["insulin"], (int, float)) and t["insulin"] is not None
        if not (has_carbs or has_insulin):
            continue
        insulin = float(t.get("insulin", 0)) if has_insulin else 0
        carbs = float(t.get("carbs", 0)) if has_carbs else 0
        ts = None
        if "date" in t and isinstance(t["date"], (int, float)):
            ts = datetime.utcfromtimestamp(t["date"] / 1000)
        elif "created_at" in t:
            try:
                ts = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
            except Exception:
                continue
        if ts:
            prof = get_profile_for_timestamp(profiles, ts)
            filtered_treatments.append({
                "insulin": insulin,
                "carbs": carbs,
                "timestamp": ts,
                "profile": prof
            })

    print(f"[DEBUG] Entries retenues: {len(clean_entries)}")
    print(f"[DEBUG] Treatments retenus: {len(filtered_treatments)}")

    return {
        "entries": clean_entries,
        "treatments": filtered_treatments
    }

def calculate_bgi(entries, treatments):
    print("[DEBUG] Entrée dans calculate_bgi (multi-profils)")
    print(f"[DEBUG] Nombre d'entries: {len(entries)}")
    print(f"[DEBUG] Nombre de traitements: {len(treatments)}")
    """
    Calcule le BGI (Blood Glucose Impact) attendu à partir :
    - des basals
    - des bolus
    - des glucides
    selon le profil utilisateur ACTIF à chaque timestamp.
    """
    # Préparer les effets bolus et glucides
    bolus_effects = []
    carb_effects = []

    for t in treatments:
        ts = t["timestamp"]
        prof = t["profile"]
        # Extraction des paramètres du profil au moment du traitement
        if not prof or not prof.get("store") or not prof.get("defaultProfile"):
            print(f"[WARN] Profil manquant ou incomplet pour traitement à {ts}")
            continue
        store = prof["store"][prof["defaultProfile"]]
        isf = float(store.get("sens", [{}])[0].get("value", 50))
        ic_ratio = float(store.get("carbratio", [{}])[0].get("value", 10))
        if t["insulin"] > 0:
            for i in range(0, 240, 5):
                effect_time = ts + timedelta(minutes=i)
                fraction = insulin_action_curve(i)
                bolus_effects.append((effect_time, t["insulin"] * isf * fraction))
        if t["carbs"] > 0:
            for i in range(0, 180, 5):
                effect_time = ts + timedelta(minutes=i)
                fraction = carb_absorption_curve(i)
                carb_effects.append((effect_time, -t["carbs"] / ic_ratio * isf * fraction))

    # Calcul du BGI total
    bgi_points = []
    for e in entries:
        ts = e["timestamp"]
        prof = e["profile"]
        if not prof or not prof.get("store") or not prof.get("defaultProfile"):
            print(f"[WARN] Profil manquant ou incomplet pour entry à {ts}")
            continue
        store = prof["store"][prof["defaultProfile"]]
        # Basal schedule
        basal_schedule = [
            {"value": b["value"], "minutes": b["timeAsSeconds"]//60 if "timeAsSeconds" in b else int(b["time"].split(":")[0])*60+int(b["time"].split(":")[1])}
            for b in store.get("basal", [])
        ]
        # ISF et I:C Ratio
        isf = float(store.get("sens", [{}])[0].get("value", 50))
        ic_ratio = float(store.get("carbratio", [{}])[0].get("value", 10))
        def get_basal_rate(timestamp):
            if not basal_schedule:
                return 0
            minutes = timestamp.hour * 60 + timestamp.minute
            for i in range(len(basal_schedule)-1, -1, -1):
                start = basal_schedule[i]["minutes"]
                if minutes >= start:
                    return float(basal_schedule[i]["value"])
            return float(basal_schedule[0]["value"])
        basal = get_basal_rate(ts)
        if basal == 0:
            print(f"[WARN] Basal nul pour entry à {ts}")
        basal_bgi = (basal / 12) * isf  # 5 min
        bolus_bgi = sum(effect for time, effect in bolus_effects if abs((time - ts).total_seconds()) < 150)
        carb_bgi = sum(effect for time, effect in carb_effects if abs((time - ts).total_seconds()) < 150)
        total_bgi = round(basal_bgi + bolus_bgi + carb_bgi, 2)
        bgi_points.append({
            "timestamp": ts,
            "bgi_basal": round(basal_bgi, 2),
            "bgi_bolus": round(bolus_bgi, 2),
            "bgi_carbs": round(carb_bgi, 2),
            "bgi_total": total_bgi
        })
    print(f"[DEBUG] Nombre de bgi_points calculés: {len(bgi_points)}")
    return bgi_points

def insulin_action_curve(minutes):
    """Courbe simplifiée d'action de l'insuline rapide (ex: Novorapid)"""
    if minutes < 0 or minutes > 240:
        return 0
    if minutes < 30:
        return minutes / 30 * 0.2
    elif minutes < 90:
        return 0.2 + (minutes - 30) / 60 * 0.5
    elif minutes < 180:
        return 0.7 - (minutes - 90) / 90 * 0.6
    else:
        return 0.1 - (minutes - 180) / 60 * 0.1

def carb_absorption_curve(minutes):
    """Courbe simplifiée d'absorption des glucides"""
    if minutes < 0 or minutes > 180:
        return 0
    if minutes < 30:
        return minutes / 30 * 0.3
    elif minutes < 90:
        return 0.3 + (minutes - 30) / 60 * 0.5
    else:
        return 0.8 - (minutes - 90) / 90 * 0.7


def calculate_deviation(entries, bgi_points):
    print("[DEBUG] Entrée dans calculate_deviation")
    print(f"[DEBUG] Nombre d'entries: {len(entries)}")
    print(f"[DEBUG] Nombre de bgi_points: {len(bgi_points)}")
    """
    Compare la glycémie réelle avec le BGI total attendu pour chaque point :
    - Calcule la déviation (écart)
    - Retourne une liste de déviations horodatées
    """
    deviations = []

    # Indexer les BGI par timestamp pour correspondance rapide
    bgi_dict = {b["timestamp"]: b["bgi_total"] for b in bgi_points}

    for e in entries:
        ts = e["timestamp"]
        glucose = e["glucose"]
        bgi = bgi_dict.get(ts)

        if bgi is not None:
            # La déviation est la différence entre la glycémie réelle et l'impact attendu
            deviation = round(glucose - bgi, 2)
            deviations.append({
                "timestamp": ts,
                "glucose": glucose,
                "bgi": bgi,
                "deviation": deviation
            })

    return deviations

def adjust_parameters(entries, deviations):
    print("[DEBUG] Entrée dans adjust_parameters (multi-profils)")
    print(f"[DEBUG] Nombre de deviations: {len(deviations)}")
    """
    Propose des ajustements aux paramètres du profil :
    - Basal rates (par tranche horaire)
    - ISF (Insulin Sensitivity Factor)
    - Carb Ratio (I:C)
    en fonction des déviations observées.
    """
    if not deviations or not entries:
        return {"error": "Aucune déviation disponible pour ajustement."}

    # Grouper les déviations par tranche horaire et par profil
    from statistics import mean
    from collections import defaultdict
    hourly_devs = defaultdict(list)
    isf_vals = []
    ic_vals = []
    basal_segments = defaultdict(list)
    for e, d in zip(entries, deviations):
        prof = e.get("profile")
        if not prof or not prof.get("store") or not prof.get("defaultProfile"):
            continue
        store = prof["store"][prof["defaultProfile"]]
        # Basal schedule
        basal_list = store.get("basal", [])
        for i, seg in enumerate(basal_list):
            minutes = seg["timeAsSeconds"]//60 if "timeAsSeconds" in seg else int(seg["time"].split(":")[0])*60+int(seg["time"].split(":")[1])
            basal_segments[minutes].append(seg["value"])
            if e["timestamp"].hour*60+e["timestamp"].minute >= minutes:
                hourly_devs[minutes].append(d["deviation"])
        # ISF et I:C Ratio
        if store.get("sens"):
            isf_vals.append(float(store["sens"][0].get("value", 50)))
        if store.get("carbratio"):
            ic_vals.append(float(store["carbratio"][0].get("value", 10)))
    # Calculer la moyenne de déviation par segment
    new_basal = []
    for minutes, devs in hourly_devs.items():
        avg_dev = mean(devs) if devs else 0
        original = mean(basal_segments[minutes]) if basal_segments[minutes] else 1.0
        adj = original * (1 - avg_dev / 100)
        adj = max(original * 0.8, min(original * 1.2, adj))
        new_basal.append({
            "minutes": minutes,
            "current": round(original, 3),
            "recommended": round(adj, 3),
        })
    avg_deviation = mean([d["deviation"] for d in deviations])
    # ISF et I:C Ratio moyens
    current_isf = mean(isf_vals) if isf_vals else 50.0
    current_ic = mean(ic_vals) if ic_vals else 10.0
    isf_adjustment = current_isf * (1 - avg_deviation / 100)
    ic_adjustment = current_ic * (1 - avg_deviation / 100)
    def clamp(value, original):
        return max(original * 0.8, min(original * 1.2, value))
    adjusted_isf = round(clamp(isf_adjustment, current_isf), 2)
    adjusted_ic = round(clamp(ic_adjustment, current_ic), 2)
    return {
        "average_deviation": round(avg_deviation, 2),
        "recommendations": {
            "ISF": {
                "current": round(current_isf, 2),
                "recommended": adjusted_isf
            },
            "IC Ratio": {
                "current": round(current_ic, 2),
                "recommended": adjusted_ic
            },
            "BasalProfile": sorted(new_basal, key=lambda x: x["minutes"])
        }
    }



def run_autotune(data, gui_callback=None):
    if gui_callback:
        gui_callback("Analyse en cours... Veuillez patienter.")
    print("[INFO] Lancement de l'analyse Autotune...")
    try:
        prepared = prepare_data(data["profile"], data["treatments"], data["entries"])
        bgi_points = calculate_bgi(prepared["entries"], prepared["treatments"])
        deviations = calculate_deviation(prepared["entries"], bgi_points)
        results = adjust_parameters(prepared["entries"], deviations)
        if "error" in results:
            if gui_callback:
                gui_callback(f"Erreur : {results['error']}")
            return f"Erreur : {results['error']}"
        rec = results["recommendations"]
        output = [
            f"✅ Recommandations Autotune (déviation moyenne : {results['average_deviation']} mg/dL)",
            "",
            f"🔹 ISF (Insulin Sensitivity Factor)",
            f"    Actuel     : {rec['ISF']['current']}",
            f"    Recommandé : {rec['ISF']['recommended']}",
            "",
            f"🔹 I:C Ratio (Insulin-to-Carb)",
            f"    Actuel     : {rec['IC Ratio']['current']}",
            f"    Recommandé : {rec['IC Ratio']['recommended']}",
            "",
            f"🔹 Basal Profile (par tranche horaire) :"
        ]
        for seg in rec["BasalProfile"]:
            h = seg["minutes"] // 60
            m = seg["minutes"] % 60
            output.append(f"    {h:02d}:{m:02d}  Actuel: {seg['current']}  Reco: {seg['recommended']}")
        if gui_callback:
            gui_callback("Calcul terminé. Résultats prêts.")
        return "\n".join(output)
    except Exception as e:
        if gui_callback:
            gui_callback(f"Erreur lors de l'exécution d'Autotune : {str(e)}")
        return f"Erreur lors de l'exécution d'Autotune : {str(e)}"