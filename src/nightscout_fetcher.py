"""
Nightscout Fetcher - Utilitaires pour l'extraction asynchrone et synchrone des données Nightscout (profils, traitements, entrées, devicestatus).

- Supporte la pagination, la récupération par jour, la gestion du token, et la robustesse réseau.
- Peut être utilisé pour des analyses avancées, du machine learning, ou l'export massif de données.

Nightscout Fetcher - Utilities for asynchronous and synchronous extraction of Nightscout data (profiles, treatments, entries, devicestatus).

- Supports pagination, per-day retrieval, token management, and network robustness.
- Can be used for advanced analytics, machine learning, or bulk data export.
"""

import requests
from datetime import datetime, timedelta, timezone
import json
import asyncio
import aiohttp
import time

def fetch_all_nightscout(endpoint, base_url, params, token=None, page_size=1000):
    """
    Récupère tous les objets d'un endpoint Nightscout via pagination (API v1).
    Fetches all objects from a Nightscout endpoint using pagination (API v1).
    Args:
        endpoint (str): Nom de l'endpoint (entries, treatments, ...)
        base_url (str): URL de base Nightscout
        params (dict): Paramètres de requête (filtres, dates...)
        token (str, optional): Token d'API
        page_size (int): Nombre d'éléments par page
    Returns:
        list: Tous les objets récupérés
    """
    from dateutil.parser import parse as parse_date
    all_results = []
    skip = 0
    page = 1
    date_min = params.get("find[created_at][$gte]")
    date_max = params.get("find[created_at][$lte]")
    while True:
        paged_params = params.copy()
        paged_params["count"] = page_size
        paged_params["skip"] = skip
        if token:
            paged_params["token"] = token
        # Réduit le bruit : log uniquement la première et dernière page
        if page == 1:
            print(f"[FETCH] {endpoint} page 1...")
        resp = requests.get(f"{base_url}/api/v1/{endpoint}.json", params=paged_params)
        data = resp.json()
        if not data:
            break
        filtered = []
        for d in data:
            created = d.get("created_at")
            if created:
                try:
                    dt = parse_date(created)
                    if date_min and dt < parse_date(date_min):
                        continue
                    if date_max and dt > parse_date(date_max):
                        continue
                except Exception:
                    pass
            filtered.append(d)
        if not filtered:
            break
        all_results.extend(filtered)
        if len(data) < page_size:
            break
        skip += page_size
        page += 1
    print(f"[FETCH] {endpoint} terminé : {len(all_results)} éléments au total.")
    return all_results

async def fetch_endpoint_per_day(session, endpoint, base_url, token, current, count=5000):
    """
    Récupère les données d'un endpoint pour une journée donnée (asynchrone).
    Fetches endpoint data for a given day (async).
    """
    if endpoint == "entries":
        day_start_ts = int(current.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        day_end_ts = int((current.replace(hour=23, minute=59, second=59, microsecond=999999)).timestamp() * 1000)
        params = {
            "find[date][$gte]": day_start_ts,
            "find[date][$lt]": day_end_ts,
            "count": count
        }
    else:
        day_start = current.strftime("%Y-%m-%dT00:00:00")
        day_end = current.strftime("%Y-%m-%dT23:59:59")
        params = {
            "find[created_at][$gte]": day_start,
            "find[created_at][$lte]": day_end,
            "count": count
        }
    if token:
        params["token"] = token
    url = f"{base_url}/api/v1/{endpoint}.json"
    async with session.get(url, params=params, timeout=30) as resp:
        data = await resp.json()
        print(f"[FETCH][async] {endpoint} {current.strftime('%Y-%m-%d')} : {len(data)} éléments")
        return data

async def fetch_profiles_per_day(session, base_url, token, current, count=100):
    """
    Récupère les profils pour une journée donnée (asynchrone).
    Fetches profiles for a given day (async).
    """
    day_start = current.strftime("%Y-%m-%dT00:00:00")
    day_end = current.strftime("%Y-%m-%dT23:59:59")
    params = {
        "find[startDate][$gte]": day_start,
        "find[startDate][$lte]": day_end,
        "count": count
    }
    if token:
        params["token"] = token
    url = f"{base_url}/api/v1/profile.json"
    async with session.get(url, params=params, timeout=30) as resp:
        data = await resp.json()
        print(f"[FETCH][async] profiles {current.strftime('%Y-%m-%d')} : {len(data) if isinstance(data, list) else 1 if isinstance(data, dict) else 0} profils")
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []

async def fetch_endpoint_per_day_with_retry(session, endpoint, base_url, token, current, count=5000, max_retries=3, retry_delay=2):
    """
    Version robuste de fetch_endpoint_per_day avec retry automatique.
    Robust version of fetch_endpoint_per_day with automatic retry.
    """
    for attempt in range(max_retries):
        try:
            return await fetch_endpoint_per_day(session, endpoint, base_url, token, current, count)
        except Exception as e:
            print(f"[RETRY][{endpoint}] {current.strftime('%Y-%m-%d')} - tentative {attempt+1}/{max_retries} : {type(e).__name__} - {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return []

async def fetch_profiles_per_day_with_retry(session, base_url, token, current, count=100, max_retries=3, retry_delay=2):
    """
    Version robuste de fetch_profiles_per_day avec retry automatique.
    Robust version of fetch_profiles_per_day with automatic retry.
    """
    for attempt in range(max_retries):
        try:
            return await fetch_profiles_per_day(session, base_url, token, current, count)
        except Exception as e:
            print(f"[RETRY][profiles] {current.strftime('%Y-%m-%d')} - tentative {attempt+1}/{max_retries} : {type(e).__name__} - {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return []

async def fetch_nightscout_data_async(base_url, token=None, days=7, max_parallel_days=10):
    """
    Récupère toutes les données Nightscout (profils, traitements, entrées, devicestatus) sur plusieurs jours (asynchrone, multi-jours, multi-endpoints).
    Fetches all Nightscout data (profiles, treatments, entries, devicestatus) over multiple days (async, multi-day, multi-endpoint).
    Args:
        base_url (str): URL de base Nightscout
        token (str, optional): Token d'API
        days (int): Nombre de jours à récupérer
        max_parallel_days (int): Nombre max de jours en parallèle
    Returns:
        dict: {"profile": ..., "treatments": ..., "entries": ..., "devicestatus": ...}
    """
    from datetime import datetime, timedelta
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    print(f"[INFO][async] Fetching Nightscout data pour {days} jours : {start_date.date()} -> {end_date.date()}")
    all_profiles = []
    all_treatments = []
    all_entries = []
    all_devicestatus = []
    async with aiohttp.ClientSession() as session:
        current = start_date
        day_count = 0
        day_tasks = []
        while current <= end_date:
            day_tasks.append(current)
            current += timedelta(days=1)
            day_count += 1
        print(f"[DEBUG] Nombre total de jours à fetch : {day_count}")
        # Découpe en batchs pour limiter le nombre de requêtes parallèles
        for i in range(0, len(day_tasks), max_parallel_days):
            batch = day_tasks[i:i+max_parallel_days]
            tasks = []
            for day in batch:
                print(f"[DEBUG] Préparation fetch pour le jour : {day.strftime('%Y-%m-%d')}")
                tasks.append(fetch_profiles_per_day_with_retry(session, base_url, token, day))
                tasks.append(fetch_endpoint_per_day_with_retry(session, "treatments", base_url, token, day))
                tasks.append(fetch_endpoint_per_day_with_retry(session, "entries", base_url, token, day))
                tasks.append(fetch_endpoint_per_day_with_retry(session, "devicestatus", base_url, token, day))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[ERROR] Exception lors du fetch du batch {i+j}: {type(result).__name__} - {result}")
                    continue
                idx = j % 4
                if idx == 0:
                    all_profiles.extend(result)
                elif idx == 1:
                    all_treatments.extend(result)
                elif idx == 2:
                    all_entries.extend(result)
                elif idx == 3:
                    all_devicestatus.extend(result)
    print(f"[SUMMARY][async] profiles: {len(all_profiles)}, treatments: {len(all_treatments)}, entries: {len(all_entries)}, devicestatus: {len(all_devicestatus)}")
    if all_entries:
        try:
            from dateutil.parser import parse as parse_date
            oldest = min(parse_date(e.get('dateString', e.get('created_at', ''))) for e in all_entries if e.get('dateString') or e.get('created_at'))
            newest = max(parse_date(e.get('dateString', e.get('created_at', ''))) for e in all_entries if e.get('dateString') or e.get('created_at'))
            print(f"[SUMMARY][async] oldest entry: {oldest}, newest entry: {newest}")
        except Exception as e:
            print(f"[WARN] Impossible de parser les dates des entrées : {e}")
    return {
        "profile": all_profiles,
        "treatments": all_treatments,
        "entries": all_entries,
        "devicestatus": all_devicestatus
    }

def fetch_nightscout_data(base_url, token=None, days=7):
    """
    Version synchrone de fetch_nightscout_data_async (pour usage simple ou script).
    Synchronous version of fetch_nightscout_data_async (for simple/script usage).
    """
    try:
        return asyncio.run(fetch_nightscout_data_async(base_url, token, days))
    except Exception as e:
        return {"error": str(e)}

def get_profile_for_timestamp(profiles, timestamp):
    """
    Sélectionne le profil actif le plus proche pour un timestamp donné.
    Selects the closest active profile for a given timestamp.
    Args:
        profiles (list): Liste de profils Nightscout
        timestamp (datetime): Timestamp cible
    Returns:
        dict: Profil sélectionné ou None
    """
    from datetime import datetime, timezone
    valid_profiles = [p for p in profiles if "startDate" in p]
    valid_profiles.sort(key=lambda p: p["startDate"])
    ts = timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    selected = None
    for p in valid_profiles:
        try:
            start = datetime.fromisoformat(p["startDate"].replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if start <= ts:
                selected = p
            else:
                break
        except Exception:
            continue
    return selected
