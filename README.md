# Glycemia Profile Optimizer

## Français

Ce projet permet d'optimiser des profils glycémiques horaires (basal, ISF, CSF) à partir de données de capteur et d'insuline. Il utilise un modèle CatBoost multi-sortie pour simuler la glycémie future et propose des profils mutants optimisés autour d'un profil de référence. Les résultats incluent des exports CSV et des graphiques de comparaison.

### Fonctionnalités principales
- Préparation et filtrage des données glycémiques
- Entraînement d'un simulateur multi-horizon (CatBoost)
- Génération et optimisation de profils mutants
- Export des meilleurs profils en CSV
- Visualisation des profils optimaux

### Utilisation
1. Placez vos données dans `features_debug.csv` et un profil de base dans `profil_base.ini`.
2. Exécutez le script principal :
   ```bash
   python glycemia_profile_optimizer.py
   ```
3. Les meilleurs profils seront exportés et des graphiques générés.

### Dépendances
- Python 3.8+
- pandas, numpy, matplotlib, catboost, scikit-learn, scipy

Installez-les avec :
```bash
pip install -r requirements.txt
```

---

## English

This project optimizes hourly glycemic profiles (basal, ISF, CSF) from sensor and insulin data. It uses a multi-output CatBoost model to simulate future glucose and proposes optimized mutant profiles around a reference profile. Results include CSV exports and comparison plots.

### Main Features
- Glycemic data preparation and filtering
- Multi-horizon simulator training (CatBoost)
- Generation and optimization of mutant profiles
- Export of top profiles to CSV
- Visualization of optimal profiles

### Usage
1. Place your data in `features_debug.csv` and a base profile in `profil_base.ini`.
2. Run the main script:
   ```bash
   python glycemia_profile_optimizer.py
   ```
3. Top profiles will be exported and plots generated.

### Dependencies
- Python 3.8+
- pandas, numpy, matplotlib, catboost, scikit-learn, scipy

Install them with:
```bash
pip install -r requirements.txt
```
