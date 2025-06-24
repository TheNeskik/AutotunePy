# DEPRECATED: Ce module n'est plus utilisé dans le pipeline principal. Gardé pour référence et recherche uniquement.
# DEPRECATED: This module is no longer used in the main pipeline. Kept for reference and research purposes only.
#
# Utilisation typique :
# X, y_basal, y_isf, y_csf = load_features_from_csv('features_debug.csv', days=nb_jours)
# model = train_transformer_model(X, y_basal, y_isf, y_csf)
# y_pred = predict_with_transformer(model, X)
#
# Typical usage:
# X, y_basal, y_isf, y_csf = load_features_from_csv('features_debug.csv', days=nb_days)
# model = train_transformer_model(X, y_basal, y_isf, y_csf)
# y_pred = predict_with_transformer(model, X)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Simple Transformer Encoder block for time series
class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, input_shape, d_model=64, num_heads=4, ff_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.input_proj = layers.Dense(d_model)
        self.encoder_layers = [
            layers.LayerNormalization(),
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
            layers.Dropout(0.1),
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(0.1)
        ] * num_layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.out_basal = layers.Dense(1, name='basal')
        self.out_isf = layers.Dense(1, name='isf')
        self.out_csf = layers.Dense(1, name='csf')

    def call(self, x, training=False):
        x = self.input_proj(x)
        for i in range(0, len(self.encoder_layers), 7):
            norm1 = self.encoder_layers[i](x)
            attn = self.encoder_layers[i+1](norm1, norm1)
            attn = self.encoder_layers[i+2](attn, training=training)
            x = x + attn
            norm2 = self.encoder_layers[i+3](x)
            ff = self.encoder_layers[i+4](norm2)
            ff = self.encoder_layers[i+5](ff)
            ff = self.encoder_layers[i+6](ff, training=training)
            x = x + ff
        x = self.global_pool(x)
        return self.out_basal(x), self.out_isf(x), self.out_csf(x)

def load_features_from_csv(csv_path, days=None):
    df = pd.read_csv(csv_path, parse_dates=['datetime_local'])
    if days is not None:
        max_date = df['datetime_local'].max()
        min_date = max_date - pd.Timedelta(days=days)
        df = df[df['datetime_local'] >= min_date]
    # Sélectionne toutes les colonnes sauf timestamp, datetime, et cibles
    feature_cols = [c for c in df.columns if c not in ['timestamp_ms','datetime_local','basal_profile','isf','csf']]
    X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    # Les cibles sont les profils à prédire
    y_basal = df['basal_profile'].to_numpy(dtype=np.float32)
    y_isf = df['isf'].to_numpy(dtype=np.float32)
    y_csf = df['csf'].to_numpy(dtype=np.float32)
    # Reshape pour séquence (batch, time, features) : ici on suppose 1h=12 pas (5min)
    # On découpe en séquences horaires si besoin
    seq_len = 12
    n_seq = X.shape[0] // seq_len
    X_seq = X[:n_seq*seq_len].reshape(n_seq, seq_len, -1)
    y_basal_seq = y_basal[:n_seq*seq_len].reshape(n_seq, seq_len).mean(axis=1)
    y_isf_seq = y_isf[:n_seq*seq_len].reshape(n_seq, seq_len).mean(axis=1)
    y_csf_seq = y_csf[:n_seq*seq_len].reshape(n_seq, seq_len).mean(axis=1)
    return X_seq, y_basal_seq, y_isf_seq, y_csf_seq

def build_transformer_model(input_shape):
    model = TimeSeriesTransformer(input_shape)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_transformer_model(X, y_basal, y_isf, y_csf, epochs=500, batch_size=64, early_stopping_patience=25, verbose=1):
    model = build_transformer_model(X.shape[1:])
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
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
    # Affichage de la courbe de loss
    try:
        plt.figure(figsize=(8,4))
        plt.plot(history.history['loss'], label='Train loss')
        plt.plot(history.history['val_loss'], label='Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Courbe de loss (Transformer)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('transformer_loss.png')
        print("[INFO] Courbe de loss sauvegardée dans transformer_loss.png")
    except Exception as e:
        print(f"[WARN] Impossible d'afficher la courbe de loss : {e}")
    return model

def predict_with_transformer(model, X):
    y_basal, y_isf, y_csf = model.predict(X)
    return y_basal, y_isf, y_csf

def format_transformer_output(y_basal_pred, y_isf_pred, y_csf_pred):
    """
    Regroupe les sorties du modèle par heure (modulo 24) et applique l'arrondi réglementaire :
    - Basal : incrément 0.05
    - ISF : arrondi à l'unité
    - CSF : arrondi à la dizaine
    Retourne une string formatée pour affichage.
    """
    import numpy as np
    n = len(y_basal_pred)
    hours = np.arange(n) % 24
    basal_hourly = [[] for _ in range(24)]
    isf_hourly = [[] for _ in range(24)]
    csf_hourly = [[] for _ in range(24)]
    for i in range(n):
        h = hours[i]
        basal_hourly[h].append(y_basal_pred[i][0])
        isf_hourly[h].append(y_isf_pred[i][0])
        csf_hourly[h].append(y_csf_pred[i][0])
    # Moyenne par heure puis arrondi réglementaire
    basal_24h = [round(np.nanmean(basal_hourly[h]) / 0.05) * 0.05 for h in range(24)]
    isf_24h = [int(round(np.nanmean(isf_hourly[h]))) for h in range(24)]
    csf_24h = [int(round(np.nanmean(csf_hourly[h]) / 10) * 10) for h in range(24)]
    # Formatage
    result = ["<b>Profil Transformer 24h (moyenne, arrondi réglementaire) :</b>"]
    for h in range(24):
        result.append(f"{h:02d}h  Basal: {basal_24h[h]:.2f}  ISF: {isf_24h[h]:d}  CSF: {csf_24h[h]:d}")
    return "<br>".join(result)
