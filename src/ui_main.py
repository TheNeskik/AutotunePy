import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from nightscout_fetcher import fetch_nightscout_data, get_profile_for_timestamp
from autotune import run_autotune
from autotune_rnn import prepare_rnn_sequences, train_rnn_model, predict_with_rnn, format_rnn_output
from autotune_transformer import train_transformer_model, predict_with_transformer, format_transformer_output
from debug_features_export import generate_features_csv
from autotune_rl import build_rl_dataset, fitted_q_iteration, extract_best_profile

class AutotuneWorker(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, url, token, days):
        super().__init__()
        self.url = url
        self.token = token
        self.days = days

    def run(self):
        def gui_callback(msg):
            self.status_update.emit(msg)
        data = fetch_nightscout_data(self.url, self.token, self.days)
        if "error" in data:
            self.result_ready.emit(f"Error: {data['error']}")
            return
        result = run_autotune(data, gui_callback=gui_callback)
        self.result_ready.emit(result)

class AutotuneRNNWorker(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, url, token, days):
        super().__init__()
        self.url = url
        self.token = token
        self.days = days

    def run(self):
        def gui_callback(msg):
            self.status_update.emit(msg)
        data = fetch_nightscout_data(self.url, self.token, self.days)
        if "error" in data:
            self.result_ready.emit(f"Error: {data['error']}")
            return
        gui_callback("Préparation des données pour le RNN...")
        X, y_basal, y_isf, y_csf = prepare_rnn_sequences(data["entries"], data["treatments"], data["profile"])
        gui_callback("Entraînement du modèle RNN...")
        model = train_rnn_model(X, y_basal, y_isf, y_csf)
        gui_callback("Prédiction du profil optimal (basal, ISF, CSF) avec le RNN...")
        y_basal_pred, y_isf_pred, y_csf_pred = predict_with_rnn(model, X)
        result = format_rnn_output(y_basal_pred, y_isf_pred, y_csf_pred)
        self.result_ready.emit(result)

class AutotuneTransformerWorker(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, url, token, days):
        super().__init__()
        self.url = url
        self.token = token
        self.days = days

    def run(self):
        import os
        from autotune_transformer import load_features_from_csv, train_transformer_model, predict_with_transformer, format_transformer_output
        # Suppression du CSV existant pour éviter les doublons
        if os.path.exists('features_debug.csv'):
            try:
                os.remove('features_debug.csv')
            except Exception as e:
                self.status_update.emit(f"Impossible de supprimer l'ancien CSV : {e}")
        self.status_update.emit("Génération des features avancées (debug_features_export.py)...")
        # Appel direct de la fonction Python pour générer le CSV
        try:
            generate_features_csv(days=self.days, output_path='features_debug.csv')
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de la génération des features : {e}")
            return
        self.status_update.emit("Chargement des features et préparation des séquences...")
        try:
            X, y_basal, y_isf, y_csf = load_features_from_csv('features_debug.csv', days=self.days)
        except Exception as e:
            self.result_ready.emit(f"Erreur lors du chargement des features : {e}")
            return
        self.status_update.emit("Entraînement du modèle Transformer...")
        try:
            model = train_transformer_model(X, y_basal, y_isf, y_csf)
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de l'entraînement du modèle : {e}")
            return
        self.status_update.emit("Prédiction du profil optimal (basal, ISF, CSF) avec le Transformer...")
        try:
            y_basal_pred, y_isf_pred, y_csf_pred = predict_with_transformer(model, X)
            # Format profil 24h arrondi réglementaire
            result = format_transformer_output(y_basal_pred, y_isf_pred, y_csf_pred)
            self.result_ready.emit(result)
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de la prédiction : {e}")

class AutotuneRLWorker(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, days):
        super().__init__()
        self.days = days

    def run(self):
        self.status_update.emit("Préparation du dataset RL (features_debug.csv)...")
        try:
            transitions = build_rl_dataset('features_debug.csv', days=self.days)
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de la préparation du dataset RL : {e}")
            return
        self.status_update.emit("Entraînement du Q-fitted RL (XGBoost)...")
        try:
            Q = fitted_q_iteration(transitions, n_iter=10)
            self.status_update.emit("Extraction du profil optimal RL (24h)...")
            _, result = extract_best_profile(Q, transitions)
            self.result_ready.emit(result)
        except Exception as e:
            self.result_ready.emit(f"Erreur RL : {e}")

class GenerateCSVWorker(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, days):
        super().__init__()
        self.days = days

    def run(self):
        def gui_status(msg):
            self.status_update.emit(msg)
        self.status_update.emit("Génération du CSV features_debug.csv...")
        try:
            generate_features_csv(days=self.days, output_path='features_debug.csv', status_callback=gui_status)
            self.result_ready.emit("CSV features_debug.csv généré avec succès.")
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de la génération du CSV : {e}")

class AutotuneOptimalProfileWorker(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, days):
        super().__init__()
        self.days = days

    def run(self):
        from autotune_rl import pipeline_optimal_profile
        self.status_update.emit("Génération du profil optimal par heure (XGBoost reward~état+action)...")
        try:
            pipeline_optimal_profile('features_debug.csv', days=self.days, output_csv='profil_optimal_par_heure.csv')
            self.result_ready.emit("Profil optimal par heure exporté dans profil_optimal_par_heure.csv.")
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de l'export du profil optimal : {e}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.setWindowTitle("Autotune GUI")
        self.setGeometry(100, 100, 600, 520)
        self.worker = None
        # Widgets
        self.url_label = QLabel("Nightscout URL:")
        self.url_input = QLineEdit()
        self.url_input.setText(os.getenv("NIGHTSCOUT_URL", ""))
        self.token_label = QLabel("API Token:")
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setText(os.getenv("NIGHTSCOUT_TOKEN", ""))
        self.days_label = QLabel("Nombre de jours à analyser:")
        self.days_input = QSpinBox()
        self.days_input.setMinimum(1)
        self.days_input.setMaximum(730)
        self.days_input.setValue(7)
        self.run_button = QPushButton("Autotune Classique")
        self.rnn_button = QPushButton("Autotune RNN")
        self.transformer_button = QPushButton("Autotune Transformer")
        self.rl_button = QPushButton("Autotune RL (batch)")
        self.csv_button = QPushButton("Générer CSV features_debug.csv")
        self.optimal_profile_button = QPushButton("Profil optimal (XGBoost)")
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        # Layouts
        url_layout = QHBoxLayout()
        url_layout.addWidget(self.url_label)
        url_layout.addWidget(self.url_input)
        token_layout = QHBoxLayout()
        token_layout.addWidget(self.token_label)
        token_layout.addWidget(self.token_input)
        days_layout = QHBoxLayout()
        days_layout.addWidget(self.days_label)
        days_layout.addWidget(self.days_input)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.rnn_button)
        button_layout.addWidget(self.transformer_button)
        button_layout.addWidget(self.rl_button)
        button_layout.addWidget(self.csv_button)
        button_layout.addWidget(self.optimal_profile_button)
        main_layout = QVBoxLayout()
        main_layout.addLayout(url_layout)
        main_layout.addLayout(token_layout)
        main_layout.addLayout(days_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_box)
        self.setLayout(main_layout)
        # Connect buttons
        self.run_button.clicked.connect(self.run_autotune)
        self.rnn_button.clicked.connect(self.run_rnn_autotune)
        self.transformer_button.clicked.connect(self.run_transformer_autotune)
        self.rl_button.clicked.connect(self.run_rl_autotune)
        self.csv_button.clicked.connect(self.run_generate_csv)
        self.optimal_profile_button.clicked.connect(self.run_optimal_profile)

    def save_env(self):
        url = self.url_input.text().strip()
        token = self.token_input.text().strip()
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(f"NIGHTSCOUT_URL={url}\nNIGHTSCOUT_TOKEN={token}\n")

    def run_autotune(self):
        self.save_env()
        url = self.url_input.text().strip()
        token = self.token_input.text().strip()
        days = self.days_input.value()
        if url:
            self.result_box.setHtml('<div style="text-align:center;">⏳<br>Analyse en cours... Veuillez patienter.</div>')
            self.result_box.setVisible(True)
            self.run_button.setEnabled(False)
            self.rnn_button.setEnabled(False)
            self.transformer_button.setEnabled(False)
            self.rl_button.setEnabled(False)
            self.worker = AutotuneWorker(url, token, days)
            self.worker.status_update.connect(self.result_box.setText)
            self.worker.result_ready.connect(self.display_result)
            self.worker.finished.connect(self.on_analysis_finished)
            self.worker.start()

    def run_rnn_autotune(self):
        self.save_env()
        url = self.url_input.text().strip()
        token = self.token_input.text().strip()
        days = self.days_input.value()
        if url:
            self.result_box.setHtml('<div style="text-align:center;">⏳<br>Entraînement RNN en cours... Veuillez patienter.</div>')
            self.result_box.setVisible(True)
            self.run_button.setEnabled(False)
            self.rnn_button.setEnabled(False)
            self.transformer_button.setEnabled(False)
            self.rl_button.setEnabled(False)
            self.worker = AutotuneRNNWorker(url, token, days)
            self.worker.status_update.connect(self.result_box.setText)
            self.worker.result_ready.connect(self.display_result)
            self.worker.finished.connect(self.on_analysis_finished)
            self.worker.start()

    def run_transformer_autotune(self):
        self.save_env()
        url = self.url_input.text().strip()
        token = self.token_input.text().strip()
        days = self.days_input.value()
        if url:
            self.result_box.setHtml('<div style="text-align:center;">⏳<br>Entraînement Transformer en cours... Veuillez patienter.</div>')
            self.result_box.setVisible(True)
            self.run_button.setEnabled(False)
            self.rnn_button.setEnabled(False)
            self.transformer_button.setEnabled(False)
            self.rl_button.setEnabled(False)
            self.worker = AutotuneTransformerWorker(url, token, days)
            self.worker.status_update.connect(self.result_box.setText)
            self.worker.result_ready.connect(self.display_result)
            self.worker.finished.connect(self.on_analysis_finished)
            self.worker.start()

    def run_rl_autotune(self):
        days = self.days_input.value()
        self.result_box.setHtml('<div style="text-align:center;">⏳<br>RL batch en cours... Veuillez patienter.</div>')
        self.result_box.setVisible(True)
        self.run_button.setEnabled(False)
        self.rnn_button.setEnabled(False)
        self.transformer_button.setEnabled(False)
        self.rl_button.setEnabled(False)
        self.worker = AutotuneRLWorker(days)
        self.worker.status_update.connect(self.result_box.setText)
        self.worker.result_ready.connect(self.display_result)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def run_generate_csv(self):
        days = self.days_input.value()
        self.result_box.setHtml('<div style="text-align:center;">⏳<br>Génération du CSV en cours...</div>')
        self.result_box.setVisible(True)
        self.run_button.setEnabled(False)
        self.rnn_button.setEnabled(False)
        self.transformer_button.setEnabled(False)
        self.rl_button.setEnabled(False)
        self.csv_button.setEnabled(False)
        self.worker = GenerateCSVWorker(days)
        self.worker.status_update.connect(self.result_box.setText)
        self.worker.result_ready.connect(self.display_result)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def run_optimal_profile(self):
        days = self.days_input.value()
        self.result_box.setHtml('<div style="text-align:center;">⏳<br>Calcul du profil optimal par heure...</div>')
        self.result_box.setVisible(True)
        self.run_button.setEnabled(False)
        self.rnn_button.setEnabled(False)
        self.transformer_button.setEnabled(False)
        self.rl_button.setEnabled(False)
        self.csv_button.setEnabled(False)
        self.optimal_profile_button.setEnabled(False)
        self.worker = AutotuneOptimalProfileWorker(days)
        self.worker.status_update.connect(self.result_box.setText)
        self.worker.result_ready.connect(self.display_result)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_finished(self):
        self.run_button.setEnabled(True)
        self.rnn_button.setEnabled(True)
        self.transformer_button.setEnabled(True)
        self.rl_button.setEnabled(True)
        self.csv_button.setEnabled(True)
        self.optimal_profile_button.setEnabled(True)

    def display_result(self, result):
        self.result_box.setText(result)
