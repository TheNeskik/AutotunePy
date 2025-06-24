import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QSpinBox, QTableWidget, QTableWidgetItem, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from debug_features_export import generate_features_csv
from glycemia_profile_optimizer import (
    prepare_data, load_profile_from_ini, train_catboost_multioutput,
    optimize_profiles, summarize_top_profiles, export_hourly_profiles, plot_mutant_vs_baseline
)
from PyQt5.QtGui import QPixmap
import io
import contextlib

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
            generate_features_csv(days=self.days, output_path='../data/features_debug.csv', status_callback=gui_status)
            self.result_ready.emit("CSV features_debug.csv généré avec succès.")
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de la génération du CSV : {e}")

class GlycemiaProfileOptimizerWorker(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def run(self):
        try:
            self.status_update.emit("Lancement de l'optimisation du profil glycémique...")
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
            outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))
            plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots'))
            csv_path = os.path.join(data_dir, 'features_debug.csv')
            ini_path = os.path.join(data_dir, 'profil_base.ini')
            if not os.path.exists(csv_path):
                self.result_ready.emit("Le fichier features_debug.csv est introuvable. Veuillez d'abord générer les données.")
                return
            if not os.path.exists(ini_path):
                self.result_ready.emit("Le fichier profil_base.ini est introuvable. Veuillez d'abord créer/éditer le profil basal.")
                return
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                X, y, df, feature_cols = prepare_data(csv_path)
                baseline_profile = load_profile_from_ini(ini_path)
                self.status_update.emit("Entraînement du modèle CatBoost...")
                model = train_catboost_multioutput(X, y, save_path=os.path.join(models_dir, 'model_catboost_multi.cbm'))
                self.status_update.emit("Optimisation des profils mutants...")
                results = optimize_profiles(model, df, feature_cols, baseline_profile, n_profiles=100)
                summarize_top_profiles(results, top_k=3)
                export_hourly_profiles(results, top_k=3, save_csv=True, outputs_dir=outputs_dir)
                plot_mutant_vs_baseline(results, baseline_profile, top_k=3, plots_dir=plots_dir)
            logs = buf.getvalue()
            self.result_ready.emit(logs + "\nOptimisation terminée. Profils et graphiques générés dans outputs/ et plots/.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.result_ready.emit(f"Erreur lors de l'optimisation : {e}\n{tb}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.setWindowTitle("Autotune GUI")
        self.setGeometry(100, 100, 700, 700)
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
        self.csv_button = QPushButton("Fetch Data (CSV)")
        self.optimizer_button = QPushButton("Lancer Glycemia Profile Optimizer")
        self.edit_profile_button = QPushButton("Éditer profil basal")
        self.profile_selector = QComboBox()
        self.profile_selector.addItems(["Afficher profil 1", "Afficher profil 2", "Afficher profil 3"])
        self.profile_selector.setEnabled(True)
        self.profile_selector.currentIndexChanged.connect(self.display_selected_profile)
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.profile_table = QTableWidget()
        self.profile_table.setColumnCount(4)
        self.profile_table.setHorizontalHeaderLabels(["Heure", "Basal", "ISF", "CSF"])
        self.profile_table.setVisible(False)
        self.profile_image_label = QLabel()
        self.profile_image_label.setVisible(False)
        self.profile_image_label.setAlignment(Qt.AlignCenter)
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
        button_layout.addWidget(self.csv_button)
        button_layout.addWidget(self.edit_profile_button)
        button_layout.addWidget(self.optimizer_button)
        button_layout.addWidget(self.profile_selector)
        main_layout = QVBoxLayout()
        main_layout.addLayout(url_layout)
        main_layout.addLayout(token_layout)
        main_layout.addLayout(days_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_box)
        main_layout.addWidget(self.profile_table)
        main_layout.addWidget(self.profile_image_label)
        self.setLayout(main_layout)
        # Connect buttons
        self.csv_button.clicked.connect(self.run_generate_csv)
        self.optimizer_button.clicked.connect(self.run_optimizer)
        self.edit_profile_button.clicked.connect(self.open_profile_ini)
        self.update_optimizer_button()

    def save_env(self):
        url = self.url_input.text().strip()
        token = self.token_input.text().strip()
        try:
            with open('.env', 'w', encoding='utf-8') as f:
                f.write(f"NIGHTSCOUT_URL={url}\nNIGHTSCOUT_TOKEN={token}\n")
        except PermissionError:
            self.result_box.setText("Erreur : impossible d'écrire le fichier .env (Permission refusée). Fermez les applications qui utilisent ce fichier et réessayez.")
        except Exception as e:
            self.result_box.setText(f"Erreur lors de l'écriture du fichier .env : {e}")

    def run_generate_csv(self):
        self.save_env()
        days = self.days_input.value()
        # Création auto du dossier data si besoin
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
            except Exception as e:
                self.result_box.setText(f"Erreur lors de la création du dossier data : {e}")
                return
        # Correction : chemin absolu pour le CSV
        csv_path = os.path.join(data_dir, 'features_debug.csv')
        self.result_box.setHtml('<div style="text-align:center;">⏳<br>Génération du CSV en cours...</div>')
        self.result_box.setVisible(True)
        self.csv_button.setEnabled(False)
        self.optimizer_button.setEnabled(False)
        self.worker = GenerateCSVWorkerWithPath(days, csv_path)
        self.worker.status_update.connect(self.result_box.setText)
        self.worker.result_ready.connect(self.display_result)
        self.worker.finished.connect(self.on_csv_finished)
        self.worker.start()

    def run_optimizer(self):
        # Création auto du dossier outputs si besoin
        outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))
        if not os.path.exists(outputs_dir):
            try:
                os.makedirs(outputs_dir)
            except Exception as e:
                self.result_box.setText(f"Erreur lors de la création du dossier outputs : {e}")
                return
        self.result_box.setHtml('<div style="text-align:center;">⏳<br>Lancement de Glycemia Profile Optimizer...</div>')
        self.result_box.setVisible(True)
        self.csv_button.setEnabled(False)
        self.optimizer_button.setEnabled(False)
        self.worker = GlycemiaProfileOptimizerWorker()
        self.worker.status_update.connect(self.result_box.setText)
        self.worker.result_ready.connect(self.display_result)
        self.worker.finished.connect(self.on_optimizer_finished)
        self.worker.start()

    def on_csv_finished(self):
        self.csv_button.setEnabled(True)
        self.update_optimizer_button()

    def on_optimizer_finished(self):
        self.csv_button.setEnabled(True)
        self.update_optimizer_button()

    def update_optimizer_button(self):
        # Active le bouton optimizer seulement si le CSV existe dans le bon dossier
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        csv_path = os.path.join(data_dir, 'features_debug.csv')
        if os.path.exists(csv_path):
            self.optimizer_button.setEnabled(True)
        else:
            self.optimizer_button.setEnabled(False)

    def display_result(self, result):
        self.result_box.setText(result)

    def display_selected_profile(self):
        # Création auto du dossier outputs si besoin
        outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))
        plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots'))
        if not os.path.exists(outputs_dir):
            try:
                os.makedirs(outputs_dir)
            except Exception as e:
                self.result_box.setText(f"Erreur lors de la création du dossier outputs : {e}")
                self.profile_table.setVisible(False)
                self.profile_image_label.setVisible(False)
                return
        idx = self.profile_selector.currentIndex() + 1
        csv_path = os.path.join('outputs', f'profile_{idx}_hourly.csv')
        if not os.path.exists(csv_path):
            self.result_box.setText(f"Fichier {csv_path} introuvable.")
            self.profile_table.setVisible(False)
            self.profile_image_label.setVisible(False)
            return
        import csv
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = list(csv.reader(csvfile))
            if not reader or len(reader) < 2:
                self.result_box.setText(f"Fichier {csv_path} vide ou invalide.")
                self.profile_table.setVisible(False)
                self.profile_image_label.setVisible(False)
                return
            self.profile_table.setRowCount(len(reader) - 1)
            self.profile_table.setColumnCount(len(reader[0]))
            self.profile_table.setHorizontalHeaderLabels(reader[0])
            for row_idx, row in enumerate(reader[1:]):
                for col_idx, val in enumerate(row):
                    self.profile_table.setItem(row_idx, col_idx, QTableWidgetItem(val))
            self.profile_table.setVisible(True)
        # Affichage de l'image du profil
        img_path = os.path.join(plots_dir, f'profile_mutant_{idx}_vs_baseline.png')
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                self.profile_image_label.setPixmap(pixmap.scaledToWidth(900))
                self.profile_image_label.setVisible(True)
            else:
                self.profile_image_label.setVisible(False)
        else:
            self.profile_image_label.setVisible(False)

    def open_profile_ini(self):
        # Ouvre le fichier profil_base.ini dans l'éditeur système, le crée si absent
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        ini_path = os.path.join(data_dir, 'profil_base.ini')
        if not os.path.exists(ini_path):
            # Création du fichier avec valeurs par défaut
            default_ini = '[basal]\n' + '\n'.join([f'{h:02d} = 1.00' for h in range(24)]) + '\n\n'
            default_ini += '[isf]\n' + '\n'.join([f'{h:02d} = 50.0' for h in range(24)]) + '\n\n'
            default_ini += '[csf]\n' + '\n'.join([f'{h:02d} = 10.0' for h in range(24)]) + '\n'
            try:
                with open(ini_path, 'w', encoding='utf-8') as f:
                    f.write(default_ini)
            except Exception as e:
                self.result_box.setText(f"Erreur lors de la création du profil : {e}")
                return
        try:
            if sys.platform.startswith('win'):
                os.startfile(ini_path)
            elif sys.platform.startswith('darwin'):
                import subprocess
                subprocess.Popen(['open', ini_path])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', ini_path])
        except Exception as e:
            self.result_box.setText(f"Erreur lors de l'ouverture du profil : {e}")

# Ajout d'une version du worker qui prend le chemin du CSV en argument
class GenerateCSVWorkerWithPath(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)
    def __init__(self, days, output_path):
        super().__init__()
        self.days = days
        self.output_path = output_path
    def run(self):
        def gui_status(msg):
            self.status_update.emit(msg)
        self.status_update.emit(f"Génération du CSV {self.output_path}...")
        try:
            generate_features_csv(days=self.days, output_path=self.output_path, status_callback=gui_status)
            self.result_ready.emit(f"CSV {self.output_path} généré avec succès.")
        except Exception as e:
            self.result_ready.emit(f"Erreur lors de la génération du CSV : {e}")
