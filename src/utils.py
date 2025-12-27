import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
import warnings
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

# Supprimer les warnings de convergence Lasso
warnings.filterwarnings('ignore', category=Warning)

def save_object(file_path, obj):
    """Sauvegarde un objet Python (modèle, préprocesseur) au format pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Objet sauvegardé avec succès à : {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Charge un objet pickle (utile pour la prédiction)."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def read_config(config_path="config.yaml"):
    """Lit le fichier de configuration YAML."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Entraîne plusieurs modèles avec optimisation d'hyperparamètres 
    et retourne un rapport de performance R2.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {})

            logging.info(f"Début de l'entraînement pour le modèle : {model_name}")

            # Recherche des meilleurs paramètres avec verbosité réduite
            gs = GridSearchCV(
                model, 
                para, 
                cv=3, 
                n_jobs=-1, 
                verbose=0,  # Changé de 1 à 0 pour réduire les logs
                scoring='r2'
            )
            gs.fit(X_train, y_train)

            # Mise à jour du modèle avec les meilleurs réglages
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Prédictions
            y_test_pred = model.predict(X_test)

            # Calcul du score R2
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            
            logging.info(f"Modèle {model_name} terminé. Score R2 : {test_model_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)