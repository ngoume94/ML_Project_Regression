import os
import sys
from dataclasses import dataclass
import warnings

# Modèles
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, read_config

# Supprimer les warnings de convergence
warnings.filterwarnings('ignore', category=Warning)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.config = read_config()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Séparation des données d'entraînement et de test (X et y)")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Définition des modèles avec paramètres de convergence optimisés
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "Ridge": Ridge(max_iter=100000),  # Augmenté pour convergence
                "Lasso": Lasso(max_iter=100000, tol=0.01)  # Augmenté + tolérance ajustée
            }

            # Grille d'hyperparamètres optimisée
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [None, 10, 20, 30]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.1, 0.5],
                    'n_estimators': [50, 100, 200]
                },
                "Ridge": {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                },
                "Lasso": {
                    # Valeurs d'alpha plus grandes pour faciliter la convergence
                    'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]
                }
            }

            logging.info("Lancement de l'évaluation des modèles avec GridSearchCV")
            
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            # Extraction du meilleur modèle
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Seuil de qualité minimal
            if best_model_score < 0.6:
                raise CustomException("Aucun modèle n'a atteint le score R2 minimum requis.")

            logging.info(f"Meilleur modèle trouvé : {best_model_name} avec un R2 de {best_model_score:.4f}")

            # Sauvegarde du modèle
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Prédiction finale
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Score R2 final sur le test set : {r2_square:.6f}")
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)