import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("--- Démarrage du Pipeline d'Entraînement ---")

            # 1. Ingestion des données
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            # 2. Transformation des données (Nettoyage + Encoding + Lasso)
            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(
                train_data_path, 
                test_data_path
            )

            # 3. Entraînement du modèle (Recherche du meilleur modèle + Save)
            trainer = ModelTrainer()
            r2_score = trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"--- Fin du Pipeline. Score R2 final : {r2_score:.4f} ---")
            print(f"Entraînement réussi ! Score R2 : {r2_score}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()