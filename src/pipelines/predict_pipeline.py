import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Chemins vers les objets sauvegardés lors de l'entraînement
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            logging.info("Chargement du modèle et du préprocesseur...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # 1. Transformation des données brutes avec le StandardScaler + TargetEncoder
            data_scaled = preprocessor.transform(features)
            
            # 2. Prédiction avec le meilleur modèle trouvé (XGBoost, RandomForest, etc.)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        variety: str,
        season: str,
        soil_type: str,
        fertilizer: str,
        rainfall: float,
        temperature: float,
        area: float):
        """
        Initialise les données d'entrée. 
        Note : Remplacez ces noms par les colonnes exactes de votre fichier CSV.
        """
        self.variety = variety
        self.season = season
        self.soil_type = soil_type
        self.fertilizer = fertilizer
        self.rainfall = rainfall
        self.temperature = temperature
        self.area = area

    def get_data_as_data_frame(self):
        """Transforme les attributs en DataFrame Pandas."""
        try:
            custom_data_input_dict = {
                "Variety": [self.variety],
                "Season": [self.season],
                "Soil_Type": [self.soil_type],
                "Fertilizer": [self.fertilizer],
                "Rainfall": [self.rainfall],
                "Temperature": [self.temperature],
                "Area": [self.area],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)