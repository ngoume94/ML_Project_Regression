import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Chemins vers les objets sauvegardés
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            logging.info("Chargement du modèle et du préprocesseur...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info(f"Features reçues: {features.columns.tolist()}")
            
            # 1. Transformation des données avec le preprocessor (TargetEncoder + StandardScaler)
            data_scaled = preprocessor.transform(features)
            logging.info(f"Données transformées, shape: {data_scaled.shape}")
            
            # 2. Prédiction avec le meilleur modèle
            preds = model.predict(data_scaled)
            logging.info(f"Prédiction effectuée: {preds[0]}")
            
            return preds
        
        except Exception as e:
            logging.error(f"Erreur dans PredictPipeline: {str(e)}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        hectares: float,
        agriblock: str,
        variety: str,
        soil_type: str,
        seedrate: int,
        lp_mainfield: float,
        nursery: str,
        nursery_area: int,
        lp_nurseryarea: int,
        dap_20days: int,
        weed_thiobencarb: int,
        urea_40days: int,
        potash_50days: int,
        micronutrients_70days: int,
        pest_60day: int,
        rain_30d: float,
        rain_30dai: float,
        rain_30_50d: float,
        rain_30_50dai: float,
        rain_51_70d: float,
        rain_51_70ai: float,
        rain_71_105d: float,
        rain_71_105dai: float,
        min_temp_d1_d30: float,
        max_temp_d1_d30: float,
        min_temp_d31_d60: float,
        max_temp_d31_d60: float,
        min_temp_d61_d90: float,
        max_temp_d61_d90: float,
        min_temp_d91_d120: float,
        max_temp_d91_d120: float,
        wind_speed_d1_d30: int,
        wind_speed_d31_d60: int,
        wind_speed_d61_d90: int,
        wind_speed_d91_d120: int,
        wind_dir_d1_d30: str,
        wind_dir_d31_d60: str,
        wind_dir_d61_d90: str,
        wind_dir_d91_d120: str,
        humidity_d1_d30: float,
        humidity_d31_d60: int,
        humidity_d61_d90: int,
        humidity_d91_d120: int,
        trash: int):
        """
        Initialise les données d'entrée avec TOUTES les 44 features du dataset.
        Les noms doivent correspondre EXACTEMENT aux colonnes du CSV d'entraînement.
        """
        self.hectares = hectares
        self.agriblock = agriblock
        self.variety = variety
        self.soil_type = soil_type
        self.seedrate = seedrate
        self.lp_mainfield = lp_mainfield
        self.nursery = nursery
        self.nursery_area = nursery_area
        self.lp_nurseryarea = lp_nurseryarea
        self.dap_20days = dap_20days
        self.weed_thiobencarb = weed_thiobencarb
        self.urea_40days = urea_40days
        self.potash_50days = potash_50days
        self.micronutrients_70days = micronutrients_70days
        self.pest_60day = pest_60day
        self.rain_30d = rain_30d
        self.rain_30dai = rain_30dai
        self.rain_30_50d = rain_30_50d
        self.rain_30_50dai = rain_30_50dai
        self.rain_51_70d = rain_51_70d
        self.rain_51_70ai = rain_51_70ai
        self.rain_71_105d = rain_71_105d
        self.rain_71_105dai = rain_71_105dai
        self.min_temp_d1_d30 = min_temp_d1_d30
        self.max_temp_d1_d30 = max_temp_d1_d30
        self.min_temp_d31_d60 = min_temp_d31_d60
        self.max_temp_d31_d60 = max_temp_d31_d60
        self.min_temp_d61_d90 = min_temp_d61_d90
        self.max_temp_d61_d90 = max_temp_d61_d90
        self.min_temp_d91_d120 = min_temp_d91_d120
        self.max_temp_d91_d120 = max_temp_d91_d120
        self.wind_speed_d1_d30 = wind_speed_d1_d30
        self.wind_speed_d31_d60 = wind_speed_d31_d60
        self.wind_speed_d61_d90 = wind_speed_d61_d90
        self.wind_speed_d91_d120 = wind_speed_d91_d120
        self.wind_dir_d1_d30 = wind_dir_d1_d30
        self.wind_dir_d31_d60 = wind_dir_d31_d60
        self.wind_dir_d61_d90 = wind_dir_d61_d90
        self.wind_dir_d91_d120 = wind_dir_d91_d120
        self.humidity_d1_d30 = humidity_d1_d30
        self.humidity_d31_d60 = humidity_d31_d60
        self.humidity_d61_d90 = humidity_d61_d90
        self.humidity_d91_d120 = humidity_d91_d120
        self.trash = trash

    def get_data_as_data_frame(self):
        """
        Transforme les attributs en DataFrame Pandas.
        Les noms de colonnes doivent correspondre EXACTEMENT au dataset d'entraînement.
        """
        try:
            custom_data_input_dict = {
                "Hectares": [self.hectares],  # Sans espace (corrigé)
                "Agriblock": [self.agriblock],
                "Variety": [self.variety],
                "Soil Types": [self.soil_type],
                "Seedrate(in Kg)": [self.seedrate],
                "LP_Mainfield(in Tonnes)": [self.lp_mainfield],
                "Nursery": [self.nursery],
                "Nursery area (Cents)": [self.nursery_area],
                "LP_nurseryarea(in Tonnes)": [self.lp_nurseryarea],
                "DAP_20days": [self.dap_20days],
                "Weed28D_thiobencarb": [self.weed_thiobencarb],
                "Urea_40Days": [self.urea_40days],
                "Potassh_50Days": [self.potash_50days],
                "Micronutrients_70Days": [self.micronutrients_70days],
                "Pest_60Day(in ml)": [self.pest_60day],
                "30DRain( in mm)": [self.rain_30d],
                "30DAI(in mm)": [self.rain_30dai],
                "30_50DRain( in mm)": [self.rain_30_50d],
                "30_50DAI(in mm)": [self.rain_30_50dai],
                "51_70DRain(in mm)": [self.rain_51_70d],
                "51_70AI(in mm)": [self.rain_51_70ai],
                "71_105DRain(in mm)": [self.rain_71_105d],
                "71_105DAI(in mm)": [self.rain_71_105dai],
                "Min temp_D1_D30": [self.min_temp_d1_d30],
                "Max temp_D1_D30": [self.max_temp_d1_d30],
                "Min temp_D31_D60": [self.min_temp_d31_d60],
                "Max temp_D31_D60": [self.max_temp_d31_d60],
                "Min temp_D61_D90": [self.min_temp_d61_d90],
                "Max temp_D61_D90": [self.max_temp_d61_d90],
                "Min temp_D91_D120": [self.min_temp_d91_d120],
                "Max temp_D91_D120": [self.max_temp_d91_d120],
                "Inst Wind Speed_D1_D30(in Knots)": [self.wind_speed_d1_d30],
                "Inst Wind Speed_D31_D60(in Knots)": [self.wind_speed_d31_d60],
                "Inst Wind Speed_D61_D90(in Knots)": [self.wind_speed_d61_d90],
                "Inst Wind Speed_D91_D120(in Knots)": [self.wind_speed_d91_d120],
                "Wind Direction_D1_D30": [self.wind_dir_d1_d30],
                "Wind Direction_D31_D60": [self.wind_dir_d31_d60],
                "Wind Direction_D61_D90": [self.wind_dir_d61_d90],
                "Wind Direction_D91_D120": [self.wind_dir_d91_d120],
                "Relative Humidity_D1_D30": [self.humidity_d1_d30],
                "Relative Humidity_D31_D60": [self.humidity_d31_d60],
                "Relative Humidity_D61_D90": [self.humidity_d61_d90],
                "Relative Humidity_D91_D120": [self.humidity_d91_d120],
                "Trash(in bundles)": [self.trash]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"DataFrame créé avec {len(df.columns)} colonnes")
            
            return df

        except Exception as e:
            raise CustomException(e, sys)