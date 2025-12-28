from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le chemin src pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)
app = application

# Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route pour les prédictions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Afficher le formulaire
        return render_template('home.html')
    else:
        try:
            # Fonction helper pour récupérer les valeurs du formulaire
            def get_float(key, default=0.0):
                try:
                    return float(request.form.get(key, default))
                except:
                    return default
            
            def get_int(key, default=0):
                try:
                    return int(request.form.get(key, default))
                except:
                    return default
            
            def get_str(key, default=''):
                return request.form.get(key, default)
            
            # Récupérer toutes les données du formulaire
            data = CustomData(
                hectares=get_float('hectares', 6.0),
                agriblock=get_str('agriblock', 'Cuddalore'),
                variety=get_str('variety', 'CO_43'),
                soil_type=get_str('soil_type', 'alluvial'),
                seedrate=get_int('seedrate', 150),
                lp_mainfield=get_float('lp_mainfield', 75.0),
                nursery=get_str('nursery', 'dry'),
                nursery_area=get_int('nursery_area', 120),
                lp_nurseryarea=get_int('lp_nurseryarea', 6),
                dap_20days=get_int('dap_20days', 240),
                weed_thiobencarb=get_int('weed_thiobencarb', 600),
                urea_40days=get_int('urea_40days', 160),
                potash_50days=get_int('potash_50days', 96),
                micronutrients_70days=get_int('micronutrients_70days', 8),
                pest_60day=get_int('pest_60day', 400),
                rain_30d=get_float('rain_30d', 250.0),
                rain_30dai=get_float('rain_30dai', 25.0),
                rain_30_50d=get_float('rain_30_50d', 300.0),
                rain_30_50dai=get_float('rain_30_50dai', 30.0),
                rain_51_70d=get_float('rain_51_70d', 350.0),
                rain_51_70ai=get_float('rain_51_70ai', 35.0),
                rain_71_105d=get_float('rain_71_105d', 400.0),
                rain_71_105dai=get_float('rain_71_105dai', 40.0),
                min_temp_d1_d30=get_float('min_temp_d1_d30', 22.0),
                max_temp_d1_d30=get_float('max_temp_d1_d30', 34.0),
                min_temp_d31_d60=get_float('min_temp_d31_d60', 23.0),
                max_temp_d31_d60=get_float('max_temp_d31_d60', 35.0),
                min_temp_d61_d90=get_float('min_temp_d61_d90', 24.0),
                max_temp_d61_d90=get_float('max_temp_d61_d90', 36.0),
                min_temp_d91_d120=get_float('min_temp_d91_d120', 22.0),
                max_temp_d91_d120=get_float('max_temp_d91_d120', 33.0),
                wind_speed_d1_d30=get_int('wind_speed_d1_d30', 10),
                wind_speed_d31_d60=get_int('wind_speed_d31_d60', 8),
                wind_speed_d61_d90=get_int('wind_speed_d61_d90', 6),
                wind_speed_d91_d120=get_int('wind_speed_d91_d120', 10),
                wind_dir_d1_d30=get_str('wind_dir_d1_d30', 'SW'),
                wind_dir_d31_d60=get_str('wind_dir_d31_d60', 'W'),
                wind_dir_d61_d90=get_str('wind_dir_d61_d90', 'NNW'),
                wind_dir_d91_d120=get_str('wind_dir_d91_d120', 'WSW'),
                humidity_d1_d30=get_float('humidity_d1_d30', 72.0),
                humidity_d31_d60=get_int('humidity_d31_d60', 78),
                humidity_d61_d90=get_int('humidity_d61_d90', 88),
                humidity_d91_d120=get_int('humidity_d91_d120', 85),
                trash=get_int('trash', 570)
            )
            
            # Convertir en DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Données reçues pour prédiction: {pred_df.shape}")
            
            # Faire la prédiction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            logging.info(f"Prédiction réussie: {results[0]}")
            
            # Retourner le résultat
            return render_template('home.html', results=round(results[0], 2))
        
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction: {str(e)}")
            return render_template('home.html', error=str(e))

# Route pour vérifier l'état de l'application
@app.route('/health')
def health():
    return {'status': 'healthy', 'message': 'Application is running'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)