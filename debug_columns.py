"""
Script pour diagnostiquer les colonnes du preprocessor et du modèle
"""
import pickle
import pandas as pd

# Charger le preprocessor
with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Charger le modèle
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=" * 80)
print("DIAGNOSTIC DES COLONNES")
print("=" * 80)

# Afficher les transformers du preprocessor
print("\n Transformers dans le ColumnTransformer:")
for name, transformer, columns in preprocessor.transformers_:
    print(f"\n{name}:")
    print(f"  Type: {type(transformer).__name__}")
    print(f"  Colonnes: {list(columns)}")

# Essayer de créer un DataFrame de test
print("\n" + "=" * 80)
print("TEST AVEC DONNÉES D'EXEMPLE")
print("=" * 80)

test_data = {
    "Hectares": [6],
    "Agriblock": ["Cuddalore"],
    "Variety": ["CO_43"],
    "Soil Types": ["alluvial"],
    "Seedrate(in Kg)": [150],
    "LP_Mainfield(in Tonnes)": [75.0],
    "Nursery": ["dry"],
    "Nursery area (Cents)": [120],
    "LP_nurseryarea(in Tonnes)": [6],
    "DAP_20days": [240],
    "Weed28D_thiobencarb": [600],
    "Urea_40Days": [160],
    "Potassh_50Days": [96],
    "Micronutrients_70Days": [8],
    "Pest_60Day(in ml)": [400],
    "30DRain( in mm)": [250.0],
    "30DAI(in mm)": [25.0],
    "30_50DRain( in mm)": [300.0],
    "30_50DAI(in mm)": [30.0],
    "51_70DRain(in mm)": [350.0],
    "51_70AI(in mm)": [35.0],
    "71_105DRain(in mm)": [400.0],
    "71_105DAI(in mm)": [40.0],
    "Min temp_D1_D30": [22.0],
    "Max temp_D1_D30": [34.0],
    "Min temp_D31_D60": [23.0],
    "Max temp_D31_D60": [35.0],
    "Min temp_D61_D90": [24.0],
    "Max temp_D61_D90": [36.0],
    "Min temp_D91_D120": [22.0],
    "Max temp_D91_D120": [33.0],
    "Inst Wind Speed_D1_D30(in Knots)": [10],
    "Inst Wind Speed_D31_D60(in Knots)": [8],
    "Inst Wind Speed_D61_D90(in Knots)": [6],
    "Inst Wind Speed_D91_D120(in Knots)": [10],
    "Wind Direction_D1_D30": ["SW"],
    "Wind Direction_D31_D60": ["W"],
    "Wind Direction_D61_D90": ["NNW"],
    "Wind Direction_D91_D120": ["WSW"],
    "Relative Humidity_D1_D30": [72.0],
    "Relative Humidity_D31_D60": [78],
    "Relative Humidity_D61_D90": [88],
    "Relative Humidity_D91_D120": [85],
    "Trash(in bundles)": [570]
}

df_test = pd.DataFrame(test_data)

print(f"\n Colonnes dans le DataFrame de test: {len(df_test.columns)}")
print(df_test.columns.tolist())

# Essayer la transformation
try:
    print("\n Tentative de transformation...")
    transformed = preprocessor.transform(df_test)
    print(f" Transformation réussie! Shape: {transformed.shape}")
    
    # Essayer la prédiction
    print("\n Tentative de prédiction...")
    prediction = model.predict(transformed)
    print(f" Prédiction réussie: {prediction[0]:.2f} kg")
    
except Exception as e:
    print(f" ERREUR: {e}")
    print(f"\nDétails de l'erreur:")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("FIN DU DIAGNOSTIC")
print("=" * 80)