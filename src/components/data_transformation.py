import sys
import os
from dataclasses import dataclass
import warnings

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, read_config

# Supprimer les warnings de convergence
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. TRANSFORMATEUR PERSONNALISÉ ---
class TargetGuidedOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mappings = {}

    def fit(self, X, y):
        temp_df = pd.DataFrame(X).copy()
        temp_df['target'] = y
        for col in temp_df.columns:
            if col != 'target':
                ordered_labels = temp_df.groupby([col])['target'].mean().sort_values().index
                self.mappings[col] = {k: i for i, k in enumerate(ordered_labels, 0)}
        return self

    def transform(self, X):
        X_copy = pd.DataFrame(X).copy()
        for col, mapping in self.mappings.items():
            X_copy[col] = X_copy[col].map(mapping).fillna(-1)
        return X_copy

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    selected_features_path = os.path.join('artifacts', "selected_features.npy")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.config = read_config()

    # --- MÉTHODE DE NETTOYAGE (DOUBLONS & MANQUANTS) ---
    def clean_data(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Nettoie les données en préservant la colonne cible
        
        Args:
            df: DataFrame à nettoyer
            target_column: Nom de la colonne cible à préserver (optionnel)
        """
        try:
            logging.info("Nettoyage : Doublons, Valeurs manquantes et Corrélation")
            
            # 1. Suppression des doublons
            df = df.drop_duplicates()

            # 2. Suppression colonnes > 30% vides
            threshold_na = 0.3
            df = df.loc[:, df.isnull().mean() <= threshold_na]

            # 3. Imputation (Médiane/Mode)
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna(df[col].mode()[0])
                    else:
                        df[col] = df[col].fillna(df[col].median())

            # 4. Suppression des variables fortement corrélées
            df_numeric = df.select_dtypes(exclude='object')
            
            # Exclure la colonne cible de l'analyse de corrélation
            if target_column and target_column in df_numeric.columns:
                df_numeric_for_corr = df_numeric.drop(columns=[target_column])
            else:
                df_numeric_for_corr = df_numeric
            
            if len(df_numeric_for_corr.columns) > 1:
                corr_matrix = df_numeric_for_corr.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
                
                if to_drop:
                    logging.info(f"Colonnes supprimées car trop corrélées (>0.85) : {to_drop}")
                    df = df.drop(columns=to_drop)

            return df
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, cat_features, num_features):
        try:
            logging.info("Initialisation du ColumnTransformer")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("TargetOrdinal", TargetGuidedOrdinalEncoder(), cat_features),
                    ("StandardScaler", StandardScaler(), num_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Nettoyer les noms de colonnes
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()
            
            logging.info(f"Colonnes trouvées dans le CSV : {train_df.columns.tolist()}")
            
            # Récupérer le nom de la colonne cible
            target_column_name = self.config['data_transformation']['target_column']
            
            # Vérifier si la colonne cible existe
            if target_column_name not in train_df.columns:
                available_cols = train_df.columns.tolist()
                logging.error(f"Colonne cible '{target_column_name}' non trouvée. Colonnes disponibles : {available_cols}")
                raise KeyError(f"La colonne cible '{target_column_name}' n'existe pas. Colonnes disponibles : {available_cols}")

            # Nettoyage en préservant la colonne cible
            train_df = self.clean_data(train_df, target_column=target_column_name)
            test_df = self.clean_data(test_df, target_column=target_column_name)
            
            # Vérifier après nettoyage
            if target_column_name not in train_df.columns:
                raise KeyError(f"La colonne cible '{target_column_name}' a été supprimée pendant le nettoyage.")

            # Séparation X et y
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Détection automatique des colonnes
            num_features = input_feature_train_df.select_dtypes(exclude="object").columns.tolist()
            cat_features = input_feature_train_df.select_dtypes(include="object").columns.tolist()

            logging.info(f"Features numériques : {num_features}")
            logging.info(f"Features catégorielles : {cat_features}")

            # Créer le preprocessor
            preprocessing_obj = self.get_data_transformer_object(cat_features, num_features)

            logging.info("Application des transformations")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Lasso Selection avec paramètres optimisés pour la convergence
            all_cols = cat_features + num_features
            X_train_df = pd.DataFrame(input_feature_train_arr, columns=all_cols)
            
            # Paramètres optimisés : alpha plus élevé, max_iter élevé, tolérance ajustée
            selector = SelectFromModel(
                Lasso(alpha=0.05, random_state=42, max_iter=100000, tol=0.01),
                threshold='median'  # Sélectionne les features au-dessus de la médiane
            )
            selector.fit(X_train_df, target_feature_train_df)
            
            # Transformer et garder les noms de features
            input_feature_train_arr = selector.transform(X_train_df)
            input_feature_test_arr = selector.transform(pd.DataFrame(input_feature_test_arr, columns=all_cols))

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Transformation des données terminée avec succès")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)