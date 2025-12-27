"""
Script de diagnostic pour tester la convergence du Lasso
et trouver les meilleurs paramètres
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt

def test_lasso_convergence(X, y, alphas=None, max_iters=None):
    """
    Teste différentes combinaisons de alpha et max_iter pour Lasso
    
    Args:
        X: Features
        y: Target
        alphas: Liste de valeurs alpha à tester
        max_iters: Liste de max_iter à tester
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    if max_iters is None:
        max_iters = [1000, 10000, 50000, 100000]
    
    results = []
    
    print("=" * 80)
    print("TEST DE CONVERGENCE DU LASSO")
    print("=" * 80)
    
    for alpha in alphas:
        for max_iter in max_iters:
            # Supprimer temporairement les warnings pour ce test
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
                lasso.fit(X, y)
                
                # Vérifier si un warning de convergence a été émis
                convergence_warning = any("Objective did not converge" in str(warning.message) for warning in w)
                
                # Nombre de features sélectionnées
                n_features_selected = np.sum(lasso.coef_ != 0)
                
                results.append({
                    'alpha': alpha,
                    'max_iter': max_iter,
                    'converged': not convergence_warning,
                    'n_features': n_features_selected,
                    'n_iter': lasso.n_iter_
                })
                
                status = "✓ CONVERGÉ" if not convergence_warning else "✗ NON CONVERGÉ"
                print(f"Alpha: {alpha:6.3f} | Max_iter: {max_iter:6d} | {status} | "
                      f"Features: {n_features_selected:3d} | Iterations: {lasso.n_iter_:5d}")
    
    print("=" * 80)
    
    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    
    # Recommandations
    print("\n RECOMMANDATIONS :")
    print("-" * 80)
    
    converged = df_results[df_results['converged'] == True]
    if len(converged) > 0:
        print(f"✓ {len(converged)}/{len(df_results)} combinaisons ont convergé")
        
        # Meilleure combinaison (convergé + bon nombre de features)
        best = converged.sort_values('n_features', ascending=False).iloc[0]
        print(f"\n MEILLEURE COMBINAISON :")
        print(f"   Alpha: {best['alpha']}")
        print(f"   Max_iter: {best['max_iter']}")
        print(f"   Features sélectionnées: {best['n_features']}")
        print(f"   Iterations utilisées: {best['n_iter']}")
        
        # Combinaison la plus rapide qui converge
        fastest = converged.sort_values('n_iter').iloc[0]
        print(f"\n⚡ CONVERGENCE LA PLUS RAPIDE :")
        print(f"   Alpha: {fastest['alpha']}")
        print(f"   Max_iter: {fastest['max_iter']}")
        print(f"   Iterations: {fastest['n_iter']}")
    else:
        print("✗ Aucune combinaison n'a convergé. Recommandations :")
        print("  1. Augmenter max_iter (essayer 200000 ou 500000)")
        print("  2. Augmenter alpha (essayer des valeurs > 10)")
        print("  3. Normaliser/standardiser vos features")
        print("  4. Utiliser Ridge au lieu de Lasso")
    
    print("=" * 80)
    
    return df_results

def check_data_quality(X, y):
    """Vérifie la qualité des données (facteurs affectant la convergence)"""
    print("\n" + "=" * 80)
    print("ANALYSE DE LA QUALITÉ DES DONNÉES")
    print("=" * 80)
    
    # Vérifier les valeurs manquantes
    n_missing = np.sum(np.isnan(X))
    print(f"Valeurs manquantes: {n_missing}")
    
    # Vérifier les valeurs infinies
    n_inf = np.sum(np.isinf(X))
    print(f"Valeurs infinies: {n_inf}")
    
    # Statistiques sur les features
    print(f"\nStatistiques des features:")
    print(f"  Moyenne min: {np.min(np.mean(X, axis=0)):.2f}")
    print(f"  Moyenne max: {np.max(np.mean(X, axis=0)):.2f}")
    print(f"  Std min: {np.min(np.std(X, axis=0)):.6f}")
    print(f"  Std max: {np.max(np.std(X, axis=0)):.2f}")
    
    # Vérifier l'échelle des données
    feature_ranges = np.ptp(X, axis=0)  # ptp = peak to peak (max - min)
    print(f"\nÉchelle des features (max - min):")
    print(f"  Min: {np.min(feature_ranges):.6f}")
    print(f"  Max: {np.max(feature_ranges):.2f}")
    print(f"  Ratio: {np.max(feature_ranges) / max(np.min(feature_ranges), 1e-10):.2f}")
    
    if np.max(feature_ranges) / max(np.min(feature_ranges), 1e-10) > 1000:
        print("\n  ATTENTION: Grande disparité d'échelle détectée!")
        print("   Recommandation: Utiliser StandardScaler avant Lasso")
    
    print("=" * 80)

if __name__ == "__main__":
    # Exemple d'utilisation
    print("Pour utiliser ce script avec vos données:")
    print("\n1. Importez vos données transformées")
    print("2. Appelez: test_lasso_convergence(X_train, y_train)")
    print("3. Utilisez les paramètres recommandés dans votre pipeline")
    print("\nExemple:")
    print("-" * 80)
    print("""
from check_convergence import test_lasso_convergence, check_data_quality
import pandas as pd

# Charger vos données
train_df = pd.read_csv('artifacts/train.csv')
X = train_df.drop('target', axis=1)
y = train_df['target']

# Vérifier la qualité des données
check_data_quality(X.values, y.values)

# Tester la convergence
results = test_lasso_convergence(X.values, y.values)
    """)