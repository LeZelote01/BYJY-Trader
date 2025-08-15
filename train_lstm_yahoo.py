#!/usr/bin/env python3
"""
🧠 PHASE 2.2 - ENTRAÎNEMENT LSTM AVEC YAHOO FINANCE
Entraînement LSTM avec données directes de Yahoo Finance
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import yfinance as yf
from datetime import datetime
from pathlib import Path

print("🚀 PHASE 2.2 - ENTRAÎNEMENT LSTM AVEC YAHOO FINANCE")
print("=" * 60)

# Configuration selon roadmap
SYMBOL = "AAPL"
SEQUENCE_LENGTH = 60
EPOCHS = 25  # Test rapide
BATCH_SIZE = 32
TEST_SIZE = 0.2

# Paramètres modèle selon PHASE2_2_SPEC_AI_MODELS.md
LSTM_UNITS = [128, 64, 32]
DROPOUT = 0.3
LEARNING_RATE = 0.001

def load_yahoo_data():
    """Charge les données depuis Yahoo Finance"""
    print(f"📊 Chargement données {SYMBOL} depuis Yahoo Finance...")
    
    try:
        # Téléchargement 1 an de données (6+ mois comme demandé)
        ticker = yf.Ticker(SYMBOL)
        df = ticker.history(period="1y")
        
        if len(df) == 0:
            print(f"❌ Aucune donnée trouvée pour {SYMBOL}")
            return None
            
        print(f"✅ {len(df)} points de données chargés")
        print(f"📅 Période: {df.index.min().date()} → {df.index.max().date()}")
        print(f"📊 Prix: ${df['Close'].min():.2f} → ${df['Close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return None

def create_sequences(data, sequence_length):
    """Crée les séquences temporelles pour LSTM"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm_model():
    """Entraîne le modèle LSTM basique"""
    
    # 1. Chargement des données
    df = load_yahoo_data()
    if df is None:
        return False
        
    # 2. Préparation des données
    print("\n🔧 Préparation données LSTM...")
    
    # Utiliser les prix de clôture
    prices = df['Close'].values
    print(f"📈 {len(prices)} prix de clôture récupérés")
    
    # Normalisation MinMaxScaler [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    print(f"✅ Normalisation: [{scaled_prices.min():.3f}, {scaled_prices.max():.3f}]")
    
    # Création séquences temporelles
    X, y = create_sequences(scaled_prices, SEQUENCE_LENGTH)
    print(f"📊 Séquences créées: {X.shape[0]} séquences de {SEQUENCE_LENGTH} jours")
    
    # Division train/test
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Reshape pour LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"✅ Division: Train {X_train.shape}, Test {X_test.shape}")
    
    # 3. Création du modèle LSTM
    print("\n🧠 Construction modèle LSTM...")
    
    model = Sequential([
        # Première couche LSTM avec return_sequences=True
        LSTM(LSTM_UNITS[0], return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        Dropout(DROPOUT),
        
        # Deuxième couche LSTM
        LSTM(LSTM_UNITS[1], return_sequences=True),
        Dropout(DROPOUT),
        
        # Troisième couche LSTM  
        LSTM(LSTM_UNITS[2], return_sequences=False),
        Dropout(DROPOUT),
        
        # Couche Dense de sortie
        Dense(1)
    ])
    
    # Compilation avec optimiseur Adam
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"✅ Architecture: {LSTM_UNITS} → 1")
    print(f"✅ Paramètres: {model.count_params():,}")
    
    # 4. Entraînement du modèle
    print("\n🚀 Démarrage entraînement...")
    print(f"⚙️  Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    
    start_time = datetime.now()
    
    # Callbacks pour optimiser l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.now() - start_time
    print(f"\n⏱️  Entraînement terminé en {training_time}")
    
    # 5. Évaluation du modèle
    print("\n📊 Évaluation performance...")
    
    # Prédictions sur test set
    predictions = model.predict(X_test, verbose=0)
    
    # Dénormalisation pour métriques réelles
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_actual = scaler.inverse_transform(predictions).flatten()
    
    # Calcul métriques
    mse = mean_squared_error(y_test_actual, predictions_actual)
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    
    # Calcul accuracy directionnelle (objectif Phase 2.2: > 60%)
    y_diff = np.diff(y_test_actual)
    pred_diff = np.diff(predictions_actual)
    directional_accuracy = np.mean(np.sign(y_diff) == np.sign(pred_diff)) * 100
    
    # Calcul RMSE relatif (objectif < 5%)
    price_std = y_test_actual.std()
    rmse_relative = (rmse / price_std) * 100
    
    print("\n📈 RÉSULTATS ÉVALUATION:")
    print(f"   💰 RMSE: ${rmse:.2f} ({rmse_relative:.1f}% de la volatilité)")
    print(f"   📊 MAE: ${mae:.2f}")
    print(f"   🎯 Accuracy Directionnelle: {directional_accuracy:.1f}%")
    print(f"   📉 Loss final: {history.history['loss'][-1]:.6f}")
    
    # Validation critères Phase 2.2 selon roadmap
    accuracy_target = 60.0  # > 60% selon PHASE2_2_SPEC
    rmse_target = 5.0       # < 5% selon PHASE2_2_SPEC
    
    accuracy_met = directional_accuracy > accuracy_target
    rmse_met = rmse_relative < rmse_target
    overall_success = accuracy_met and rmse_met
    
    print(f"\n🎯 VALIDATION OBJECTIFS PHASE 2.2:")
    print(f"   📈 Accuracy > {accuracy_target}%: {'✅' if accuracy_met else '❌'} ({directional_accuracy:.1f}%)")
    print(f"   📊 RMSE < {rmse_target}%: {'✅' if rmse_met else '❌'} ({rmse_relative:.1f}%)")
    print(f"   🎉 Objectifs globaux: {'✅ ATTEINTS' if overall_success else '❌ PARTIELS'}")
    
    # 6. Sauvegarde du modèle
    print("\n💾 Sauvegarde du modèle...")
    
    # Création dossier modèles
    models_dir = Path('/app/ai/trained_models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde modèle TensorFlow
    model_path = models_dir / f'lstm_{SYMBOL.lower()}_phase2.h5'
    model.save(model_path)
    
    # Sauvegarde scaler
    scaler_path = models_dir / f'scaler_{SYMBOL.lower()}_phase2.pkl'
    joblib.dump(scaler, scaler_path)
    
    # Sauvegarde métriques et métadonnées
    metadata = {
        'symbol': SYMBOL,
        'sequence_length': SEQUENCE_LENGTH,
        'training_date': datetime.now().isoformat(),
        'epochs_trained': len(history.history['loss']),
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'rmse_relative_percent': float(rmse_relative),
            'directional_accuracy_percent': float(directional_accuracy),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        },
        'objectives_met': {
            'accuracy_target': accuracy_target,
            'accuracy_achieved': float(directional_accuracy),
            'accuracy_met': accuracy_met,
            'rmse_target': rmse_target,
            'rmse_achieved': float(rmse_relative),
            'rmse_met': rmse_met,
            'overall_success': overall_success
        }
    }
    
    # Sauvegarde JSON
    import json
    metadata_path = models_dir / f'metadata_{SYMBOL.lower()}_phase2.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Modèle: {model_path}")
    print(f"✅ Scaler: {scaler_path}")
    print(f"✅ Métadonnées: {metadata_path}")
    
    # 7. Résumé final
    print("\n" + "="*60)
    print("🎉 PHASE 2.2 - ENTRAÎNEMENT LSTM TERMINÉ")
    print("="*60)
    
    if overall_success:
        print("🏆 SUCCÈS COMPLET: Tous les objectifs Phase 2.2 atteints!")
        print("🚀 Prêt pour Phase 2.3: Tests prédictions temps réel")
        print("📈 Le modèle peut être utilisé pour trading simulation")
    elif accuracy_met:
        print("✅ SUCCÈS PARTIEL: Accuracy atteinte, RMSE à améliorer")
        print("💡 Recommandation: Plus de données ou fine-tuning hyperparamètres")  
    else:
        print("⚠️  OBJECTIFS PARTIELLEMENT ATTEINTS")
        print("💡 Le modèle de base est fonctionnel pour tests")
        print("🔧 Recommandations: Étendre dataset ou ajuster architecture")
    
    print(f"\n📊 Modèle final: {directional_accuracy:.1f}% accuracy, {rmse_relative:.1f}% RMSE")
    print("🎯 Prochaine étape selon Roadmap: Configuration modèles ensemble")
    
    return overall_success

if __name__ == "__main__":
    success = train_lstm_model()
    print(f"\n🔚 Statut final: {'✅ SUCCESS' if success else '⚠️ PARTIAL'}")