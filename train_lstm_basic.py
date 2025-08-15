#!/usr/bin/env python3
"""
🧠 PHASE 2.2 - ENTRAÎNEMENT MODÈLE LSTM BASIQUE
Entraînement d'un modèle LSTM sur données AAPL selon roadmap
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configuration du chemin
sys.path.insert(0, '/app')

# Importation des modules BYJY-Trader  
from core.config import get_settings
from core.database import get_database_manager
from data.storage.data_manager import DataManager

# Imports IA/ML
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

print("🚀 PHASE 2.2 - ENTRAÎNEMENT LSTM BASIQUE")
print("=" * 50)

# Configuration
settings = get_settings()
db_manager = get_database_manager()
data_manager = DataManager()

# Paramètres LSTM selon spécifications
SYMBOL = "AAPL"
SEQUENCE_LENGTH = 60  # 60 jours selon roadmap
EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Paramètres modèle selon PHASE2_2_SPEC_AI_MODELS.md
LSTM_UNITS = [128, 64, 32]  # 3 couches LSTM
DROPOUT = 0.3
LEARNING_RATE = 0.001

async def load_training_data():
    """Charge les données d'entraînement depuis la base"""
    print(f"📊 Chargement données {SYMBOL}...")
    
    try:
        # Récupération des données via DataManager
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 12 mois pour l'entraînement
        
        data = await data_manager.get_historical_data(
            symbol=SYMBOL,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        
        if data is None or len(data) == 0:
            print(f"❌ Aucune donnée trouvée pour {SYMBOL}")
            return None
            
        # Conversion en DataFrame
        df = pd.DataFrame(data)
        print(f"✅ {len(df)} points de données chargés")
        print(f"📅 Période: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return None

def prepare_lstm_data(df, target_column='close'):
    """Prépare les données pour l'entraînement LSTM"""
    print(f"🔧 Préparation données LSTM...")
    
    # Utiliser uniquement les prix de clôture
    data = df[target_column].values.reshape(-1, 1)
    
    # Normalisation avec MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Création des séquences
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQUENCE_LENGTH:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Division train/test
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Reshape pour LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"✅ Données préparées:")
    print(f"   - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   - Séquences: {SEQUENCE_LENGTH} jours")
    print(f"   - Normalisation: MinMaxScaler [0,1]")
    
    return X_train, X_test, y_train, y_test, scaler

def create_lstm_model():
    """Crée le modèle LSTM selon spécifications"""
    print("🧠 Création modèle LSTM...")
    
    model = Sequential([
        # Première couche LSTM
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
    
    print("✅ Modèle LSTM créé:")
    print(f"   - Architecture: {LSTM_UNITS[0]}→{LSTM_UNITS[1]}→{LSTM_UNITS[2]}→1")
    print(f"   - Dropout: {DROPOUT}")
    print(f"   - Optimiseur: Adam (lr={LEARNING_RATE})")
    print(f"   - Loss: MSE")
    
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test):
    """Entraîne le modèle LSTM"""
    print("🚀 Entraînement du modèle LSTM...")
    
    # Callbacks pour l'entraînement
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Entraînement
    start_time = datetime.now()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = datetime.now() - start_time
    print(f"✅ Entraînement terminé en {training_time}")
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    """Évalue la performance du modèle"""
    print("📊 Évaluation du modèle...")
    
    # Prédictions
    predictions = model.predict(X_test)
    
    # Dénormalisation
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_actual = scaler.inverse_transform(predictions)
    
    # Métriques
    mse = mean_squared_error(y_test_actual, predictions_actual)
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    
    # Calcul accuracy directionnelle (Phase 2.2 target: >60%)
    y_diff_actual = np.diff(y_test_actual.flatten())
    pred_diff = np.diff(predictions_actual.flatten())
    directional_accuracy = np.mean(np.sign(y_diff_actual) == np.sign(pred_diff)) * 100
    
    print("📈 RÉSULTATS ÉVALUATION:")
    print(f"   - RMSE: ${rmse:.2f}")
    print(f"   - MAE: ${mae:.2f}")
    print(f"   - MSE: {mse:.2f}")
    print(f"   - Accuracy Directionnelle: {directional_accuracy:.1f}%")
    
    # Validation critères Phase 2.2
    target_accuracy = 60.0  # Selon roadmap
    rmse_target = y_test_actual.std() * 0.05  # <5% écart selon spec
    
    print("\n🎯 VALIDATION CRITÈRES PHASE 2.2:")
    print(f"   - Accuracy > 60%: {'✅' if directional_accuracy > target_accuracy else '❌'} ({directional_accuracy:.1f}%)")
    print(f"   - RMSE < 5% std: {'✅' if rmse < rmse_target else '❌'} ({rmse:.2f} vs {rmse_target:.2f})")
    
    return {
        'rmse': rmse,
        'mae': mae, 
        'mse': mse,
        'directional_accuracy': directional_accuracy,
        'target_met': directional_accuracy > target_accuracy and rmse < rmse_target
    }

def save_model(model, scaler, metrics):
    """Sauvegarde le modèle et les métriques"""
    print("💾 Sauvegarde du modèle...")
    
    # Créer le dossier models s'il n'existe pas
    models_dir = Path('/app/ai/trained_models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde modèle
    model_path = models_dir / f'lstm_{SYMBOL.lower()}_basic.h5'
    model.save(model_path)
    
    # Sauvegarde scaler
    scaler_path = models_dir / f'scaler_{SYMBOL.lower()}_basic.pkl'
    joblib.dump(scaler, scaler_path)
    
    # Sauvegarde métriques
    metrics_path = models_dir / f'metrics_{SYMBOL.lower()}_basic.json'
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            **metrics,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'training_date': datetime.now().isoformat(),
            'symbol': SYMBOL,
            'sequence_length': SEQUENCE_LENGTH
        }, f, indent=2)
    
    print(f"✅ Modèle sauvegardé:")
    print(f"   - Modèle: {model_path}")
    print(f"   - Scaler: {scaler_path}")
    print(f"   - Métriques: {metrics_path}")

async def main():
    """Fonction principale d'entraînement LSTM"""
    try:
        print(f"🎯 OBJECTIF PHASE 2.2: Accuracy > 60%, RMSE < 5%")
        print(f"📊 Symbol: {SYMBOL}, Séquence: {SEQUENCE_LENGTH}, Epochs: {EPOCHS}")
        
        # 1. Chargement des données
        df = await load_training_data()
        if df is None:
            return False
            
        # 2. Préparation des données
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df)
        
        # 3. Création du modèle
        model = create_lstm_model()
        
        # 4. Entraînement
        model, history = train_lstm_model(model, X_train, y_train, X_test, y_test)
        
        # 5. Évaluation
        metrics = evaluate_model(model, X_test, y_test, scaler)
        
        # 6. Sauvegarde
        save_model(model, scaler, metrics)
        
        # 7. Résumé final
        print("\n🎉 PHASE 2.2 - ENTRAÎNEMENT LSTM TERMINÉ")
        print("=" * 50)
        if metrics['target_met']:
            print("✅ SUCCÈS: Objectifs Phase 2.2 atteints!")
            print("🚀 Prêt pour Phase 2.3: Modèles Ensemble")
        else:
            print("⚠️  Objectifs partiellement atteints")
            print("💡 Recommandation: Ajuster hyperparamètres ou étendre dataset")
            
        return metrics['target_met']
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())