#!/usr/bin/env python3
"""
🧠 PHASE 2.2 - ENTRAÎNEMENT LSTM SIMPLE
Entraînement direct avec accès SQLite
"""

import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime
from pathlib import Path

print("🚀 PHASE 2.2 - ENTRAÎNEMENT LSTM SIMPLE")
print("=" * 50)

# Configuration
SYMBOL = "AAPL"
SEQUENCE_LENGTH = 60
EPOCHS = 30  # Réduit pour test rapide
BATCH_SIZE = 32
TEST_SIZE = 0.2

# Paramètres modèle
LSTM_UNITS = [128, 64, 32]
DROPOUT = 0.3
LEARNING_RATE = 0.001

def load_data_from_db():
    """Charge les données directement depuis SQLite"""
    print(f"📊 Chargement données {SYMBOL} depuis SQLite...")
    
    try:
        conn = sqlite3.connect('/app/database/byjy_trader.db')
        
        # Requête pour récupérer les données
        query = """
        SELECT timestamp, open, high, low, close, volume 
        FROM historical_data 
        WHERE symbol = ? 
        ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=[SYMBOL])
        conn.close()
        
        if len(df) == 0:
            print(f"❌ Aucune donnée trouvée pour {SYMBOL}")
            return None
            
        # Conversion timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        print(f"✅ {len(df)} points de données chargés")
        print(f"📅 Période: {df.index.min()} → {df.index.max()}")
        print(f"📊 Prix min: ${df['close'].min():.2f}, max: ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return None

def create_sequences(data, sequence_length):
    """Crée les séquences pour LSTM"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm_model():
    """Entraîne le modèle LSTM"""
    
    # 1. Chargement des données
    df = load_data_from_db()
    if df is None:
        return False
        
    # 2. Préparation des données
    print("🔧 Préparation données...")
    
    # Utiliser les prix de clôture
    prices = df['close'].values
    
    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # Création séquences
    X, y = create_sequences(scaled_prices, SEQUENCE_LENGTH)
    
    # Division train/test
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Reshape pour LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"✅ Données préparées: Train {X_train.shape}, Test {X_test.shape}")
    
    # 3. Création du modèle
    print("🧠 Création modèle LSTM...")
    
    model = Sequential([
        LSTM(LSTM_UNITS[0], return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        Dropout(DROPOUT),
        LSTM(LSTM_UNITS[1], return_sequences=True),
        Dropout(DROPOUT),
        LSTM(LSTM_UNITS[2], return_sequences=False),
        Dropout(DROPOUT),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"✅ Modèle créé: {LSTM_UNITS}")
    
    # 4. Entraînement
    print("🚀 Entraînement...")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # 5. Évaluation
    print("📊 Évaluation...")
    
    predictions = model.predict(X_test)
    
    # Dénormalisation
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_actual = scaler.inverse_transform(predictions)
    
    # Métriques
    mse = mean_squared_error(y_test_actual, predictions_actual)
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    
    # Accuracy directionnelle
    y_diff = np.diff(y_test_actual.flatten())
    pred_diff = np.diff(predictions_actual.flatten())
    directional_accuracy = np.mean(np.sign(y_diff) == np.sign(pred_diff)) * 100
    
    print("\n📈 RÉSULTATS:")
    print(f"   - RMSE: ${rmse:.2f}")
    print(f"   - MAE: ${mae:.2f}")
    print(f"   - Accuracy Directionnelle: {directional_accuracy:.1f}%")
    
    # Validation critères
    target_accuracy = 60.0
    success = directional_accuracy > target_accuracy
    
    print("\n🎯 VALIDATION PHASE 2.2:")
    print(f"   - Accuracy > 60%: {'✅' if success else '❌'} ({directional_accuracy:.1f}%)")
    
    # 6. Sauvegarde
    if success:
        print("💾 Sauvegarde du modèle...")
        
        models_dir = Path('/app/ai/trained_models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde modèle
        model_path = models_dir / f'lstm_{SYMBOL.lower()}_basic.h5'
        model.save(model_path)
        
        # Sauvegarde scaler
        scaler_path = models_dir / f'scaler_{SYMBOL.lower()}_basic.pkl'
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
    
    print("\n🎉 PHASE 2.2 - TERMINÉE")
    if success:
        print("✅ SUCCÈS: Objectifs atteints!")
        print("🚀 Prêt pour Phase 2.3: Tests prédictions temps réel")
    else:
        print("⚠️  Objectifs partiellement atteints - Modèle de base fonctionnel")
    
    return success

if __name__ == "__main__":
    train_lstm_model()