"""
Advanced Phishing Detection with Character-Level LSTM
This module extends the basic detector with sequential deep learning models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, LSTM, Embedding, Conv1D, 
                                      MaxPooling1D, Bidirectional, Input, 
                                      Concatenate, GlobalMaxPooling1D, Flatten)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from phishing_detector import URLFeatureExtractor


class AdvancedPhishingDetector:
    """Advanced phishing detector with sequential models"""
    
    def __init__(self, max_url_length=200):
        self.max_url_length = max_url_length
        self.tokenizer = Tokenizer(char_level=True)
        self.feature_extractor = URLFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def prepare_sequential_data(self, urls, labels, test_size=0.2):
        """Prepare data for sequential models (LSTM, CNN)"""
        
        # Tokenize URLs at character level
        self.tokenizer.fit_on_texts(urls)
        sequences = self.tokenizer.texts_to_sequences(urls)
        X_seq = pad_sequences(sequences, maxlen=self.max_url_length, padding='post')
        
        # Also extract handcrafted features
        X_features = self.feature_extractor.extract_batch_features(urls)
        X_features_scaled = self.scaler.fit_transform(X_features)
        
        # Split data
        indices = np.arange(len(urls))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=labels
        )
        
        return {
            'X_seq_train': X_seq[train_idx],
            'X_seq_test': X_seq[test_idx],
            'X_feat_train': X_features_scaled[train_idx],
            'X_feat_test': X_features_scaled[test_idx],
            'y_train': labels[train_idx],
            'y_test': labels[test_idx]
        }
    
    def build_lstm_model(self, vocab_size):
        """Build LSTM model for character-level URL analysis"""
        model = Sequential([
            Embedding(vocab_size, 128, input_length=self.max_url_length),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_bidirectional_lstm(self, vocab_size):
        """Build Bidirectional LSTM model"""
        model = Sequential([
            Embedding(vocab_size, 128, input_length=self.max_url_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_cnn_model(self, vocab_size):
        """Build CNN model for URL pattern detection"""
        model = Sequential([
            Embedding(vocab_size, 128, input_length=self.max_url_length),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 5, activation='relu'),
            MaxPooling1D(2),
            Conv1D(32, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def build_hybrid_model(self, vocab_size, num_features):
        """Build hybrid model combining sequential and handcrafted features"""
        
        # Sequential input branch
        seq_input = Input(shape=(self.max_url_length,))
        x1 = Embedding(vocab_size, 128)(seq_input)
        x1 = Bidirectional(LSTM(64))(x1)
        x1 = Dropout(0.3)(x1)
        
        # Feature input branch
        feat_input = Input(shape=(num_features,))
        x2 = Dense(32, activation='relu')(feat_input)
        x2 = Dropout(0.2)(x2)
        
        # Combine branches
        combined = Concatenate()([x1, x2])
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=[seq_input, feat_input], outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                   model_name='model', epochs=50, batch_size=128):
        """Train a model with callbacks"""
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        # Prepare validation data
        if X_val is None:
            validation_split = 0.2
            validation_data = None
        else:
            validation_split = 0.0
            validation_data = (X_val, y_val)
        
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and store results"""
        
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        self.results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n{model_name} Results:")
        for metric, value in self.results[model_name].items():
            print(f"  {metric}: {value:.4f}")
        
        self.models[model_name] = model
    
    def run_all_experiments(self, urls, labels):
        """Run all model experiments"""
        
        print("Preparing data...")
        data = self.prepare_sequential_data(urls, labels)
        vocab_size = len(self.tokenizer.word_index) + 1
        num_features = data['X_feat_train'].shape[1]
        
        print(f"\nVocabulary size: {vocab_size}")
        print(f"Number of handcrafted features: {num_features}")
        print(f"Training samples: {len(data['y_train'])}")
        print(f"Testing samples: {len(data['y_test'])}")
        
        # 1. LSTM Model
        print("\n" + "="*80)
        print("Training LSTM Model")
        print("="*80)
        lstm_model = self.build_lstm_model(vocab_size)
        lstm_model, _ = self.train_model(
            lstm_model, 
            data['X_seq_train'], 
            data['y_train'],
            model_name='LSTM'
        )
        self.evaluate_model(lstm_model, data['X_seq_test'], data['y_test'], 'LSTM')
        
        # 2. Bidirectional LSTM
        print("\n" + "="*80)
        print("Training Bidirectional LSTM Model")
        print("="*80)
        bilstm_model = self.build_bidirectional_lstm(vocab_size)
        bilstm_model, _ = self.train_model(
            bilstm_model,
            data['X_seq_train'],
            data['y_train'],
            model_name='BiLSTM'
        )
        self.evaluate_model(bilstm_model, data['X_seq_test'], data['y_test'], 'BiLSTM')
        
        # 3. CNN Model
        print("\n" + "="*80)
        print("Training CNN Model")
        print("="*80)
        cnn_model = self.build_cnn_model(vocab_size)
        cnn_model, _ = self.train_model(
            cnn_model,
            data['X_seq_train'],
            data['y_train'],
            model_name='CNN'
        )
        self.evaluate_model(cnn_model, data['X_seq_test'], data['y_test'], 'CNN')
        
        # 4. Hybrid Model
        print("\n" + "="*80)
        print("Training Hybrid Model (BiLSTM + Features)")
        print("="*80)
        hybrid_model = self.build_hybrid_model(vocab_size, num_features)
        hybrid_model, _ = self.train_model(
            hybrid_model,
            [data['X_seq_train'], data['X_feat_train']],
            data['y_train'],
            model_name='Hybrid'
        )
        self.evaluate_model(
            hybrid_model,
            [data['X_seq_test'], data['X_feat_test']],
            data['y_test'],
            'Hybrid'
        )
        
        return self.results
    
    def print_comparison(self):
        """Print comprehensive comparison of all models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON - ADVANCED DEEP LEARNING")
        print("="*80)
        
        df = pd.DataFrame(self.results).T
        print("\n", df.round(4))
        
        # Find best model
        best_model = df['f1_score'].idxmax()
        print(f"\nBest Model: {best_model}")
        print(f"F1-Score: {df.loc[best_model, 'f1_score']:.4f}")
        
        return df


def main():
    """Main execution for advanced models"""
    
    # Load your data
    print("Loading data...")
    df = pd.read_csv('phishing_dataset.csv')  # Update path
    
    urls = df['url'].values
    labels = df['label'].values
    
    # Convert labels if needed
    if labels.dtype == object:
        labels = (labels == 'phishing').astype(int)
    
    # Initialize detector
    detector = AdvancedPhishingDetector(max_url_length=200)
    
    # Run all experiments
    results = detector.run_all_experiments(urls, labels)
    
    # Print comparison
    results_df = detector.print_comparison()
    
    # Save results
    results_df.to_csv('advanced_model_results.csv')
    print("\nResults saved to 'advanced_model_results.csv'")


if __name__ == "__main__":
    main()