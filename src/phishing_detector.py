"""
CYSE 689 Midterm Project: Phishing URL Detection - OPTIMIZED FOR SPEED
This version uses faster algorithms without compromising accuracy
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_words = ['secure', 'account', 'update', 'login', 'verify', 
                                 'banking', 'confirm', 'password', 'click', 'signin']
    
    def extract_features(self, url):
        """Extract comprehensive features from a URL"""
        features = {}
        
        try:
            # Basic URL features
            features['url_length'] = len(url)
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_questions'] = url.count('?')
            features['num_equals'] = url.count('=')
            features['num_ats'] = url.count('@')
            features['num_ampersands'] = url.count('&')
            features['num_percent'] = url.count('%')
            features['num_digits'] = sum(c.isdigit() for c in url)
            
            # Parse URL
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            
            # Domain features
            features['domain_length'] = len(domain)
            features['has_ip'] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', domain)))
            features['num_subdomains'] = domain.count('.') - 1 if domain.count('.') > 0 else 0
            
            # Protocol features
            features['is_https'] = int(url.startswith('https'))
            features['has_port'] = int(':' in domain and not domain.startswith('['))
            
            # Path features
            features['path_length'] = len(path)
            features['num_path_tokens'] = len(path.split('/'))
            
            # Suspicious patterns
            features['has_suspicious_words'] = int(any(word in url.lower() for word in self.suspicious_words))
            features['has_double_slash'] = int('//' in path)
            features['has_prefix_suffix'] = int('-' in domain)
            
            # Character ratios
            features['digit_ratio'] = features['num_digits'] / features['url_length'] if features['url_length'] > 0 else 0
            features['special_char_ratio'] = (features['num_dots'] + features['num_hyphens'] + 
                                             features['num_underscores']) / features['url_length'] if features['url_length'] > 0 else 0
            
            # Entropy (randomness measure)
            features['entropy'] = self._calculate_entropy(url)
            
        except Exception as e:
            # Return default features on error
            features = {f'feature_{i}': 0 for i in range(24)}
        
        return features
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        entropy = 0
        for x in range(256):
            p_x = float(text.count(chr(x))) / len(text)
            if p_x > 0:
                entropy += - p_x * np.log2(p_x)
        return entropy
    
    def extract_batch_features(self, urls):
        """Extract features from a list of URLs"""
        print(f"  Extracting features from {len(urls)} URLs...")
        features_list = [self.extract_features(url) for url in urls]
        return pd.DataFrame(features_list)


class PhishingDetector:
    """Main class for phishing detection - OPTIMIZED VERSION"""
    
    def __init__(self):
        self.feature_extractor = URLFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}
        self.history = {}
        self.results = {}
        
    def load_data(self, filepath):
        """Load the phishing dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {len(df)} records")
        return df
    
    def preprocess_data(self, df, url_column='url', label_column='label', test_size=0.2):
        """Preprocess the dataset and extract features"""
        print("\nPreprocessing data...")
        
        # Extract features
        X = self.feature_extractor.extract_batch_features(df[url_column].values)
        y = df[label_column].values
        
        # Ensure binary labels
        if y.dtype == object or len(np.unique(y)) != 2:
            unique_vals = np.unique(y)
            if len(unique_vals) != 2:
                raise ValueError(f"Expected 2 classes, got {len(unique_vals)}")
            # Map to 0 and 1
            y = (y == unique_vals[1]).astype(int)
        else:
            y = y.astype(int)
        
        print(f"  Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Feature dimensions: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df[url_column].values
    
    def build_feedforward_nn(self, input_dim):
        """Build a feedforward neural network"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_deep_nn(self, input_dim):
        """Build a deeper neural network"""
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_deep_learning(self, X_train, X_test, y_train, y_test, model_type='feedforward'):
        """Train deep learning model"""
        print(f"\nTraining {model_type} neural network...")
        
        # Build model
        if model_type == 'feedforward':
            model = self.build_feedforward_nn(X_train.shape[1])
        else:
            model = self.build_deep_nn(X_train.shape[1])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,  # Reduced from 100 for speed
            batch_size=256,  # Increased for speed
            callbacks=[early_stop, reduce_lr],
            verbose=0  # Silent training
        )
        
        print(f"  Completed in {len(history.history['loss'])} epochs")
        
        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Store results
        self.models[model_type] = model
        self.history[model_type] = history
        self.results[model_type] = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  {model_type} Accuracy: {self.results[model_type]['accuracy']:.4f}")
        print(f"  {model_type} F1-Score: {self.results[model_type]['f1_score']:.4f}")
        
        return model, history
    
    def train_svm(self, X_train, X_test, y_train, y_test):
        """Train SVM-like classifier (SGD for speed)"""
        print("\nTraining SVM (using SGD approximation for speed)...")
        
        # Use SGDClassifier which is MUCH faster than SVC on large datasets
        base_model = SGDClassifier(
            loss='hinge',        # SVM-like loss
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,          # Use all CPU cores
            verbose=0
        )
        
        # Calibrate for probability estimates
        model = CalibratedClassifierCV(base_model, cv=3)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        self.models['svm'] = model
        self.results['svm'] = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  SVM Accuracy: {self.results['svm']['accuracy']:.4f}")
        print(f"  SVM F1-Score: {self.results['svm']['f1_score']:.4f}")
        
        return model
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression"""
        print("\nTraining Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,  # Use all cores
            verbose=0
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  LR Accuracy: {self.results['logistic_regression']['accuracy']:.4f}")
        print(f"  LR F1-Score: {self.results['logistic_regression']['f1_score']:.4f}")
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def print_results(self):
        """Print comparison results"""
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        
        results_df = pd.DataFrame({
            model: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            }
            for model, metrics in self.results.items()
        }).T
        
        print("\n", results_df.round(4))
        return results_df
    
    def plot_results(self):
        """Plot comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Metrics comparison
        results_df = pd.DataFrame({
            model: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }
            for model, metrics in self.results.items()
        })
        
        results_df.T.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC-AUC comparison
        roc_data = pd.DataFrame({
            'Model': list(self.results.keys()),
            'ROC-AUC': [metrics['roc_auc'] for metrics in self.results.values()]
        })
        axes[0, 1].bar(roc_data['Model'], roc_data['ROC-AUC'])
        axes[0, 1].set_title('ROC-AUC Scores')
        axes[0, 1].set_ylabel('ROC-AUC Score')
        axes[0, 1].set_ylim([0.9, 1.0])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training history for DL models
        if 'feedforward' in self.history:
            history = self.history['feedforward']
            axes[1, 0].plot(history.history['accuracy'], label='Train Accuracy')
            axes[1, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
            axes[1, 0].set_title('Deep Learning Training History')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confusion matrix for best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        cm = self.results[best_model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model}')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('phishing_detection_results.png', dpi=300, bbox_inches='tight')
        print("\nResults plot saved as 'phishing_detection_results.png'")
        plt.close()


def main():
    """Main execution function"""
    
    detector = PhishingDetector()
    
    print("="*80)
    print("PHISHING URL DETECTION SYSTEM - OPTIMIZED VERSION")
    print("="*80)
    
    filepath = 'phishing_dataset.csv'
    df = detector.load_data(filepath)
    
    # Preprocess
    X_train, X_test, y_train, y_test, urls = detector.preprocess_data(df)
    
    # Train all models
    detector.train_deep_learning(X_train, X_test, y_train, y_test, model_type='feedforward')
    detector.train_deep_learning(X_train, X_test, y_train, y_test, model_type='deep')
    detector.train_svm(X_train, X_test, y_train, y_test)
    detector.train_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Print and plot results
    results_df = detector.print_results()
    detector.plot_results()
    
    # Save results
    results_df.to_csv('model_comparison_results.csv')
    print("\nResults saved to 'model_comparison_results.csv'")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
