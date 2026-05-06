import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========== ПУТИ ==========
CSV_PATH = r"C:\\Паша\\Проекты\\VKR\\wazuh_logs.csv"
RF_MODEL_PATH = r"C:\\Паша\\Проекты\\VKR\\rf_model.pkl"
XGB_MODEL_PATH = r"C:\\Паша\\Проекты\\VKR\\xgb_model.pkl"
OUTPUT_DIR = r"C:\\Паша\\Проекты\\VKR\\results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class WazuhCSVProcessor:
    """Обработка без группировки по IP — только по времени."""
    
    FEATURE_NAMES = [
        'rule_level', 'time_delta', 'event_count', 'unique_ips',
        'avg_message_len', 'max_message_len', 'events_per_sec',
        'severity_score', 'avg_rule_level', 'auth_failures',
        'brute_indicators', 'error_count', 'warning_count',
        'info_count', 'other_count'
    ]
    
    def __init__(self, csv_path: str, time_window: str = '2min'):
        self.csv_path = csv_path
        self.time_window = time_window
        self.df = None
        self.features = None
        
    def load_and_process(self):
        logger.info(f"Загрузка {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        logger.info(f"✅ Загружено {len(self.df)} записей")
        
        col_mapping = {col: col.replace('_source.', '').replace('.', '_') for col in self.df.columns}
        self.df = self.df.rename(columns=col_mapping)
        
        if 'predecoder_timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(
                self.df['predecoder_timestamp'], 
                format='%b %d %H:%M:%S',
                errors='coerce'
            )
            self.df['timestamp'] = self.df['timestamp'].apply(
                lambda x: x.replace(year=2026) if pd.notna(x) else x
            )
            logger.info(f"📅 Диапазон: {self.df['timestamp'].min()} - {self.df['timestamp'].max()}")
        
        return self
    
    def extract_features(self):
        """Группировка ТОЛЬКО по времени (без IP)."""
        logger.info(f"🔧 Извлечение признаков (окно: {self.time_window})...")
        
        self.df['time_window'] = self.df['timestamp'].dt.floor(self.time_window)
        grouped = self.df.groupby('time_window')
        
        logger.info(f"Найдено {len(grouped)} временных окон")
        
        features_list = []
        
        for window, group in grouped:
            if len(group) < 1:
                continue
            
            group = group.sort_values('timestamp')
            
            time_delta = (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
            if time_delta == 0:
                time_delta = 1
            
            rule_levels = pd.to_numeric(group.get('rule_level', 0), errors='coerce').fillna(0)
            descriptions = group.get('rule_description', '').fillna('').str.lower()
            unique_src_ips = group.get('data_srcip', pd.Series([0])).nunique()
            
            auth_failures = descriptions.str.contains('authentication|failed|password', na=False).sum()
            brute_indicators = descriptions.str.contains('brute|maximum attempts', na=False).sum()
            errors = descriptions.str.contains('error|failed', na=False).sum()
            warnings = descriptions.str.contains('warning|missed', na=False).sum()
            infos = descriptions.str.contains('install|success', na=False).sum()
            
            feat = {
                'time_window': window,
                'rule_level': rule_levels.mean(),
                'time_delta': time_delta,
                'event_count': len(group),
                'unique_ips': unique_src_ips,
                'avg_message_len': descriptions.str.len().mean(),
                'max_message_len': descriptions.str.len().max(),
                'events_per_sec': len(group) / time_delta,
                'severity_score': rule_levels.max(),
                'avg_rule_level': rule_levels.mean(),
                'auth_failures': auth_failures,
                'brute_indicators': brute_indicators,
                'error_count': errors,
                'warning_count': warnings,
                'info_count': infos,
                'other_count': len(group) - errors - warnings - infos
            }
            
            features_list.append(feat)
        
        self.features = pd.DataFrame(features_list)
        self.features = self.features.fillna(0)
        
        logger.info(f"✅ Извлечено {len(self.features)} feature-векторов")
        return self.features
    
    def get_X(self):
        if self.features is None:
            self.extract_features()
        
        X = self.features[self.FEATURE_NAMES].values
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        range_vals = X_max - X_min
        range_vals[range_vals == 0] = 1
        X_normalized = (X - X_min) / range_vals
        
        return X_normalized, self.features


class ModelTester:
    def __init__(self, X: np.ndarray, features_df: pd.DataFrame):
        self.X = X
        self.features_df = features_df
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
        
    def load_models(self):
        logger.info("📥 Загрузка моделей...")
        
        model_paths = {
            'RandomForest': RF_MODEL_PATH,
            'XGBoost': XGB_MODEL_PATH
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    self.models[name] = model
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    logger.info(f"✅ {name}: {size_mb:.1f} MB")
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки {name}: {e}")
            else:
                logger.error(f"❌ Файл не найден: {path}")
        
        if not self.models:
            raise ValueError("Не загружена ни одна модель!")
        
        return self
    
    def predict(self):
        logger.info("🤖 Предсказание...")
        
        for name, model in self.models.items():
            if hasattr(model, 'n_features_in_'):
                expected = model.n_features_in_
                actual = self.X.shape[1]
                
                if actual != expected:
                    logger.warning(f"⚠️ Адаптация: {actual} -> {expected}")
                    if actual < expected:
                        X = np.hstack([self.X, np.zeros((self.X.shape[0], expected - actual))])
                    else:
                        X = self.X[:, :expected]
                else:
                    X = self.X
            else:
                X = self.X
            
            preds = model.predict(X)
            self.predictions[name] = preds
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
            else:
                probs = preds.astype(float)
            self.probabilities[name] = probs
            
            n_benign = np.sum(preds == 0)
            n_attack = np.sum(preds == 1)
            logger.info(f"📊 {name}: Норма={n_benign}, Атака={n_attack} ({n_attack/len(preds)*100:.1f}%)")
        
        return self
    
    def calculate_metrics(self, y_true: np.ndarray = None):
        """Расчёт метрик. Если y_true нет — выводим только распределение."""
        logger.info("📊 Расчёт метрик...")
        
        metrics = {}
        
        for name in self.models.keys():
            preds = self.predictions[name]
            probs = self.probabilities[name]
            
            # Базовые метрики (без ground truth)
            n_total = len(preds)
            n_attack = np.sum(preds == 1)
            n_benign = np.sum(preds == 0)
            attack_rate = n_attack / n_total
            avg_prob = np.mean(probs)
            max_prob = np.max(probs)
            
            metrics[name] = {
                'Total': n_total,
                'Attacks': n_attack,
                'Benign': n_benign,
                'Attack_Rate_%': round(attack_rate * 100, 2),
                'Avg_Probability': round(avg_prob, 4),
                'Max_Probability': round(max_prob, 4)
            }
            
            # Если есть ground truth — считаем полные метрики
            if y_true is not None:
                metrics[name]['Accuracy'] = round(accuracy_score(y_true, preds), 4)
                metrics[name]['Precision'] = round(precision_score(y_true, preds, zero_division=0), 4)
                metrics[name]['Recall'] = round(recall_score(y_true, preds, zero_division=0), 4)
                metrics[name]['F1'] = round(f1_score(y_true, preds, zero_division=0), 4)
        
        return metrics
    
    def save_results(self):
        logger.info("💾 Сохранение...")
        
        results_df = self.features_df[['time_window']].copy()
        
        for name in self.models.keys():
            results_df[f'{name}_pred'] = self.predictions[name]
            results_df[f'{name}_prob'] = self.probabilities[name]
        
        if len(self.models) > 1:
            all_preds = np.array([self.predictions[name] for name in self.models.keys()])
            results_df['Consensus'] = np.round(all_preds.mean(axis=0)).astype(int)
            results_df['Consensus_Prob'] = np.mean(
                [self.probabilities[name] for name in self.models.keys()], axis=0
            )
        
        # ТОЛЬКО CSV (без Excel)
        csv_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV: {csv_path}")
        
        self.visualize()
        return results_df
    
    def visualize(self):
        logger.info("📊 Графики...")
        n_samples = len(self.features_df)
        
        # 1. Распределение
        fig, axes = plt.subplots(1, len(self.models), figsize=(6*len(self.models), 5))
        if len(self.models) == 1:
            axes = [axes]
        
        for ax, (name, preds) in zip(axes, self.predictions.items()):
            counts = [np.sum(preds == 0), np.sum(preds == 1)]
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.bar(['BENIGN', 'ATTACK'], counts, color=colors, edgecolor='black')
            ax.set_title(f'{name}\\n({len(preds)} flows)', fontweight='bold')
            ax.set_ylabel('Count')
            
            total = len(preds)
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\\n({count/total*100:.1f}%)',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ distribution.png")
        
        # 2. Вероятности
        plt.figure(figsize=(10, 6))
        for name, probs in self.probabilities.items():
            plt.hist(probs, bins=min(20, n_samples), alpha=0.6, label=name, edgecolor='black')
        plt.xlabel('Вероятность атаки')
        plt.ylabel('Количество flows')
        plt.title('Распределение вероятностей')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'probabilities.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ probabilities.png")
        
        # 3. Топ-N (ИСПРАВЛЕНО!)
        if len(self.models) > 1 and n_samples > 0:
            plt.figure(figsize=(12, 8))
            
            consensus_probs = np.mean(
                [self.probabilities[name] for name in self.models.keys()], axis=0
            )
            
            # ИСПРАВЛЕНИЕ: берём min(20, n_samples)
            n_top = min(20, n_samples)
            top_indices = np.argsort(consensus_probs)[-n_top:][::-1]
            
            plt.barh(range(n_top), consensus_probs[top_indices], color='#e74c3c')
            plt.yticks(range(n_top), [f'Flow {i+1}' for i in range(n_top)])
            plt.xlabel('Вероятность атаки (консенсус)')
            plt.title(f'Топ-{n_top} подозрительных flows')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'top_attacks.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✅ top_attacks.png (топ-{n_top})")


def main():
    print("=" * 70)
    print("🚀 ТЕСТИРОВАНИЕ МОДЕЛЕЙ НА ЛОГАХ WAZUH")
    print("=" * 70)
    
    try:
        processor = WazuhCSVProcessor(CSV_PATH, time_window='2min')
        processor.load_and_process()
        X, features_df = processor.get_X()
        
        logger.info(f"📐 X: {X.shape}")
        
        if len(features_df) == 0:
            logger.error("❌ Нет данных!")
            return
        
        tester = ModelTester(X, features_df)
        tester.load_models()
        tester.predict()
        
        # Метрики
        metrics = tester.calculate_metrics()
        
        results = tester.save_results()
        
        # Вывод результатов
        print("\\n" + "=" * 70)
        print("📋 МЕТРИКИ МОДЕЛЕЙ")
        print("=" * 70)
        
        for name, met in metrics.items():
            print(f"\\n{name}:")
            for k, v in met.items():
                print(f"  {k:20}: {v}")
        
        print("\\n" + "=" * 70)
        print("📋 ПРЕДСКАЗАНИЯ")
        print("=" * 70)
        
        for name in tester.models.keys():
            preds = tester.predictions[name]
            n_attack = np.sum(preds == 1)
            print(f"{name:15} | Атак: {n_attack:4d}/{len(preds)} ({n_attack/len(preds)*100:5.1f}%)")
        
        if len(tester.models) > 1:
            consensus = results['Consensus']
            n_attack = np.sum(consensus == 1)
            print(f"{'Консенсус':15} | Атак: {n_attack:4d}/{len(consensus)} ({n_attack/len(consensus)*100:5.1f}%)")
        
        print(f"\\n💾 Результаты: {OUTPUT_DIR}")
        print("=" * 70)
        print("✅ ГОТОВО!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
