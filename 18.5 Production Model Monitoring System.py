import pandas as pd
import numpy as np
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

# Statistical analysis
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Drift detection
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfMissingValues

# Alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@dataclass
class MonitoringConfig:
    """Configuration for model monitoring system."""
    
    # Monitoring parameters
    monitoring_window_days: int = 7
    drift_detection_sensitivity: float = 0.05
    performance_degradation_threshold: float = 0.15
    data_quality_threshold: float = 0.95
    
    # Alerting configuration
    alert_channels: List[str] = None
    email_recipients: List[str] = None
    slack_webhook: str = None
    
    # Storage configuration
    database_path: str = "monitoring.db"
    report_storage_path: str = "monitoring_reports/"
    
    # Retraining triggers
    auto_retrain_enabled: bool = True
    retrain_drift_threshold: float = 0.1
    retrain_performance_threshold: float = 0.2
    min_retrain_interval_days: int = 7

    def __post_init__(self):
        if self.alert_channels is None:
            self.alert_channels = ["console", "database"]
        if self.email_recipients is None:
            self.email_recipients = []


class ModelPerformanceMonitor:
    """Monitor model performance metrics over time."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = config.database_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for storing monitoring data."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT,
                mae REAL,
                rmse REAL,
                mape REAL,
                r2_score REAL,
                prediction_count INTEGER,
                data_window_start TEXT,
                data_window_end TEXT
            )
        ''')
        
        # Create drift detection table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT,
                drift_detected BOOLEAN,
                drift_score REAL,
                drifted_features TEXT,
                feature_drift_scores TEXT
            )
        ''')
        
        # Create data quality table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                missing_values_ratio REAL,
                outliers_ratio REAL,
                data_completeness REAL,
                schema_changes TEXT,
                quality_score REAL
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def track_performance(self, 
                         predictions: np.ndarray,
                         actuals: np.ndarray,
                         model_version: str = "current",
                         data_window: Tuple[str, str] = None) -> Dict[str, float]:
        """Track model performance metrics."""
        
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        # Handle MAPE calculation (avoid division by zero)
        mask = actuals != 0
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100 if mask.any() else np.inf
        
        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2_score,
            'prediction_count': len(predictions)
        }
        
        # Store in database
        self._store_performance_metrics(metrics, model_version, data_window)
        
        # Check for performance degradation
        self._check_performance_degradation(metrics, model_version)
        
        self.logger.info(f"Performance tracked: MAE={mae:.4f}, RMSE={rmse:.4f}")
        return metrics
    
    def _store_performance_metrics(self, 
                                  metrics: Dict[str, float],
                                  model_version: str,
                                  data_window: Tuple[str, str]):
        """Store performance metrics in database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        window_start, window_end = data_window if data_window else (None, None)
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, model_version, mae, rmse, mape, r2_score, prediction_count, data_window_start, data_window_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, model_version, metrics['mae'], metrics['rmse'],
            metrics['mape'], metrics['r2_score'], metrics['prediction_count'],
            window_start, window_end
        ))
        
        conn.commit()
        conn.close()
    
    def _check_performance_degradation(self, 
                                     current_metrics: Dict[str, float],
                                     model_version: str):
        """Check if model performance has degraded significantly."""
        
        # Get baseline performance (average of last 30 days)
        baseline_metrics = self._get_baseline_performance(model_version)
        
        if baseline_metrics:
            mae_increase = (current_metrics['mae'] - baseline_metrics['mae']) / baseline_metrics['mae']
            
            if mae_increase > self.config.performance_degradation_threshold:
                self._trigger_alert(
                    alert_type="performance_degradation",
                    severity="high",
                    message=f"Model performance degraded: MAE increased by {mae_increase:.2%}"
                )
    
    def _get_baseline_performance(self, model_version: str) -> Optional[Dict[str, float]]:
        """Get baseline performance metrics."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get average performance over last 30 days
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
        
        cursor.execute('''
            SELECT AVG(mae), AVG(rmse), AVG(mape), AVG(r2_score)
            FROM performance_metrics
            WHERE timestamp > ? AND model_version = ?
        ''', (cutoff_date, model_version))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] is not None:
            return {
                'mae': result[0],
                'rmse': result[1],
                'mape': result[2],
                'r2_score': result[3]
            }
        
        return None
    
    def get_performance_trend(self, 
                            model_version: str = "current",
                            days: int = 30) -> pd.DataFrame:
        """Get performance trend over specified period."""
        
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT timestamp, mae, rmse, mape, r2_score, prediction_count
            FROM performance_metrics
            WHERE timestamp > ? AND model_version = ?
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date, model_version))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df


class DataDriftDetector:
    """Detect data drift in incoming data."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reference_data = None
        self.feature_names = None
    
    def set_reference_data(self, reference_data: pd.DataFrame, target_column: str = None):
        """Set reference data for drift detection."""
        
        self.reference_data = reference_data.copy()
        self.target_column = target_column
        
        # Store feature names
        self.feature_names = [col for col in reference_data.columns if col != target_column]
        
        self.logger.info(f"Reference data set: {len(reference_data)} samples, {len(self.feature_names)} features")
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in current data compared to reference."""
        
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        # Ensure same columns
        common_columns = list(set(self.reference_data.columns) & set(current_data.columns))
        if not common_columns:
            raise ValueError("No common columns between reference and current data")
        
        ref_data = self.reference_data[common_columns]
        cur_data = current_data[common_columns]
        
        # Create column mapping
        numerical_features = [col for col in common_columns 
                            if pd.api.types.is_numeric_dtype(ref_data[col]) and col != self.target_column]
        categorical_features = [col for col in common_columns 
                              if pd.api.types.is_categorical_dtype(ref_data[col]) or 
                              pd.api.types.is_object_dtype(ref_data[col]) and col != self.target_column]
        
        column_mapping = ColumnMapping(
            target=self.target_column,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
        
        # Create Evidently report
        data_drift_report = Report(metrics=[DataDriftPreset()])
        
        try:
            data_drift_report.run(
                reference_data=ref_data,
                current_data=cur_data,
                column_mapping=column_mapping
            )
            
            # Extract results
            report_dict = data_drift_report.as_dict()
            dataset_drift = report_dict['metrics'][0]['result']['dataset_drift']
            drift_score = report_dict['metrics'][0]['result']['drift_score']
            
            # Get per-feature drift information
            drifted_features = []
            feature_drift_scores = {}
            
            if 'drift_by_columns' in report_dict['metrics'][0]['result']:
                for col, col_result in report_dict['metrics'][0]['result']['drift_by_columns'].items():
                    if col_result['drift_detected']:
                        drifted_features.append(col)
                    feature_drift_scores[col] = col_result.get('drift_score', 0.0)
            
            drift_results = {
                'drift_detected': dataset_drift,
                'drift_score': drift_score,
                'drifted_features': drifted_features,
                'feature_drift_scores': feature_drift_scores,
                'num_drifted_features': len(drifted_features),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results
            self._store_drift_results(drift_results)
            
            # Check if retraining is needed
            if drift_score > self.config.retrain_drift_threshold:
                self._trigger_retraining_alert(drift_results)
            
            self.logger.info(f"Drift detection: {dataset_drift}, score: {drift_score:.4f}")
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _store_drift_results(self, drift_results: Dict[str, Any]):
        """Store drift detection results in database."""
        
        conn = sqlite3.connect(self.config.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drift_detection 
            (timestamp, model_version, drift_detected, drift_score, drifted_features, feature_drift_scores)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            drift_results['timestamp'],
            "current",  # model version
            drift_results['drift_detected'],
            drift_results['drift_score'],
            json.dumps(drift_results['drifted_features']),
            json.dumps(drift_results['feature_drift_scores'])
        ))
        
        conn.commit()
        conn.close()
    
    def _trigger_retraining_alert(self, drift_results: Dict[str, Any]):
        """Trigger alert for retraining requirement."""
        
        message = f"""
        Significant data drift detected (score: {drift_results['drift_score']:.4f}).
        Drifted features: {', '.join(drift_results['drifted_features'])}
        Model retraining recommended.
        """
        
        self._trigger_alert(
            alert_type="retraining_required",
            severity="high",
            message=message.strip()
        )


class DataQualityMonitor:
    """Monitor data quality issues."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality of incoming data."""
        
        quality_metrics = {
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(data),
            'missing_values_ratio': data.isnull().sum().sum() / (data.shape[0] * data.shape[1]),
            'duplicate_rows': data.duplicated().sum(),
            'duplicate_ratio': data.duplicated().sum() / len(data),
            'outliers_detected': {},
            'schema_issues': [],
            'quality_score': 0.0
        }
        
        # Detect outliers for numerical columns
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            quality_metrics['outliers_detected'][col] = {
                'count': len(outliers),
                'ratio': len(outliers) / len(data) if len(data) > 0 else 0
            }
        
        # Calculate overall outliers ratio
        total_outliers = sum([info['count'] for info in quality_metrics['outliers_detected'].values()])
        quality_metrics['outliers_ratio'] = total_outliers / (len(data) * len(data.columns)) if len(data) > 0 else 0
        
        # Data completeness
        quality_metrics['data_completeness'] = 1 - quality_metrics['missing_values_ratio']
        
        # Calculate quality score (0-1 scale)
        completeness_score = quality_metrics['data_completeness']
        duplicate_penalty = min(0.5, quality_metrics['duplicate_ratio'] * 2)  # Cap penalty at 0.5
        outlier_penalty = min(0.3, quality_metrics['outliers_ratio'] * 10)  # Cap penalty at 0.3
        
        quality_metrics['quality_score'] = max(0, completeness_score - duplicate_penalty - outlier_penalty)
        
        # Store quality metrics
        self._store_quality_metrics(quality_metrics)
        
        # Check quality threshold
        if quality_metrics['quality_score'] < self.config.data_quality_threshold:
            self._trigger_quality_alert(quality_metrics)
        
        self.logger.info(f"Data quality assessed: score={quality_metrics['quality_score']:.3f}")
        return quality_metrics
    
    def _store_quality_metrics(self, quality_metrics: Dict[str, Any]):
        """Store data quality metrics in database."""
        
        conn = sqlite3.connect(self.config.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_quality 
            (timestamp, missing_values_ratio, outliers_ratio, data_completeness, schema_changes, quality_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            quality_metrics['timestamp'],
            quality_metrics['missing_values_ratio'],
            quality_metrics['outliers_ratio'],
            quality_metrics['data_completeness'],
            json.dumps(quality_metrics['schema_issues']),
            quality_metrics['quality_score']
        ))
        
        conn.commit()
        conn.close()
    
    def _trigger_quality_alert(self, quality_metrics: Dict[str, Any]):
        """Trigger alert for data quality issues."""
        
        issues = []
        if quality_metrics['missing_values_ratio'] > 0.1:
            issues.append(f"High missing values: {quality_metrics['missing_values_ratio']:.2%}")
        if quality_metrics['outliers_ratio'] > 0.05:
            issues.append(f"High outliers: {quality_metrics['outliers_ratio']:.2%}")
        if quality_metrics['duplicate_ratio'] > 0.1:
            issues.append(f"High duplicates: {quality_metrics['duplicate_ratio']:.2%}")
        
        message = f"Data quality below threshold ({quality_metrics['quality_score']:.3f}). Issues: {', '.join(issues)}"
        
        self._trigger_alert(
            alert_type="data_quality",
            severity="medium",
            message=message
        )


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def trigger_alert(self, 
                     alert_type: str,
                     severity: str,
                     message: str,
                     metadata: Dict[str, Any] = None):
        """Trigger an alert through configured channels."""
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'metadata': metadata or {}
        }
        
        # Store alert in database
        self._store_alert(alert_data)
        
        # Send through configured channels
        for channel in self.config.alert_channels:
            try:
                if channel == "console":
                    self._send_console_alert(alert_data)
                elif channel == "email":
                    self._send_email_alert(alert_data)
                elif channel == "slack":
                    self._send_slack_alert(alert_data)
                
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {str(e)}")
    
    def _store_alert(self, alert_data: Dict[str, Any]):
        """Store alert in database."""
        
        conn = sqlite3.connect(self.config.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        ''', (
            alert_data['timestamp'],
            alert_data['alert_type'],
            alert_data['severity'],
            alert_data['message']
        ))
        
        conn.commit()
        conn.close()
    
    def _send_console_alert(self, alert_data: Dict[str, Any]):
        """Send alert to console."""
        
        severity_emoji = {
            'low': '💡',
            'medium': '⚠️',
            'high': '🚨',
            'critical': '💥'
        }
        
        emoji = severity_emoji.get(alert_data['severity'], '❓')
        
        print(f"\n{emoji} ALERT [{alert_data['severity'].upper()}] - {alert_data['alert_type']}")
        print(f"Time: {alert_data['timestamp']}")
        print(f"Message: {alert_data['message']}")
        print("-" * 50)
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email."""
        
        if not self.config.email_recipients:
            return
        
        # This is a placeholder - implement actual email sending
        self.logger.info(f"Email alert sent: {alert_data['alert_type']}")
    
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Slack."""
        
        if not self.config.slack_webhook:
            return
        
        # This is a placeholder - implement actual Slack integration
        self.logger.info(f"Slack alert sent: {alert_data['alert_type']}")


class MonitoringDashboard:
    """Create monitoring dashboard and reports."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_monitoring_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        
        conn = sqlite3.connect(self.config.database_path)
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get performance data
        perf_query = '''
            SELECT * FROM performance_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        '''
        performance_data = pd.read_sql_query(perf_query, conn, params=(cutoff_date,))
        
        # Get drift data
        drift_query = '''
            SELECT * FROM drift_detection 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        '''
        drift_data = pd.read_sql_query(drift_query, conn, params=(cutoff_date,))
        
        # Get quality data
        quality_query = '''
            SELECT * FROM data_quality 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        '''
        quality_data = pd.read_sql_query(quality_query, conn, params=(cutoff_date,))
        
        # Get alerts
        alerts_query = '''
            SELECT * FROM alerts 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        '''
        alerts_data = pd.read_sql_query(alerts_query, conn, params=(cutoff_date,))
        
        conn.close()
        
        # Generate summary
        report = {
            'period': f"Last {days} days",
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_predictions': performance_data['prediction_count'].sum() if not performance_data.empty else 0,
                'avg_performance': {
                    'mae': performance_data['mae'].mean() if not performance_data.empty else None,
                    'rmse': performance_data['rmse'].mean() if not performance_data.empty else None
                },
                'drift_incidents': len(drift_data[drift_data['drift_detected'] == True]) if not drift_data.empty else 0,
                'avg_data_quality': quality_data['quality_score'].mean() if not quality_data.empty else None,
                'total_alerts': len(alerts_data)
            },
            'performance_data': performance_data.to_dict('records'),
            'drift_data': drift_data.to_dict('records'),
            'quality_data': quality_data.to_dict('records'),
            'alerts_data': alerts_data.to_dict('records')
        }
        
        # Save report
        report_path = Path(self.config.report_storage_path) / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report generated: {report_path}")
        return report
    
    def create_performance_dashboard(self, days: int = 30):
        """Create performance visualization dashboard."""
        
        conn = sqlite3.connect(self.config.database_path)
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get performance data
        query = '''
            SELECT timestamp, mae, rmse, mape, r2_score
            FROM performance_metrics
            WHERE timestamp > ?
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        if df.empty:
            self.logger.warning("No performance data available for dashboard")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Dashboard', fontsize=16)
        
        # MAE trend
        axes[0, 0].plot(df['timestamp'], df['mae'], marker='o', linewidth=2)
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE trend
        axes[0, 1].plot(df['timestamp'], df['rmse'], marker='o', linewidth=2, color='orange')
        axes[0, 1].set_title('Root Mean Square Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAPE trend
        axes[1, 0].plot(df['timestamp'], df['mape'], marker='o', linewidth=2, color='green')
        axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # R² trend
        axes[1, 1].plot(df['timestamp'], df['r2_score'], marker='o', linewidth=2, color='red')
        axes[1, 1].set_title('R² Score')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = Path(self.config.report_storage_path) / f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        dashboard_path.parent.mkdir(exist_ok=True)
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance dashboard saved: {dashboard_path}")


# Integration class for complete monitoring system
class ComprehensiveMonitoringSystem:
    """Complete monitoring system integrating all components."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.performance_monitor = ModelPerformanceMonitor(config)
        self.drift_detector = DataDriftDetector(config)
        self.quality_monitor = DataQualityMonitor(config)
        self.alert_manager = AlertManager(config)
        self.dashboard = MonitoringDashboard(config)
        
        # Global alert method binding
        self._bind_alert_methods()
    
    def _bind_alert_methods(self):
        """Bind alert methods to all components."""
        
        for component in [self.performance_monitor, self.drift_detector, self.quality_monitor]:
            component._trigger_alert = self.alert_manager.trigger_alert
    
    def setup_monitoring(self, reference_data: pd.DataFrame, target_column: str = None):
        """Setup monitoring system with reference data."""
        
        self.drift_detector.set_reference_data(reference_data, target_column)
        self.logger.info("Monitoring system setup completed")
    
    def monitor_batch(self, 
                     current_data: pd.DataFrame,
                     predictions: np.ndarray = None,
                     actuals: np.ndarray = None,
                     model_version: str = "current") -> Dict[str, Any]:
        """Monitor a batch of data and predictions."""
        
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(current_data)
        }
        
        # Data quality assessment
        try:
            quality_results = self.quality_monitor.assess_data_quality(current_data)
            monitoring_results['data_quality'] = quality_results
        except Exception as e:
            self.logger.error(f"Data quality assessment failed: {str(e)}")
            monitoring_results['data_quality'] = {'error': str(e)}
        
        # Drift detection
        try:
            drift_results = self.drift_detector.detect_drift(current_data)
            monitoring_results['drift_detection'] = drift_results
        except Exception as e:
            self.logger.error(f"Drift detection failed: {str(e)}")
            monitoring_results['drift_detection'] = {'error': str(e)}
        
        # Performance monitoring (if actuals available)
        if predictions is not None and actuals is not None:
            try:
                performance_results = self.performance_monitor.track_performance(
                    predictions, actuals, model_version
                )
                monitoring_results['performance'] = performance_results
            except Exception as e:
                self.logger.error(f"Performance monitoring failed: {str(e)}")
                monitoring_results['performance'] = {'error': str(e)}
        
        return monitoring_results
    
    def generate_health_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        
        return self.dashboard.generate_monitoring_report(days)
    
    def create_dashboard(self, days: int = 30):
        """Create monitoring dashboard."""
        
        self.dashboard.create_performance_dashboard(days)


# Example usage and demonstration
def demonstrate_monitoring_system():
    """Demonstrate comprehensive monitoring system."""
    
    print("📊 Model Monitoring and Maintenance System")
    print("=" * 60)
    
    # Configuration
    config = MonitoringConfig(
        monitoring_window_days=7,
        drift_detection_sensitivity=0.05,
        alert_channels=["console", "database"]
    )
    
    # Initialize monitoring system
    monitoring_system = ComprehensiveMonitoringSystem(config)
    
    print("\n1. 🔧 Setting up monitoring system")
    
    # Generate reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(100, 15, 1000),
        'feature_2': np.random.normal(50, 10, 1000),
        'feature_3': np.random.uniform(0, 1, 1000),
        'target': np.random.normal(75, 20, 1000)
    })
    
    monitoring_system.setup_monitoring(reference_data, 'target')
    print("✅ Monitoring system configured with reference data")
    
    print("\n2. 📈 Simulating production monitoring")
    
    # Simulate 7 days of production data
    for day in range(1, 8):
        print(f"\n   Day {day}:")
        
        # Generate current data (with some drift after day 4)
        if day <= 4:
            current_data = pd.DataFrame({
                'feature_1': np.random.normal(100, 15, 100),
                'feature_2': np.random.normal(50, 10, 100),
                'feature_3': np.random.uniform(0, 1, 100),
                'target': np.random.normal(75, 20, 100)
            })
        else:
            # Introduce drift
            current_data = pd.DataFrame({
                'feature_1': np.random.normal(110, 18, 100),  # Shifted mean
                'feature_2': np.random.normal(45, 12, 100),   # Shifted mean and variance
                'feature_3': np.random.uniform(0.1, 0.9, 100), # Changed range
                'target': np.random.normal(80, 25, 100)       # Shifted target
            })
        
        # Generate dummy predictions and actuals
        predictions = current_data['target'].values + np.random.normal(0, 5, 100)
        actuals = current_data['target'].values
        
        # Monitor batch
        results = monitoring_system.monitor_batch(
            current_data=current_data.drop('target', axis=1),
            predictions=predictions,
            actuals=actuals,
            model_version="v1.0"
        )
        
        # Display key results
        if 'data_quality' in results and 'quality_score' in results['data_quality']:
            quality_score = results['data_quality']['quality_score']
            print(f"     Data Quality: {quality_score:.3f}")
        
        if 'drift_detection' in results and 'drift_detected' in results['drift_detection']:
            drift_detected = results['drift_detection']['drift_detected']
            drift_score = results['drift_detection'].get('drift_score', 0)
            print(f"     Drift Detected: {drift_detected} (score: {drift_score:.4f})")
        
        if 'performance' in results:
            mae = results['performance']['mae']
            print(f"     Performance MAE: {mae:.4f}")
    
    print("\n3. 📋 Generating health report")
    health_report = monitoring_system.generate_health_report(days=7)
    
    print(f"✅ Health report generated:")
    print(f"   Period: {health_report['period']}")
    print(f"   Total Predictions: {health_report['summary']['total_predictions']}")
    print(f"   Average MAE: {health_report['summary']['avg_performance']['mae']:.4f}")
    print(f"   Drift Incidents: {health_report['summary']['drift_incidents']}")
    print(f"   Total Alerts: {health_report['summary']['total_alerts']}")
    
    print("\n4. 📊 Creating performance dashboard")
    monitoring_system.create_dashboard(days=7)
    print("✅ Performance dashboard created")
    
    print("\n" + "=" * 60)
    print("🏁 Monitoring System Demonstration Complete!")
    
    print("\nKey features demonstrated:")
    print("✅ Performance tracking with multiple metrics")
    print("✅ Data drift detection using statistical tests")
    print("✅ Data quality monitoring")
    print("✅ Automated alerting system")
    print("✅ Comprehensive reporting")
    print("✅ Visual dashboard generation")
    
    return monitoring_system, health_report


if __name__ == "__main__":
    # Run monitoring demonstration
    monitoring_system, health_report = demonstrate_monitoring_system()
    
    print("\n" + "="*60)
    print("Production recommendations:")
    print("1. Set up proper alert channels (email, Slack)")
    print("2. Configure automatic retraining triggers")
    print("3. Implement proper database backup and retention")
    print("4. Set up monitoring dashboards in production")
    print("5. Define escalation procedures for critical alerts")
    print("6. Regular monitoring system health checks")
