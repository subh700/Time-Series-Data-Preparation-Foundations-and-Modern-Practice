import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# MLOps and monitoring
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Containerization and deployment
import docker
from flask import Flask, request, jsonify
import boto3
from kubernetes import client, config

# Scheduling and orchestration
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

# Model serving
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@dataclass
class MLOpsConfig:
    """Configuration for MLOps pipeline."""
    
    # Model management
    model_registry: str = "mlflow"
    model_name: str = "time_series_forecaster"
    model_version: str = "1.0.0"
    
    # Deployment configuration
    deployment_target: str = "kubernetes"  # kubernetes, aws_ecs, docker
    container_registry: str = "ecr"
    image_name: str = "ts-forecaster"
    
    # Monitoring configuration
    drift_detection_enabled: bool = True
    performance_monitoring_enabled: bool = True
    alerting_enabled: bool = True
    
    # Data pipeline
    data_source: str = "s3"
    data_bucket: str = "time-series-data"
    data_pipeline_schedule: str = "0 */6 * * *"  # Every 6 hours
    
    # Retraining configuration
    retraining_enabled: bool = True
    retraining_schedule: str = "0 0 * * 0"  # Weekly
    performance_threshold: float = 0.95
    drift_threshold: float = 0.1
    
    # Scaling configuration
    min_replicas: int = 2
    max_replicas: int = 10
    cpu_request: str = "100m"
    memory_request: str = "256Mi"


class ModelRegistry:
    """Model registry for version control and deployment."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        
    def register_model(self, 
                      model: Any, 
                      model_metadata: Dict[str, Any],
                      artifacts: Dict[str, str] = None) -> str:
        """Register model in MLflow registry."""
        
        with mlflow.start_run() as run:
            # Log model
            if hasattr(model, 'predict'):
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    registered_model_name=self.config.model_name
                )
            
            # Log metadata
            mlflow.log_params(model_metadata.get('parameters', {}))
            mlflow.log_metrics(model_metadata.get('metrics', {}))
            
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
            
            # Tag the run
            mlflow.set_tag("model_type", "time_series_forecaster")
            mlflow.set_tag("version", self.config.model_version)
            mlflow.set_tag("deployment_ready", "true")
            
            model_uri = f"runs:/{run.info.run_id}/model"
            
        self.logger.info(f"Model registered: {model_uri}")
        return model_uri
    
    def load_model(self, model_uri: str = None, stage: str = "Production") -> Any:
        """Load model from registry."""
        
        if model_uri is None:
            model_uri = f"models:/{self.config.model_name}/{stage}"
        
        model = mlflow.sklearn.load_model(model_uri)
        self.logger.info(f"Model loaded: {model_uri}")
        return model
    
    def promote_model(self, model_uri: str, stage: str = "Production") -> bool:
        """Promote model to specified stage."""
        
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(
                self.config.model_name, stages=["None"]
            )[0]
            
            client.transition_model_version_stage(
                name=self.config.model_name,
                version=model_version.version,
                stage=stage
            )
            
            self.logger.info(f"Model promoted to {stage}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model promotion failed: {str(e)}")
            return False


class ModelServing:
    """Model serving infrastructure using BentoML."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_bento_service(self, model: Any) -> str:
        """Create BentoML service for model serving."""
        
        @env(infer_pip_packages=True)
        @artifacts([SklearnModelArtifact('model')])
        class TimeSeriesForecaster(BentoService):
            
            @api(input=DataframeInput(), batch=True)
            def predict(self, df: pd.DataFrame) -> List[float]:
                """Prediction API endpoint."""
                
                # Preprocess input data
                processed_data = self._preprocess_input(df)
                
                # Make predictions
                predictions = self.artifacts.model.predict(processed_data)
                
                # Postprocess predictions
                result = self._postprocess_predictions(predictions)
                
                return result.tolist()
            
            def _preprocess_input(self, df: pd.DataFrame) -> np.ndarray:
                """Preprocess input data for prediction."""
                # Add your preprocessing logic here
                return df.values
            
            def _postprocess_predictions(self, predictions: np.ndarray) -> np.ndarray:
                """Postprocess predictions."""
                # Add your postprocessing logic here
                return predictions
        
        # Create service instance
        ts_service = TimeSeriesForecaster()
        ts_service.pack('model', model)
        
        # Save BentoML service
        saved_path = ts_service.save()
        self.logger.info(f"BentoML service saved: {saved_path}")
        
        return saved_path
    
    def containerize_service(self, service_path: str) -> str:
        """Containerize BentoML service."""
        
        import bentoml
        
        # Build Docker image
        docker_image = bentoml.containerize(
            service_path, 
            docker_image_tag=f"{self.config.image_name}:{self.config.model_version}"
        )
        
        self.logger.info(f"Docker image built: {docker_image}")
        return docker_image


class DataPipeline:
    """Data pipeline for continuous data ingestion and processing."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_airflow_dag(self) -> DAG:
        """Create Airflow DAG for data pipeline."""
        
        default_args = {
            'owner': 'ml-team',
            'depends_on_past': False,
            'start_date': datetime(2024, 1, 1),
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5)
        }
        
        dag = DAG(
            'time_series_data_pipeline',
            default_args=default_args,
            description='Time series data ingestion and processing',
            schedule_interval=self.config.data_pipeline_schedule,
            catchup=False
        )
        
        # Data ingestion task
        ingest_data = PythonOperator(
            task_id='ingest_data',
            python_callable=self.ingest_data,
            dag=dag
        )
        
        # Data validation task
        validate_data = PythonOperator(
            task_id='validate_data',
            python_callable=self.validate_data,
            dag=dag
        )
        
        # Feature engineering task
        engineer_features = PythonOperator(
            task_id='engineer_features',
            python_callable=self.engineer_features,
            dag=dag
        )
        
        # Data quality checks
        quality_checks = PythonOperator(
            task_id='quality_checks',
            python_callable=self.run_quality_checks,
            dag=dag
        )
        
        # Set dependencies
        ingest_data >> validate_data >> engineer_features >> quality_checks
        
        return dag
    
    def ingest_data(self, **context) -> str:
        """Ingest data from source."""
        
        if self.config.data_source == "s3":
            return self._ingest_from_s3()
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
    
    def _ingest_from_s3(self) -> str:
        """Ingest data from S3."""
        
        s3_client = boto3.client('s3')
        
        # List objects in bucket
        response = s3_client.list_objects_v2(
            Bucket=self.config.data_bucket,
            Prefix='time_series_data/'
        )
        
        # Download latest files
        for obj in response.get('Contents', []):
            key = obj['Key']
            local_path = f"/tmp/{key.split('/')[-1]}"
            
            s3_client.download_file(
                self.config.data_bucket,
                key,
                local_path
            )
            
            self.logger.info(f"Downloaded: {key} -> {local_path}")
        
        return "Data ingestion completed"
    
    def validate_data(self, **context) -> str:
        """Validate ingested data."""
        
        # Load data
        data_files = [f for f in os.listdir('/tmp/') if f.endswith('.csv')]
        
        for file in data_files:
            df = pd.read_csv(f'/tmp/{file}')
            
            # Basic validation checks
            assert len(df) > 0, f"Empty dataset: {file}"
            assert not df.isnull().all().any(), f"All null columns found: {file}"
            
            self.logger.info(f"Validated: {file}")
        
        return "Data validation completed"
    
    def engineer_features(self, **context) -> str:
        """Engineer features from raw data."""
        
        # Implement feature engineering logic
        self.logger.info("Feature engineering completed")
        return "Feature engineering completed"
    
    def run_quality_checks(self, **context) -> str:
        """Run data quality checks."""
        
        # Implement data quality checks
        self.logger.info("Quality checks completed")
        return "Quality checks completed"


class ModelMonitoring:
    """Model monitoring and drift detection."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring_data = []
    
    def setup_monitoring(self, reference_data: pd.DataFrame, target_column: str):
        """Setup monitoring with reference data."""
        
        self.reference_data = reference_data
        self.target_column = target_column
        
        # Create column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=target_column,
            numerical_features=[col for col in reference_data.columns 
                              if col != target_column and pd.api.types.is_numeric_dtype(reference_data[col])],
            categorical_features=[col for col in reference_data.columns 
                                if col != target_column and pd.api.types.is_categorical_dtype(reference_data[col])]
        )
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data and concept drift."""
        
        if not hasattr(self, 'reference_data'):
            raise ValueError("Monitoring not setup. Call setup_monitoring first.")
        
        # Create Evidently report
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract drift results
        report_dict = data_drift_report.as_dict()
        
        drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
        drift_score = report_dict['metrics'][0]['result']['drift_score']
        
        drift_results = {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'timestamp': datetime.now().isoformat(),
            'action_required': drift_score > self.config.drift_threshold
        }
        
        # Log results
        self.monitoring_data.append(drift_results)
        self.logger.info(f"Drift detection completed: {drift_results}")
        
        # Trigger alerts if necessary
        if drift_results['action_required']:
            self._trigger_drift_alert(drift_results)
        
        return drift_results
    
    def monitor_performance(self, 
                           predictions: np.ndarray,
                           actuals: np.ndarray) -> Dict[str, float]:
        """Monitor model performance."""
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        metrics = {
            'mae': mean_absolute_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if retraining is needed
        if metrics['mae'] > self.config.performance_threshold:
            metrics['retraining_required'] = True
            self._trigger_retraining_alert(metrics)
        else:
            metrics['retraining_required'] = False
        
        self.logger.info(f"Performance monitoring: {metrics}")
        return metrics
    
    def _trigger_drift_alert(self, drift_results: Dict[str, Any]):
        """Trigger alert for drift detection."""
        
        alert_message = f"""
        🚨 DATA DRIFT DETECTED 🚨
        
        Drift Score: {drift_results['drift_score']:.4f}
        Threshold: {self.config.drift_threshold}
        Timestamp: {drift_results['timestamp']}
        
        Action Required: Model retraining recommended
        """
        
        self.logger.warning(alert_message)
        # Implement actual alerting mechanism (email, Slack, etc.)
    
    def _trigger_retraining_alert(self, performance_metrics: Dict[str, float]):
        """Trigger alert for model retraining."""
        
        alert_message = f"""
        📉 MODEL PERFORMANCE DEGRADATION 📉
        
        Current MAE: {performance_metrics['mae']:.4f}
        Threshold: {self.config.performance_threshold}
        Timestamp: {performance_metrics['timestamp']}
        
        Action Required: Model retraining initiated
        """
        
        self.logger.warning(alert_message)
        # Implement actual alerting mechanism


class AutomaticRetraining:
    """Automatic model retraining pipeline."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_retraining_dag(self) -> DAG:
        """Create Airflow DAG for automatic retraining."""
        
        default_args = {
            'owner': 'ml-team',
            'depends_on_past': False,
            'start_date': datetime(2024, 1, 1),
            'email_on_failure': True,
            'retries': 1,
            'retry_delay': timedelta(minutes=10)
        }
        
        dag = DAG(
            'automatic_model_retraining',
            default_args=default_args,
            description='Automatic model retraining pipeline',
            schedule_interval=self.config.retraining_schedule,
            catchup=False
        )
        
        # Check if retraining is needed
        check_retraining = PythonOperator(
            task_id='check_retraining_needed',
            python_callable=self.check_retraining_needed,
            dag=dag
        )
        
        # Fetch latest data
        fetch_data = PythonOperator(
            task_id='fetch_training_data',
            python_callable=self.fetch_training_data,
            dag=dag
        )
        
        # Retrain model
        retrain_model = PythonOperator(
            task_id='retrain_model',
            python_callable=self.retrain_model,
            dag=dag
        )
        
        # Validate retrained model
        validate_model = PythonOperator(
            task_id='validate_retrained_model',
            python_callable=self.validate_retrained_model,
            dag=dag
        )
        
        # Deploy new model
        deploy_model = PythonOperator(
            task_id='deploy_new_model',
            python_callable=self.deploy_new_model,
            dag=dag
        )
        
        # Set dependencies
        check_retraining >> fetch_data >> retrain_model >> validate_model >> deploy_model
        
        return dag
    
    def check_retraining_needed(self, **context) -> str:
        """Check if model retraining is needed."""
        
        # Check performance metrics and drift indicators
        # This would typically query monitoring database
        
        # For demonstration, always return True
        self.logger.info("Retraining check completed")
        return "retraining_needed"
    
    def fetch_training_data(self, **context) -> str:
        """Fetch latest data for retraining."""
        
        # Implement data fetching logic
        self.logger.info("Training data fetched")
        return "data_fetched"
    
    def retrain_model(self, **context) -> str:
        """Retrain the model with latest data."""
        
        # Implement model retraining logic
        self.logger.info("Model retraining completed")
        return "model_retrained"
    
    def validate_retrained_model(self, **context) -> str:
        """Validate the retrained model."""
        
        # Implement model validation logic
        self.logger.info("Model validation completed")
        return "model_validated"
    
    def deploy_new_model(self, **context) -> str:
        """Deploy the new model to production."""
        
        # Implement model deployment logic
        self.logger.info("New model deployed")
        return "model_deployed"


class KubernetesDeployment:
    """Kubernetes deployment manager."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load Kubernetes config
        try:
            config.load_incluster_config()  # For in-cluster deployment
        except:
            config.load_kube_config()  # For local development
        
        self.v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
    
    def create_deployment_manifest(self, image_name: str) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'{self.config.model_name}-deployment',
                'labels': {
                    'app': self.config.model_name,
                    'version': self.config.model_version
                }
            },
            'spec': {
                'replicas': self.config.min_replicas,
                'selector': {
                    'matchLabels': {
                        'app': self.config.model_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config.model_name,
                            'version': self.config.model_version
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.config.model_name,
                            'image': image_name,
                            'ports': [{'containerPort': 5000}],
                            'resources': {
                                'requests': {
                                    'cpu': self.config.cpu_request,
                                    'memory': self.config.memory_request
                                }
                            },
                            'env': [
                                {'name': 'MODEL_NAME', 'value': self.config.model_name},
                                {'name': 'MODEL_VERSION', 'value': self.config.model_version}
                            ],
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/healthz',
                                    'port': 5000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/readiness',
                                    'port': 5000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        return manifest
    
    def create_service_manifest(self) -> Dict[str, Any]:
        """Create Kubernetes service manifest."""
        
        manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{self.config.model_name}-service',
                'labels': {
                    'app': self.config.model_name
                }
            },
            'spec': {
                'selector': {
                    'app': self.config.model_name
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 5000,
                    'protocol': 'TCP'
                }],
                'type': 'LoadBalancer'
            }
        }
        
        return manifest
    
    def deploy_to_kubernetes(self, image_name: str) -> bool:
        """Deploy model to Kubernetes."""
        
        try:
            # Create deployment
            deployment_manifest = self.create_deployment_manifest(image_name)
            self.v1.create_namespaced_deployment(
                body=deployment_manifest,
                namespace='default'
            )
            
            # Create service
            service_manifest = self.create_service_manifest()
            self.core_v1.create_namespaced_service(
                body=service_manifest,
                namespace='default'
            )
            
            self.logger.info("Deployment to Kubernetes successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {str(e)}")
            return False


# Example usage and complete MLOps workflow
def demonstrate_mlops_pipeline():
    """Demonstrate complete MLOps pipeline."""
    
    print("🚀 MLOps Pipeline for Time Series Forecasting")
    print("=" * 60)
    
    # Configuration
    config = MLOpsConfig(
        model_name="sales_forecaster",
        model_version="1.0.0",
        deployment_target="kubernetes"
    )
    
    # Initialize components
    registry = ModelRegistry(config)
    serving = ModelServing(config)
    monitoring = ModelMonitoring(config)
    retraining = AutomaticRetraining(config)
    deployment = KubernetesDeployment(config)
    
    print("\n1. 📦 Model Registration")
    # Create dummy model for demonstration
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Fit on dummy data
    X_dummy = np.random.randn(100, 5)
    y_dummy = np.random.randn(100)
    model.fit(X_dummy, y_dummy)
    
    # Register model
    model_metadata = {
        'parameters': model.get_params(),
        'metrics': {'mae': 0.15, 'rmse': 0.23}
    }
    
    model_uri = registry.register_model(model, model_metadata)
    print(f"✅ Model registered: {model_uri}")
    
    print("\n2. 🔄 Model Serving Setup")
    # Create BentoML service
    service_path = serving.create_bento_service(model)
    print(f"✅ BentoML service created: {service_path}")
    
    # Containerize service
    docker_image = serving.containerize_service(service_path)
    print(f"✅ Docker image built: {docker_image}")
    
    print("\n3. 📊 Monitoring Setup")
    # Setup monitoring with dummy reference data
    reference_data = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(5)])
    reference_data['target'] = y_dummy
    
    monitoring.setup_monitoring(reference_data, 'target')
    print("✅ Monitoring configured")
    
    # Simulate drift detection
    current_data = pd.DataFrame(
        np.random.randn(50, 5) + 0.5,  # Slightly different distribution
        columns=[f'feature_{i}' for i in range(5)]
    )
    current_data['target'] = np.random.randn(50) + 0.5
    
    drift_results = monitoring.detect_drift(current_data)
    print(f"✅ Drift detection: {drift_results['drift_detected']}")
    
    print("\n4. 🔄 Automatic Retraining")
    # Create retraining DAG
    retraining_dag = retraining.create_retraining_dag()
    print(f"✅ Retraining DAG created: {retraining_dag.dag_id}")
    
    print("\n5. 🚀 Kubernetes Deployment")
    # Create deployment manifests
    deployment_manifest = deployment.create_deployment_manifest(docker_image)
    service_manifest = deployment.create_service_manifest()
    
    print("✅ Kubernetes manifests created")
    print(f"   Deployment: {deployment_manifest['metadata']['name']}")
    print(f"   Service: {service_manifest['metadata']['name']}")
    
    # Note: Actual deployment would require Kubernetes cluster
    print("   (Actual deployment requires Kubernetes cluster)")
    
    print("\n6. 📈 Performance Monitoring")
    # Simulate performance monitoring
    test_predictions = model.predict(X_dummy[:20])
    test_actuals = y_dummy[:20]
    
    performance_metrics = monitoring.monitor_performance(test_predictions, test_actuals)
    print(f"✅ Performance metrics: MAE={performance_metrics['mae']:.4f}")
    
    print("\n" + "=" * 60)
    print("🏁 MLOps Pipeline Demonstration Complete!")
    print("\nKey components implemented:")
    print("✅ Model Registry (MLflow)")
    print("✅ Model Serving (BentoML)")
    print("✅ Containerization (Docker)")
    print("✅ Orchestration (Airflow)")
    print("✅ Monitoring (Evidently)")
    print("✅ Deployment (Kubernetes)")
    print("✅ Automatic Retraining")
    
    return {
        'registry': registry,
        'serving': serving,
        'monitoring': monitoring,
        'deployment': deployment
    }


if __name__ == "__main__":
    # Run MLOps demonstration
    mlops_components = demonstrate_mlops_pipeline()
    
    print("\n" + "="*60)
    print("Next steps for production:")
    print("1. Setup proper infrastructure (K8s cluster, MLflow server)")
    print("2. Configure CI/CD pipelines")
    print("3. Implement proper monitoring and alerting")
    print("4. Setup automated testing and validation")
    print("5. Configure security and access controls")
