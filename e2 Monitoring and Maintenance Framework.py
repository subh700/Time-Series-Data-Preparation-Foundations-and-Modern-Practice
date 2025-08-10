class ProductionMonitoringFramework:
    """Framework for monitoring time series models in production."""
    
    def __init__(self):
        self.monitoring_layers = self._define_monitoring_layers()
        self.alert_thresholds = self._define_alert_thresholds()
        self.maintenance_schedule = self._create_maintenance_schedule()
    
    def _define_monitoring_layers(self) -> Dict[str, Any]:
        """Define comprehensive monitoring layers."""
        
        return {
            "infrastructure_monitoring": {
                "description": "System health and resource utilization",
                "metrics": [
                    {"name": "CPU Utilization", "threshold": "< 80%", "alert_level": "Warning"},
                    {"name": "Memory Usage", "threshold": "< 85%", "alert_level": "Warning"}, 
                    {"name": "Disk Space", "threshold": "< 90%", "alert_level": "Critical"},
                    {"name": "Network Latency", "threshold": "< 100ms", "alert_level": "Warning"},
                    {"name": "Request Rate", "threshold": "< 1000 req/min", "alert_level": "Info"},
                    {"name": "Error Rate", "threshold": "< 1%", "alert_level": "Critical"}
                ],
                "monitoring_tools": ["Prometheus", "Grafana", "DataDog", "New Relic"],
                "frequency": "Real-time"
            },
            
            "data_quality_monitoring": {
                "description": "Input data quality and characteristics",
                "metrics": [
                    {"name": "Missing Data Rate", "threshold": "< 5%", "alert_level": "Warning"},
                    {"name": "Data Freshness", "threshold": "< 30 minutes delay", "alert_level": "Critical"},
                    {"name": "Schema Changes", "threshold": "No unexpected changes", "alert_level": "Critical"},
                    {"name": "Data Distribution Drift", "threshold": "KL Divergence < 0.1", "alert_level": "Warning"},
                    {"name": "Outlier Rate", "threshold": "< 10%", "alert_level": "Info"},
                    {"name": "Data Volume", "threshold": "Within expected range", "alert_level": "Warning"}
                ],
                "monitoring_tools": ["Great Expectations", "Evidently", "Custom scripts"],
                "frequency": "Every data ingestion cycle"
            },
            
            "model_performance_monitoring": {
                "description": "Model accuracy and prediction quality",
                "metrics": [
                    {"name": "Prediction Accuracy", "threshold": "MAE within 10% of baseline", "alert_level": "Warning"},
                    {"name": "Prediction Latency", "threshold": "< 500ms", "alert_level": "Critical"},
                    {"name": "Model Drift", "threshold": "Performance degradation < 15%", "alert_level": "Warning"},
                    {"name": "Concept Drift", "threshold": "Statistical tests pass", "alert_level": "Info"},
                    {"name": "Uncertainty Calibration", "threshold": "Coverage within expected range", "alert_level": "Info"},
                    {"name": "Prediction Distribution", "threshold": "Within expected bounds", "alert_level": "Warning"}
                ],
                "monitoring_tools": ["MLflow", "Weights & Biases", "Custom dashboards"],
                "frequency": "Daily or after each prediction batch"
            },
            
            "business_metrics_monitoring": {
                "description": "Business impact and value metrics",
                "metrics": [
                    {"name": "Decision Quality", "threshold": "Improvement over baseline", "alert_level": "Info"},
                    {"name": "Cost Savings", "threshold": "Meet expected ROI", "alert_level": "Info"},
                    {"name": "User Satisfaction", "threshold": "Rating > 4.0", "alert_level": "Warning"},
                    {"name": "Adoption Rate", "threshold": "Growing usage", "alert_level": "Info"},
                    {"name": "Business KPI Impact", "threshold": "Positive contribution", "alert_level": "Warning"},
                    {"name": "Forecast Consumption", "threshold": "Predictions being used", "alert_level": "Critical"}
                ],
                "monitoring_tools": ["Business Intelligence tools", "Custom dashboards"],
                "frequency": "Weekly/Monthly"
            }
        }
    
    def _define_alert_thresholds(self) -> Dict[str, Any]:
        """Define alert thresholds and escalation procedures."""
        
        return {
            "alert_levels": {
                "Info": {
                    "description": "Informational, no immediate action required",
                    "response_time": "Next business day",
                    "escalation": "Log only",
                    "example": "Data volume slightly higher than usual"
                },
                "Warning": {
                    "description": "Attention needed, potential issue developing",
                    "response_time": "Within 4 hours",
                    "escalation": "Notify on-call engineer",
                    "example": "Model accuracy declining"
                },
                "Critical": {
                    "description": "Immediate attention required, system impact",
                    "response_time": "Within 30 minutes",
                    "escalation": "Page on-call team, notify management",
                    "example": "Prediction service down"
                }
            },
            
            "escalation_matrix": {
                "Level 1": "On-call Data Scientist",
                "Level 2": "ML Engineering Team Lead", 
                "Level 3": "Engineering Manager",
                "Level 4": "VP of Engineering"
            },
            
            "notification_channels": [
                "PagerDuty for critical alerts",
                "Slack for warnings and info",
                "Email for daily/weekly summaries",
                "Dashboard for real-time visibility"
            ]
        }
    
    def _create_maintenance_schedule(self) -> Dict[str, Any]:
        """Create regular maintenance schedule."""
        
        return {
            "daily_tasks": [
                "Check system health metrics",
                "Review prediction accuracy",
                "Monitor data quality alerts",
                "Verify data pipeline execution"
            ],
            
            "weekly_tasks": [
                "Model performance review",
                "Data drift analysis", 
                "Business metrics assessment",
                "Capacity planning review",
                "Alert threshold evaluation"
            ],
            
            "monthly_tasks": [
                "Full system health audit",
                "Model retraining evaluation",
                "Performance benchmark update",
                "Documentation update",
                "Cost optimization review"
            ],
            
            "quarterly_tasks": [
                "Model architecture review",
                "Technology stack evaluation",
                "Business requirements reassessment",
                "Disaster recovery testing",
                "Security audit"
            ]
        }
    
    def create_monitoring_dashboard_spec(self) -> Dict[str, Any]:
        """Create specification for monitoring dashboard."""
        
        dashboard_spec = {
            "dashboard_name": "Time Series Forecasting Production Monitor",
            
            "panels": [
                {
                    "title": "System Health",
                    "type": "metrics",
                    "metrics": ["CPU", "Memory", "Disk", "Network"],
                    "visualization": "Time series charts",
                    "refresh_rate": "30 seconds"
                },
                
                {
                    "title": "Data Quality",
                    "type": "quality_metrics", 
                    "metrics": ["Missing data %", "Outliers", "Schema compliance"],
                    "visualization": "Gauge charts",
                    "refresh_rate": "5 minutes"
                },
                
                {
                    "title": "Model Performance",
                    "type": "ml_metrics",
                    "metrics": ["MAE", "RMSE", "Prediction latency"],
                    "visualization": "Line charts with thresholds",
                    "refresh_rate": "15 minutes"
                },
                
                {
                    "title": "Business Impact",
                    "type": "business_metrics",
                    "metrics": ["Forecast accuracy", "Decision quality", "Cost impact"],
                    "visualization": "Summary cards and trends",
                    "refresh_rate": "1 hour"
                },
                
                {
                    "title": "Alerts Summary",
                    "type": "alerts",
                    "content": "Recent alerts by severity",
                    "visualization": "Alert table with status",
                    "refresh_rate": "1 minute"
                }
            ],
            
            "user_access": {
                "data_scientists": "Full access",
                "engineers": "Full access", 
                "business_stakeholders": "Business metrics only",
                "executives": "Summary dashboard"
            }
        }
        
        return dashboard_spec
    
    def generate_monitoring_report(self) -> str:
        """Generate monitoring setup report."""
        
        total_metrics = sum(len(layer['metrics']) for layer in self.monitoring_layers.values())
        
        report = f"""
# PRODUCTION MONITORING SETUP REPORT

## Overview
- Monitoring layers: {len(self.monitoring_layers)}
- Total metrics tracked: {total_metrics}
- Alert levels: {len(self.alert_thresholds['alert_levels'])}
- Maintenance frequency: Daily to Quarterly

## Monitoring Coverage
"""
        
        for layer_name, layer_info in self.monitoring_layers.items():
            report += f"""
### {layer_name.replace('_', ' ').title()}
- Description: {layer_info['description']}
- Metrics count: {len(layer_info['metrics'])}
- Monitoring frequency: {layer_info['frequency']}
- Tools: {', '.join(layer_info['monitoring_tools'])}
"""
        
        report += f"""
## Alert Configuration
- Info alerts: {self.alert_thresholds['alert_levels']['Info']['response_time']} response time
- Warning alerts: {self.alert_thresholds['alert_levels']['Warning']['response_time']} response time  
- Critical alerts: {self.alert_thresholds['alert_levels']['Critical']['response_time']} response time

## Maintenance Schedule
- Daily tasks: {len(self.maintenance_schedule['daily_tasks'])} items
- Weekly tasks: {len(self.maintenance_schedule['weekly_tasks'])} items
- Monthly tasks: {len(self.maintenance_schedule['monthly_tasks'])} items
- Quarterly tasks: {len(self.maintenance_schedule['quarterly_tasks'])} items
        """
        
        return report

# Demonstrate monitoring framework
monitoring = ProductionMonitoringFramework()

print("📊 PRODUCTION MONITORING FRAMEWORK")
print("=" * 60)

# Show monitoring layers summary
print(f"\n🔍 MONITORING LAYERS:")
for layer_name, layer_info in monitoring.monitoring_layers.items():
    print(f"\n• {layer_name.replace('_', ' ').title()}")
    print(f"  Metrics: {len(layer_info['metrics'])}")
    print(f"  Frequency: {layer_info['frequency']}")
    print(f"  Tools: {', '.join(layer_info['monitoring_tools'][:2])}")

# Show alert configuration
alert_config = monitoring.alert_thresholds
print(f"\n🚨 ALERT CONFIGURATION:")
for level, config in alert_config['alert_levels'].items():
    print(f"\n• {level}")
    print(f"  Response time: {config['response_time']}")
    print(f"  Escalation: {config['escalation']}")

# Show maintenance schedule
maintenance = monitoring.maintenance_schedule
print(f"\n🔧 MAINTENANCE SCHEDULE:")
print(f"• Daily: {len(maintenance['daily_tasks'])} tasks")
print(f"• Weekly: {len(maintenance['weekly_tasks'])} tasks") 
print(f"• Monthly: {len(maintenance['monthly_tasks'])} tasks")
print(f"• Quarterly: {len(maintenance['quarterly_tasks'])} tasks")

# Show dashboard specification
dashboard_spec = monitoring.create_monitoring_dashboard_spec()
print(f"\n📈 MONITORING DASHBOARD:")
print(f"• Panels: {len(dashboard_spec['panels'])}")
print(f"• User roles: {len(dashboard_spec['user_access'])}")
print(f"• Refresh rates: 30 seconds to 1 hour")

# Generate monitoring report
monitoring_report = monitoring.generate_monitoring_report()
print(f"\n📋 MONITORING SETUP SUMMARY:")
print("Full monitoring report generated with:")
print("• Coverage details for all monitoring layers")
print("• Alert configuration and escalation procedures")
print("• Maintenance schedules and responsibilities")
print("• Dashboard specifications and access controls")

print(f"\n💡 MONITORING BEST PRACTICES:")
print("1. Start with basic system health, then add ML-specific metrics")
print("2. Set alert thresholds based on historical baseline plus buffer")
print("3. Review and adjust thresholds regularly based on experience")
print("4. Ensure alerts are actionable and not just informational")
print("5. Create runbooks for common alert scenarios")
