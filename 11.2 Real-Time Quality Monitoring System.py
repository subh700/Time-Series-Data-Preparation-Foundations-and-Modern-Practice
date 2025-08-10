class RealTimeQualityMonitor:
    """
    Real-time monitoring system for time series data quality.
    """
    
    def __init__(self, quality_framework, alert_thresholds=None, history_window='30D'):
        self.quality_framework = quality_framework
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.history_window = history_window
        self.quality_history = []
        self.alert_history = []
        
    def _default_thresholds(self):
        """Default alert thresholds for quality dimensions."""
        return {
            'overall_score': {'critical': 0.6, 'warning': 0.8},
            'completeness': {'critical': 0.7, 'warning': 0.9},
            'timeliness': {'critical': 0.6, 'warning': 0.8},
            'consistency': {'critical': 0.7, 'warning': 0.9},
            'accuracy': {'critical': 0.8, 'warning': 0.95}
        }
    
    def monitor_batch(self, data, timestamp_col='timestamp', value_cols=None, 
                     metadata=None):
        """
        Monitor a batch of data and generate alerts if necessary.
        """
        
        # Assess quality
        quality_result = self.quality_framework.assess_quality(
            data, timestamp_col, value_cols
        )
        
        # Add metadata
        quality_result['batch_metadata'] = metadata or {}
        quality_result['batch_size'] = len(data)
        quality_result['batch_timespan'] = self._calculate_timespan(data, timestamp_col)
        
        # Store in history
        self._update_history(quality_result)
        
        # Check for alerts
        alerts = self._check_alerts(quality_result)
        
        # Generate monitoring report
        monitoring_report = {
            'quality_assessment': quality_result,
            'alerts': alerts,
            'trends': self._analyze_trends(),
            'recommendations': self._generate_recommendations(quality_result, alerts)
        }
        
        return monitoring_report
    
    def _calculate_timespan(self, data, timestamp_col):
        """Calculate timespan of data batch."""
        timestamps = pd.to_datetime(data[timestamp_col])
        return {
            'start': timestamps.min(),
            'end': timestamps.max(),
            'duration': timestamps.max() - timestamps.min()
        }
    
    def _update_history(self, quality_result):
        """Update quality monitoring history."""
        self.quality_history.append(quality_result)
        
        # Limit history size (keep only recent entries)
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(self.history_window)
        self.quality_history = [
            result for result in self.quality_history
            if result['assessment_timestamp'] >= cutoff_time
        ]
    
    def _check_alerts(self, quality_result):
        """Check for quality alerts based on thresholds."""
        alerts = []
        
        overall_score = quality_result['overall_score']
        dimension_scores = quality_result['dimension_scores']
        
        # Check overall score
        if overall_score <= self.alert_thresholds['overall_score']['critical']:
            alerts.append({
                'type': 'critical',
                'dimension': 'overall',
                'score': overall_score,
                'threshold': self.alert_thresholds['overall_score']['critical'],
                'message': f"Critical: Overall data quality score ({overall_score:.3f}) below critical threshold"
            })
        elif overall_score <= self.alert_thresholds['overall_score']['warning']:
            alerts.append({
                'type': 'warning',
                'dimension': 'overall',
                'score': overall_score,
                'threshold': self.alert_thresholds['overall_score']['warning'],
                'message': f"Warning: Overall data quality score ({overall_score:.3f}) below warning threshold"
            })
        
        # Check dimension scores
        for dimension, score in dimension_scores.items():
            if dimension in self.alert_thresholds:
                thresholds = self.alert_thresholds[dimension]
                
                if score <= thresholds['critical']:
                    alerts.append({
                        'type': 'critical',
                        'dimension': dimension,
                        'score': score,
                        'threshold': thresholds['critical'],
                        'message': f"Critical: {dimension.title()} score ({score:.3f}) below critical threshold"
                    })
                elif score <= thresholds['warning']:
                    alerts.append({
                        'type': 'warning',
                        'dimension': dimension,
                        'score': score,
                        'threshold': thresholds['warning'],
                        'message': f"Warning: {dimension.title()} score ({score:.3f}) below warning threshold"
                    })
        
        # Store alerts in history
        for alert in alerts:
            alert['timestamp'] = pd.Timestamp.now()
            self.alert_history.append(alert)
        
        return alerts
    
    def _analyze_trends(self):
        """Analyze quality trends over time."""
        if len(self.quality_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Extract time series of quality scores
        timestamps = [result['assessment_timestamp'] for result in self.quality_history]
        overall_scores = [result['overall_score'] for result in self.quality_history]
        
        # Calculate trend
        from scipy.stats import linregress
        
        x = np.arange(len(overall_scores))
        slope, intercept, r_value, p_value, std_err = linregress(x, overall_scores)
        
        # Trend analysis
        trend_direction = 'improving' if slope > 0.01 else 'degrading' if slope < -0.01 else 'stable'
        trend_strength = abs(r_value)
        
        # Recent vs. historical comparison
        recent_scores = overall_scores[-5:] if len(overall_scores) >= 5 else overall_scores
        historical_scores = overall_scores[:-5] if len(overall_scores) >= 10 else overall_scores[:-len(recent_scores)]
        
        if historical_scores:
            recent_avg = np.mean(recent_scores)
            historical_avg = np.mean(historical_scores)
            relative_change = (recent_avg - historical_avg) / historical_avg if historical_avg != 0 else 0
        else:
            relative_change = 0
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'trend_slope': slope,
            'trend_significance': p_value,
            'recent_vs_historical': {
                'relative_change': relative_change,
                'recent_average': np.mean(recent_scores),
                'historical_average': np.mean(historical_scores) if historical_scores else None
            }
        }
    
    def _generate_recommendations(self, quality_result, alerts):
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []
        
        # Overall quality recommendations
        if quality_result['overall_score'] < 0.8:
            recommendations.append({
                'priority': 'high',
                'category': 'overall',
                'action': 'Implement comprehensive data quality improvement plan',
                'details': 'Overall data quality is below acceptable threshold'
            })
        
        # Dimension-specific recommendations
        dimension_scores = quality_result['dimension_scores']
        
        if dimension_scores.get('completeness', 1.0) < 0.8:
            recommendations.append({
                'priority': 'high',
                'category': 'completeness',
                'action': 'Investigate data collection processes',
                'details': 'High levels of missing data detected'
            })
        
        if dimension_scores.get('timeliness', 1.0) < 0.7:
            recommendations.append({
                'priority': 'medium',
                'category': 'timeliness',
                'action': 'Review data pipeline latency',
                'details': 'Data arrival delays or irregular patterns detected'
            })
        
        if dimension_scores.get('consistency', 1.0) < 0.8:
            recommendations.append({
                'priority': 'medium',
                'category': 'consistency',
                'action': 'Standardize data formats and validation rules',
                'details': 'Inconsistencies in data format or distribution detected'
            })
        
        # Alert-based recommendations
        critical_alerts = [alert for alert in alerts if alert['type'] == 'critical']
        if critical_alerts:
            recommendations.append({
                'priority': 'critical',
                'category': 'alerts',
                'action': 'Immediate investigation required',
                'details': f'{len(critical_alerts)} critical quality alerts triggered'
            })
        
        return recommendations
    
    def generate_quality_dashboard(self):
        """Generate dashboard data for quality monitoring visualization."""
        
        if not self.quality_history:
            return {'status': 'no_data'}
        
        # Time series data for charts
        timestamps = [result['assessment_timestamp'] for result in self.quality_history]
        overall_scores = [result['overall_score'] for result in self.quality_history]
        
        # Dimension score time series
        dimension_time_series = {}
        if self.quality_history:
            dimensions = self.quality_history[0]['dimension_scores'].keys()
            for dim in dimensions:
                dimension_time_series[dim] = [
                    result['dimension_scores'].get(dim, 0) for result in self.quality_history
                ]
        
        # Recent alerts summary
        recent_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] >= pd.Timestamp.now() - pd.Timedelta('7D')
        ]
        
        alert_summary = {
            'critical_count': len([a for a in recent_alerts if a['type'] == 'critical']),
            'warning_count': len([a for a in recent_alerts if a['type'] == 'warning']),
            'total_count': len(recent_alerts)
        }
        
        # Current status
        current_quality = self.quality_history[-1] if self.quality_history else None
        
        return {
            'current_status': {
                'overall_score': current_quality['overall_score'] if current_quality else None,
                'dimension_scores': current_quality['dimension_scores'] if current_quality else {},
                'last_assessment': current_quality['assessment_timestamp'] if current_quality else None
            },
            'time_series': {
                'timestamps': timestamps,
                'overall_scores': overall_scores,
                'dimension_scores': dimension_time_series
            },
            'alerts': {
                'recent_summary': alert_summary,
                'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
            },
            'trends': self._analyze_trends()
        }
