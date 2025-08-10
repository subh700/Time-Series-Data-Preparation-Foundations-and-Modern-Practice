class FinancialDataQualityAssessor:
    """Specialized quality assessment for financial time series data."""
    
    def __init__(self):
        self.trading_hours = {
            'NYSE': {'start': '09:30', 'end': '16:00', 'timezone': 'America/New_York'},
            'LSE': {'start': '08:00', 'end': '16:30', 'timezone': 'Europe/London'},
            'TSE': {'start': '09:00', 'end': '15:00', 'timezone': 'Asia/Tokyo'}
        }
        
    def assess_financial_quality(self, data, price_col='close', volume_col='volume', 
                                exchange='NYSE'):
        """
        Comprehensive quality assessment for financial data.
        """
        
        quality_results = {
            'price_quality': self._assess_price_quality(data, price_col),
            'volume_quality': self._assess_volume_quality(data, volume_col),
            'market_hours_compliance': self._assess_market_hours(data, exchange),
            'price_volume_relationship': self._assess_price_volume_relationship(
                data, price_col, volume_col
            ),
            'market_microstructure': self._assess_microstructure_quality(
                data, price_col, volume_col
            )
        }
        
        # Calculate weighted overall score
        weights = {
            'price_quality': 0.3,
            'volume_quality': 0.2,
            'market_hours_compliance': 0.1,
            'price_volume_relationship': 0.2,
            'market_microstructure': 0.2
        }
        
        overall_score = sum(
            quality_results[dimension]['score'] * weight
            for dimension, weight in weights.items()
            if 'score' in quality_results[dimension]
        )
        
        return {
            'overall_score': overall_score,
            'detailed_results': quality_results,
            'assessment_type': 'financial'
        }
    
    def _assess_price_quality(self, data, price_col):
        """Assess quality of price data."""
        
        prices = pd.to_numeric(data[price_col], errors='coerce')
        
        # Basic price validation
        negative_prices = (prices < 0).sum()
        zero_prices = (prices == 0).sum()
        missing_prices = prices.isnull().sum()
        
        # Price movement validation
        price_changes = prices.pct_change()
        extreme_movements = (abs(price_changes) > 0.2).sum()  # >20% moves
        
        # Price continuity (gaps)
        price_gaps = self._detect_price_gaps(prices)
        
        # Calculate quality score
        total_records = len(prices)
        quality_issues = negative_prices + zero_prices + missing_prices + extreme_movements
        quality_score = max(0, 1 - (quality_issues / total_records))
        
        return {
            'score': quality_score,
            'negative_prices': negative_prices,
            'zero_prices': zero_prices,
            'missing_prices': missing_prices,
            'extreme_movements': extreme_movements,
            'price_gaps': len(price_gaps),
            'price_range': (prices.min(), prices.max())
        }
    
    def _assess_volume_quality(self, data, volume_col):
        """Assess quality of volume data."""
        
        if volume_col not in data.columns:
            return {'score': 0.0, 'reason': 'Volume data not available'}
        
        volumes = pd.to_numeric(data[volume_col], errors='coerce')
        
        # Volume validation
        negative_volumes = (volumes < 0).sum()
        zero_volumes = (volumes == 0).sum()
        missing_volumes = volumes.isnull().sum()
        
        # Volume patterns
        volume_spikes = self._detect_volume_spikes(volumes)
        
        # Calculate quality score
        total_records = len(volumes)
        quality_issues = negative_volumes + missing_volumes
        
        # Zero volumes might be acceptable (e.g., after hours)
        if zero_volumes / total_records > 0.5:  # Too many zero volumes
            quality_issues += zero_volumes * 0.5
        
        quality_score = max(0, 1 - (quality_issues / total_records))
        
        return {
            'score': quality_score,
            'negative_volumes': negative_volumes,
            'zero_volumes': zero_volumes,
            'missing_volumes': missing_volumes,
            'volume_spikes': len(volume_spikes),
            'volume_statistics': {
                'mean': volumes.mean(),
                'median': volumes.median(),
                'std': volumes.std()
            }
        }
    
    def _assess_market_hours(self, data, exchange):
        """Assess compliance with market hours."""
        
        if 'timestamp' not in data.columns:
            return {'score': 0.5, 'reason': 'No timestamp column for market hours analysis'}
        
        timestamps = pd.to_datetime(data['timestamp'])
        
        if exchange not in self.trading_hours:
            return {'score': 0.5, 'reason': f'Unknown exchange: {exchange}'}
        
        market_config = self.trading_hours[exchange]
        
        # Convert to market timezone
        market_tz = market_config['timezone']
        market_timestamps = timestamps.dt.tz_convert(market_tz)
        
        # Define market hours
        market_start = pd.to_datetime(market_config['start']).time()
        market_end = pd.to_datetime(market_config['end']).time()
        
        # Check market hours compliance
        in_market_hours = (
            (market_timestamps.dt.time >= market_start) &
            (market_timestamps.dt.time <= market_end) &
            (market_timestamps.dt.weekday < 5)  # Monday to Friday
        )
        
        compliance_rate = in_market_hours.sum() / len(timestamps)
        
        return {
            'score': compliance_rate,
            'compliance_rate': compliance_rate,
            'records_in_market_hours': in_market_hours.sum(),
            'records_outside_market_hours': (~in_market_hours).sum(),
            'exchange': exchange
        }
    
    def _assess_price_volume_relationship(self, data, price_col, volume_col):
        """Assess the relationship between price and volume."""
        
        if volume_col not in data.columns:
            return {'score': 0.5, 'reason': 'Volume data not available'}
        
        prices = pd.to_numeric(data[price_col], errors='coerce')
        volumes = pd.to_numeric(data[volume_col], errors='coerce')
        
        # Calculate price changes and volume
        price_changes = prices.pct_change().abs()
        
        # Remove invalid data
        valid_mask = ~(prices.isnull() | volumes.isnull() | price_changes.isnull())
        valid_price_changes = price_changes[valid_mask]
        valid_volumes = volumes[valid_mask]
        
        if len(valid_price_changes) < 10:
            return {'score': 0.5, 'reason': 'Insufficient valid data'}
        
        # Calculate correlation between price changes and volume
        correlation = valid_price_changes.corr(valid_volumes)
        
        # Higher correlation suggests better quality relationship
        # (large price moves should correspond to high volume)
        correlation_score = abs(correlation) if not pd.isna(correlation) else 0
        
        return {
            'score': correlation_score,
            'price_volume_correlation': correlation,
            'valid_observations': len(valid_price_changes)
        }
    
    def _assess_microstructure_quality(self, data, price_col, volume_col):
        """Assess market microstructure quality indicators."""
        
        prices = pd.to_numeric(data[price_col], errors='coerce')
        
        # Bid-ask spread estimation (using price volatility as proxy)
        returns = prices.pct_change()
        volatility = returns.std()
        
        # Tick size analysis
        price_increments = prices.diff().dropna()
        price_increments = price_increments[price_increments != 0]
        
        if len(price_increments) > 0:
            min_increment = price_increments.abs().min()
            tick_regularity = (price_increments.abs() % min_increment == 0).mean()
        else:
            tick_regularity = 0
        
        # Price impact analysis (simplified)
        if volume_col in data.columns:
            volumes = pd.to_numeric(data[volume_col], errors='coerce')
            volume_returns = volumes.pct_change()
            
            # Price impact should be correlated with volume changes
            if len(returns.dropna()) > 0 and len(volume_returns.dropna()) > 0:
                impact_correlation = abs(returns.corr(volume_returns))
                if pd.isna(impact_correlation):
                    impact_correlation = 0
            else:
                impact_correlation = 0
        else:
            impact_correlation = 0.5
        
        # Combine microstructure indicators
        microstructure_score = (tick_regularity * 0.5 + impact_correlation * 0.5)
        
        return {
            'score': microstructure_score,
            'volatility': volatility,
            'tick_regularity': tick_regularity,
            'price_impact_correlation': impact_correlation,
            'min_price_increment': min_increment if len(price_increments) > 0 else None
        }
    
    def _detect_price_gaps(self, prices):
        """Detect significant price gaps."""
        price_changes = prices.pct_change()
        # Define gap as price change > 10%
        gaps = price_changes[abs(price_changes) > 0.1]
        return gaps.dropna()
    
    def _detect_volume_spikes(self, volumes):
        """Detect volume spikes (unusually high volume)."""
        volume_mean = volumes.mean()
        volume_std = volumes.std()
        
        # Define spike as volume > mean + 3*std
        threshold = volume_mean + 3 * volume_std
        spikes = volumes[volumes > threshold]
        
        return spikes.dropna()
