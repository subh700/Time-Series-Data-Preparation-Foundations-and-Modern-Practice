class UnitStandardizer:
    """Standardize units across time series data."""
    
    def __init__(self):
        self.unit_conversions = {
            'temperature': {
                'fahrenheit_to_celsius': lambda f: (f - 32) * 5/9,
                'kelvin_to_celsius': lambda k: k - 273.15,
                'celsius_to_fahrenheit': lambda c: c * 9/5 + 32,
                'celsius_to_kelvin': lambda c: c + 273.15,
            },
            'pressure': {
                'psi_to_pascal': lambda psi: psi * 6894.76,
                'bar_to_pascal': lambda bar: bar * 100000,
                'mmhg_to_pascal': lambda mmhg: mmhg * 133.322,
            },
            'distance': {
                'feet_to_meters': lambda ft: ft * 0.3048,
                'inches_to_meters': lambda inch: inch * 0.0254,
                'miles_to_meters': lambda mi: mi * 1609.34,
            },
            'volume': {
                'gallons_to_liters': lambda gal: gal * 3.78541,
                'cubic_feet_to_liters': lambda cf: cf * 28.3168,
            }
        }
    
    def detect_unit(self, data, column_name, unit_hint=None):
        """Detect the unit of measurement for a given column."""
        if unit_hint:
            return unit_hint
        
        # Unit detection heuristics based on column name and value ranges
        column_lower = column_name.lower()
        values = data.dropna()
        
        if any(temp_word in column_lower for temp_word in ['temp', 'temperature']):
            # Temperature unit detection based on typical ranges
            min_val, max_val = values.min(), values.max()
            
            if min_val >= -50 and max_val <= 50:
                return 'celsius'
            elif min_val >= 200 and max_val <= 400:
                return 'kelvin'
            elif min_val >= -50 and max_val <= 120:
                return 'fahrenheit'
        
        elif any(press_word in column_lower for press_word in ['pressure', 'press']):
            min_val, max_val = values.min(), values.max()
            
            if min_val >= 900 and max_val <= 1100:
                return 'hpa'  # Atmospheric pressure in hPa
            elif min_val >= 0 and max_val <= 50:
                return 'psi'
            elif min_val >= 90000 and max_val <= 110000:
                return 'pascal'
        
        return 'unknown'
    
    def standardize_units(self, data, column_units, target_units):
        """
        Standardize units across specified columns.
        
        Args:
            data: DataFrame with time series data
            column_units: Dict mapping column names to their current units
            target_units: Dict mapping column names to desired units
        """
        standardized_data = data.copy()
        conversion_log = []
        
        for column, current_unit in column_units.items():
            if column not in data.columns:
                continue
                
            target_unit = target_units.get(column, current_unit)
            
            if current_unit == target_unit:
                continue
            
            # Find appropriate conversion
            conversion_key = f"{current_unit}_to_{target_unit}"
            
            # Search through unit conversion categories
            conversion_func = None
            for category, conversions in self.unit_conversions.items():
                if conversion_key in conversions:
                    conversion_func = conversions[conversion_key]
                    break
            
            if conversion_func:
                try:
                    standardized_data[column] = conversion_func(data[column])
                    conversion_log.append(f"{column}: {current_unit} -> {target_unit}")
                except Exception as e:
                    conversion_log.append(f"{column}: conversion failed - {str(e)}")
            else:
                conversion_log.append(f"{column}: no conversion available for {current_unit} -> {target_unit}")
        
        return standardized_data, conversion_log
