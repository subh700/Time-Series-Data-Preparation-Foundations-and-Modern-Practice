def standardize_data_types(data, type_hints=None, error_handling='coerce'):
    """
    Intelligent data type standardization for time series data.
    """
    standardized_data = data.copy()
    conversion_log = []
    
    for column in data.columns:
        original_dtype = data[column].dtype
        
        # Apply explicit type hints if provided
        if type_hints and column in type_hints:
            target_type = type_hints[column]
            try:
                if target_type == 'datetime':
                    standardized_data[column] = pd.to_datetime(
                        data[column], 
                        errors=error_handling,
                        infer_datetime_format=True
                    )
                elif target_type == 'numeric':
                    standardized_data[column] = pd.to_numeric(
                        data[column], 
                        errors=error_handling
                    )
                elif target_type == 'category':
                    standardized_data[column] = data[column].astype('category')
                
                conversion_log.append(f"{column}: {original_dtype} -> {target_type}")
                
            except Exception as e:
                conversion_log.append(f"{column}: conversion failed - {str(e)}")
                continue
        
        # Automatic type inference for unspecified columns
        else:
            inferred_type = infer_optimal_dtype(data[column])
            
            if inferred_type != original_dtype:
                try:
                    standardized_data[column] = data[column].astype(inferred_type)
                    conversion_log.append(f"{column}: {original_dtype} -> {inferred_type}")
                except Exception as e:
                    conversion_log.append(f"{column}: auto-conversion failed - {str(e)}")
    
    return standardized_data, conversion_log

def infer_optimal_dtype(series):
    """Infer the optimal data type for a pandas Series."""
    # Skip if already optimal
    if series.dtype in ['datetime64[ns]', 'category']:
        return series.dtype
    
    # Try datetime conversion if string-like
    if series.dtype == 'object':
        # Sample a few values to test datetime conversion
        sample_values = series.dropna().head(100)
        try:
            pd.to_datetime(sample_values, errors='raise')
            return 'datetime64[ns]'
        except:
            pass
        
        # Check if categorical (low cardinality)
        unique_ratio = len(series.unique()) / len(series.dropna())
        if unique_ratio < 0.1 and len(series.unique()) < 1000:
            return 'category'
    
    # Try numeric conversion
    if series.dtype == 'object':
        try:
            pd.to_numeric(series.dropna().head(100), errors='raise')
            # Determine if integer or float
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.dropna().apply(lambda x: x.is_integer()).all():
                # Check if can fit in smaller integer types
                min_val = numeric_series.min()
                max_val = numeric_series.max()
                
                if min_val >= -128 and max_val <= 127:
                    return 'int8'
                elif min_val >= -32768 and max_val <= 32767:
                    return 'int16'
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return 'int32'
                else:
                    return 'int64'
            else:
                return 'float64'
        except:
            return 'object'  # Keep as object if conversion fails
    
    return series.dtype  # Return current dtype if no optimization found
