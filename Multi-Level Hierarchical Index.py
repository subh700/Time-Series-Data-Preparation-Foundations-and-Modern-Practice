# For panel data with entities and time
data.set_index(['entity_id', 'timestamp'], inplace=True)
data.sort_index(inplace=True)  # Critical for performance

