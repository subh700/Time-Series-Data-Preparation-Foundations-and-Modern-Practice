# Convert repeated strings to categorical
data['entity_id'] = data['entity_id'].astype('category')
# Memory reduction: ~70% for high-cardinality repeated values
