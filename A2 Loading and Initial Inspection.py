# Standard loading pattern
df = pd.read_csv('dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.sort_index()
