# Respect temporal order
n = len(df)
train_size = int(0.6 * n)
val_size = int(0.2 * n)

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size+val_size]
test_data = df.iloc[train_size+val_size:]
