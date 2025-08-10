# Handle missing values
df = df.interpolate(method='time')

# Normalization (optional)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df), 
    index=df.index, 
    columns=df.columns
)
