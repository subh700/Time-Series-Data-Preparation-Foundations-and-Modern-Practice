# Pandas DatetimeIndex approach
data.set_index('timestamp', inplace=True)
data = data.asfreq('D')  # Ensure daily frequency
