import pandas as pd

# Load the main archive file
df = pd.read_csv("data/fires/fire_archive_SV-C2_716427.csv")

print("="*70)
print("CALIFORNIA FIRE DATA (2017-2024)")
print("="*70)

print(f"\n📊 Total fire detections: {len(df):,}")
print(f"📅 Date range: {df['acq_date'].min()} to {df['acq_date'].max()}")

# Parse dates
df['date'] = pd.to_datetime(df['acq_date'])
df['year'] = df['date'].dt.year

# Fires per year
print("\n🔥 Detections by year:")
for year in sorted(df['year'].unique()):
    count = len(df[df['year'] == year])
    print(f"   {year}: {count:,}")

# High confidence fires
high_conf = df[df['confidence'] == 'h']
print(f"\n✅ High confidence fires: {len(high_conf):,} ({len(high_conf)/len(df)*100:.1f}%)")

# Major fire events (cluster detections by location and date)
print("\n🔍 Sample of data:")
print(df[['latitude', 'longitude', 'acq_date', 'acq_time', 'frp', 'confidence']].head(10))

print("\n" + "="*70)