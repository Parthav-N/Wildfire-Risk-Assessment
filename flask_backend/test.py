"""
Check actual column names in your CSV files
"""
import pandas as pd

print("🔍 Checking CSV columns...\n")

# Check fires
try:
    fires_df = pd.read_csv("D:/wildfire-risk-system/data/fires/fire_archive_SV-C2_716427.csv", nrows=5)
    print("🔥 FIRES columns:")
    print(fires_df.columns.tolist())
    print()
except Exception as e:
    print(f"❌ Fires: {e}\n")

# Check infrastructure
try:
    infra_df = pd.read_csv("D:/wildfire-risk-system/data/infrastructure/all_infrastructure_with_residential.csv", nrows=5)
    print("🏥 INFRASTRUCTURE columns:")
    print(infra_df.columns.tolist())
    print()
except Exception as e:
    print(f"❌ Infrastructure: {e}\n")

# Check training
try:
    training_df = pd.read_csv("D:/wildfire-risk-system/data/training_dataset_final.csv", nrows=5)
    print("📊 TRAINING columns:")
    print(training_df.columns.tolist())
except Exception as e:
    print(f"❌ Training: {e}\n")