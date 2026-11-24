from preprocessing import ROOT, load_data

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = load_data()

# Remove original target column
df = df.drop(columns=["num"], errors="ignore")

output_path = PROCESSED_DIR / "heart_disease_cleaned.csv"

df.to_csv(output_path, index=False)

print("Processed dataset saved successfully.")
print("Dataset shape:", df.shape)
print("Location:", output_path)
print("\nMissing values:")
print(df.isnull().sum())