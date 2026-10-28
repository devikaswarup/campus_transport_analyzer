import os
import zipfile
import py7zr
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# 1. Extract Archive if needed
os.makedirs("data", exist_ok=True)
archives = [f for f in os.listdir('.') if f.endswith(('.zip', '.7z'))]

if archives and not os.path.exists("data/train.csv"):
    archive_name = archives[0]
    print(f"[+] Extracting {archive_name}...")
    if archive_name.endswith('.zip'):
        with zipfile.ZipFile(archive_name, 'r') as z:
            z.extractall("data")
    elif archive_name.endswith('.7z'):
        with py7zr.SevenZipFile(archive_name, mode='r') as z:
            z.extractall("data")

# 2. Load Datasets
print("[+] Loading CSV datasets...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submission = pd.read_csv("data/sample_submission.csv")

target_col = 'final_seatcount'
id_col = 'route_key'

# 3. Feature Engineering
print("[+] Processing features...")
for df in [train, test]:
    if 'doj' in df.columns:
        df['doj'] = pd.to_datetime(df['doj'])
        df['year'] = df['doj'].dt.year
        df['month'] = df['doj'].dt.month
        df['day'] = df['doj'].dt.day
        df['dayofweek'] = df['doj'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['dayofyear'] = df['doj'].dt.dayofyear
        df.drop(columns=['doj'], inplace=True)

for df in [train, test]:
    if 'srcid' in df.columns and 'destid' in df.columns:
        df['route'] = df['srcid'].astype(str) + "_" + df['destid'].astype(str)

# Target encoding for routes
route_stats = train.groupby('route')[target_col].agg(
    route_mean='mean',
    route_median='median',
    route_std='std',
    route_count='count'
).reset_index()

train = train.merge(route_stats, on='route', how='left')
test = test.merge(route_stats, on='route', how='left')

global_median = train[target_col].median()
test['route_mean'] = test['route_mean'].fillna(global_median)
test['route_median'] = test['route_median'].fillna(global_median)
test['route_std'] = test['route_std'].fillna(0)
test['route_count'] = test['route_count'].fillna(0)

# Categorical handling
cat_cols = [c for c in ['srcid', 'destid', 'route'] if c in train.columns]
for col in cat_cols:
    categories = pd.concat([train[col], test[col]], axis=0).astype(str).unique()
    train[col] = pd.Categorical(train[col].astype(str), categories=categories)
    test[col] = pd.Categorical(test[col].astype(str), categories=categories)

# 4. Prepare Features
X = train.drop(columns=[target_col], errors='ignore')
y = train[target_col]
X_test = test.drop(columns=[id_col], errors='ignore')
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 5. Train Model (5-Fold Cross Validation)
print("[+] Training model...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=42+fold, verbose=-1)
    model.fit(X_tr, y_tr)
    test_preds += np.round(np.clip(model.predict(X_test), 0, None)) / kf.n_splits

# 6. Generate and Save Submission File
os.makedirs("output", exist_ok=True)
final_preds = np.round(np.clip(test_preds, 0, None))

if id_col in test.columns:
    test_results = pd.DataFrame({id_col: test[id_col], target_col: final_preds})
    sub = submission[[id_col]].merge(test_results, on=id_col, how='left')
else:
    sub = submission.copy()
    sub[target_col] = final_preds

submission_path = "output/submission.csv"
sub.to_csv(submission_path, index=False)
print(f"\n[+] SUCCESS! Final predictions saved to: '{submission_path}'")
print("\nFirst 5 rows of output:")
print(sub.head())
import matplotlib.pyplot as plt

# Generate and save feature importance plot
lgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
print("[+] Saved plot to 'output/feature_importance.png'")