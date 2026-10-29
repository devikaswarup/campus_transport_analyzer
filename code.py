import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SEED = 42
# Save outputs next to this script, no matter which folder the terminal is in
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:               # e.g. running inside a notebook cell
    BASE_DIR = os.getcwd()
OUT = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")

# ======================================================================
# 1. DATA  (simulated campus bus logs, calibrated to campus patterns)
# ======================================================================
def make_simulated_data(n=6000, seed=SEED):
    """
    Realistic campus bus data:
      - class-time peaks (8-9 AM, 4-6 PM) and a lunch bump
      - route 'personalities' (morning-peak, evening-peak, all-day, low)
      - stop weights, weekend drop, exam-week surges, random noise
    """
    rng = np.random.default_rng(seed)
    routes = [f"R{i}" for i in range(1, 13)]
    personality = {}
    for i, r in enumerate(routes):
        personality[r] = ("morning" if i < 3 else "evening" if i < 6
                          else "allday" if i < 9 else "low")
    stops = {"Main Gate": 1.2, "Hostel A": 1.15, "Hostel B": 1.1,
             "Library": 1.0, "Canteen": 0.9, "Admin Block": 0.8, "Sports Complex": 0.7}
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_factor = {"Mon": 1.0, "Tue": 1.0, "Wed": 1.0, "Thu": 1.0, "Fri": 0.95,
                  "Sat": 0.45, "Sun": 0.30}

    rows = []
    stop_names = list(stops)
    stop_p = np.array(list(stops.values())) / sum(stops.values())
    for _ in range(n):
        r = rng.choice(routes)
        h = int(rng.integers(6, 21))
        m = int(rng.choice([0, 15, 30, 45]))
        d = rng.choice(days)
        s = rng.choice(stop_names, p=stop_p)
        p = personality[r]

        if p == "morning":
            base = 42 if h in (7, 8, 9) else 13
        elif p == "evening":
            base = 42 if h in (16, 17, 18) else 13
        elif p == "allday":
            base = 34 if h in (12, 13) else 28 if 7 <= h <= 18 else 12
        else:  # low
            base = 10 if h in (8, 9, 17, 18) else 7

        base *= day_factor[d] * stops[s]
        if rng.random() < 0.05:          # exam-week surge
            base *= 1.25
        count = rng.normal(base, 0.15 * base + 2)
        count = int(np.clip(count, 1, 60))   # bus capacity cap
        rows.append([r, s, f"{h:02d}:{m:02d}", count, d])
    return pd.DataFrame(rows, columns=["route", "boarding_location", "time",
                                       "passenger_count", "day"])


if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1])
    print(f"Loaded real data: {sys.argv[1]}")
else:
    df = make_simulated_data()
    print("Using SIMULATED campus data (replace with a real CSV when available)")

df.to_csv(f"{OUT}/raw_data.csv", index=False)
print(f"Records: {len(df)} | Routes: {df.route.nunique()} | Stops: {df.boarding_location.nunique()}")

# ======================================================================
# 2. PREPROCESSING
# ======================================================================
df = df.dropna().copy()
df["hour"] = pd.to_datetime(df["time"], format="%H:%M").dt.hour
df["is_weekend"] = df["day"].isin(["Sat", "Sun"]).astype(int)
df["slot"] = pd.cut(df["hour"], bins=[-1, 11, 15, 24], labels=["morning", "midday", "evening"])
df["is_peak"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

# ======================================================================
# 3. FEATURE ENGINEERING  (12 route-level features)
# ======================================================================
g = df.groupby("route")
feat = pd.DataFrame({
    "avg_passengers": g.passenger_count.mean(),
    "max_passengers": g.passenger_count.max(),
    "std_passengers": g.passenger_count.std(),
    "peak_share": g.is_peak.mean(),
    "morning_avg": df[df.slot == "morning"].groupby("route").passenger_count.mean(),
    "midday_avg": df[df.slot == "midday"].groupby("route").passenger_count.mean(),
    "evening_avg": df[df.slot == "evening"].groupby("route").passenger_count.mean(),
    "weekday_avg": df[df.is_weekend == 0].groupby("route").passenger_count.mean(),
    "weekend_avg": df[df.is_weekend == 1].groupby("route").passenger_count.mean(),
    "n_stops": g.boarding_location.nunique(),
}).fillna(0)
feat["weekend_ratio"] = (feat.weekend_avg / feat.weekday_avg.replace(0, np.nan)).fillna(0)
feat["morning_evening_ratio"] = (feat.morning_avg / feat.evening_avg.replace(0, np.nan)).fillna(0)
N_FEATURES = feat.shape[1]
feat.round(2).to_csv(f"{OUT}/route_features.csv")
print(f"\nEngineered {N_FEATURES} route-level features for {len(feat)} routes")

# Scaling
X = StandardScaler().fit_transform(feat)

# ======================================================================
# 4. CHOOSING K  (Elbow + Silhouette)
# ======================================================================
ks = list(range(2, min(7, len(feat))))
inertia, sil = [], []
print("\nK | Inertia | Silhouette")
for k in ks:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    inertia.append(km_k.inertia_)
    sil.append(silhouette_score(X, km_k.labels_))
    print(f"{k} | {inertia[-1]:7.2f} | {sil[-1]:.3f}")
# Pick the smallest K whose silhouette is within 0.01 of the maximum
# (simpler model, avoids splitting similar routes for a tiny score gain)
best_sil = max(sil)
best_k = next(k for k, sc in zip(ks, sil) if sc >= best_sil - 0.01)
print(f"\nChosen K = {best_k} (max silhouette {best_sil:.3f}; smallest K within 0.01 of it)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow Method")
ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, "o-", color="green"); ax[1].axvline(best_k, ls="--", color="red")
ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("K")
plt.tight_layout(); plt.savefig(f"{OUT}/1_elbow_silhouette.png", dpi=150); plt.show()

# ======================================================================
# 5. FINAL K-MEANS MODEL
# ======================================================================
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
feat["cluster"] = km.labels_
final_sil = silhouette_score(X, km.labels_)

# ======================================================================
# 6. INTERPRETATION  (auto-name clusters)
# ======================================================================
prof = feat.groupby("cluster").mean()
overall = feat.avg_passengers.mean()
names = {}
for c, row in prof.iterrows():
    if row.avg_passengers < 0.8 * overall:
        names[c] = "Low-demand routes"
    elif row.morning_avg > 1.4 * row.evening_avg:
        names[c] = "Morning-peak routes"
    elif row.evening_avg > 1.4 * row.morning_avg:
        names[c] = "Evening-peak routes"
    elif row.avg_passengers > 1.2 * overall:
        names[c] = "High-demand all-day routes"
    else:
        names[c] = "Balanced all-day routes"
feat["segment"] = feat.cluster.map(names)

print(f"\nFINAL MODEL: K={best_k}, silhouette={final_sil:.3f}")
print("\n=== Route -> Cluster ===")
print(feat[["cluster", "segment", "avg_passengers", "morning_avg", "evening_avg"]]
      .round(1).sort_values("cluster").to_string())

prof_out = prof.round(2)
prof_out["segment"] = prof_out.index.map(names)
prof_out["n_routes"] = feat.cluster.value_counts().sort_index()
print("\n=== Cluster profiles (means) ===")
print(prof_out[["segment", "n_routes", "avg_passengers", "peak_share", "morning_avg",
                "evening_avg", "weekend_ratio"]].to_string())
feat.round(2).to_csv(f"{OUT}/route_clusters.csv")
prof_out.to_csv(f"{OUT}/cluster_profiles.csv")

# ======================================================================
# 7. PLOTS
# ======================================================================
# (a) PCA scatter
pca = PCA(2, random_state=SEED)
P = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
for c in sorted(feat.cluster.unique()):
    idx = feat.cluster.values == c
    plt.scatter(P[idx, 0], P[idx, 1], s=170, edgecolor="k", label=names[c])
for i, r in enumerate(feat.index):
    plt.annotate(r, (P[i, 0] + 0.07, P[i, 1] + 0.07))
plt.title(f"Route demand clusters (PCA, {pca.explained_variance_ratio_.sum():.0%} variance)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/2_pca_clusters.png", dpi=150); plt.show()

# (b) Route x hour heatmap, routes grouped by cluster
order = feat.sort_values("cluster").index
heat = df.pivot_table(index="route", columns="hour", values="passenger_count",
                      aggfunc="mean").loc[order]
plt.figure(figsize=(11, 5.5))
sns.heatmap(heat, cmap="YlOrRd", cbar_kws={"label": "Avg passengers"})
plt.title("Average passengers: route x hour (routes grouped by cluster)")
plt.tight_layout(); plt.savefig(f"{OUT}/3_route_hour_heatmap.png", dpi=150); plt.show()

# (c) Segment size + hourly demand
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
feat.segment.value_counts().plot.bar(ax=ax[0], color="steelblue")
ax[0].set_title("Routes per segment"); ax[0].tick_params(axis="x", rotation=20)
hourly = (df.merge(feat[["segment"]], left_on="route", right_index=True)
            .groupby(["hour", "segment"]).passenger_count.mean().unstack())
hourly.plot(ax=ax[1], marker="o")
ax[1].set_title("Hourly demand by segment"); ax[1].set_ylabel("Avg passengers")
plt.tight_layout(); plt.savefig(f"{OUT}/4_segment_summary.png", dpi=150); plt.show()

# (d) Cluster feature profile (standardized)
zprof = pd.DataFrame(X, columns=feat.drop(columns=["cluster", "segment"]).columns,
                     index=feat.index).groupby(feat.cluster).mean()
zprof.index = [names[c] for c in zprof.index]
plt.figure(figsize=(12, 4.5))
sns.heatmap(zprof, annot=True, fmt=".1f", cmap="coolwarm", center=0)
plt.title("Cluster feature profile (standardized values)")
plt.tight_layout(); plt.savefig(f"{OUT}/5_cluster_feature_profile.png", dpi=150); plt.show()

# ======================================================================
# 8. SCHEDULING RECOMMENDATIONS
# ======================================================================
tips = {
    "Morning-peak routes": "add buses 07:00-10:00",
    "Evening-peak routes": "add buses 16:00-19:00",
    "Low-demand routes": "reduce frequency or merge trips",
    "Balanced all-day routes": "keep steady frequency",
    "High-demand all-day routes": "increase frequency across the day",
}
print("\n=== SCHEDULING RECOMMENDATIONS ===")
for c, nm in names.items():
    rs = ", ".join(feat[feat.cluster == c].index)
    print(f"- {nm} ({rs}): {tips[nm]}")

# ======================================================================
# 9. RESUME NUMBERS  (paste into resume bullets 2 and 3)
# ======================================================================
sizes = feat.segment.value_counts().to_dict()
print("\n" + "=" * 60)
print("RESUME NUMBERS")
print("=" * 60)
print(f"Records analysed        : {len(df)}")
print(f"Engineered features (N) : {N_FEATURES}")
print(f"Chosen K                : {best_k}")
print(f"Silhouette score        : {final_sil:.2f}")
print(f"Segments                : {sizes}")
print("\nBullet 2: Engineered {} route-level features (peak share, morning/evening "
      "averages, weekend ratio, demand variability) from boarding logs and "
      "standardized them for clustering. Selected K={} using Elbow and Silhouette "
      "analysis (silhouette score {:.2f}).".format(N_FEATURES, best_k, final_sil))
print("Bullet 3: Segmented routes into {} demand profiles ({}) with K-Means and PCA "
      "validation, producing bus-allocation recommendations for peak hours and "
      "low-demand routes.".format(best_k, ", ".join(sorted(set(names.values())))))
print(f"\nAll files saved in: {OUT}")