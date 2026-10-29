
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix, adjusted_rand_score)

APP_VERSION = "CampusPulse v3 (interactive charts)"
SEED = 42
REQUIRED = ["route", "boarding_location", "time", "passenger_count", "day"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIPS = {
    "Morning-peak routes": "add buses 07:00-10:00",
    "Evening-peak routes": "add buses 16:00-19:00",
    "Low-demand routes": "reduce frequency or merge trips",
    "Balanced all-day routes": "keep steady frequency",
    "High-demand all-day routes": "increase frequency across the day",
}
sns.set_theme(style="whitegrid")


# ======================================================================
# SHARED LOGIC (used by both the script and the dashboard)
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
    day_factor = {"Mon": 1.0, "Tue": 1.0, "Wed": 1.0, "Thu": 1.0, "Fri": 0.95,
                  "Sat": 0.45, "Sun": 0.30}

    rows = []
    stop_names = list(stops)
    stop_p = np.array(list(stops.values())) / sum(stops.values())
    for _ in range(n):
        r = rng.choice(routes)
        h = int(rng.integers(6, 21))
        m = int(rng.choice([0, 15, 30, 45]))
        d = rng.choice(DAY_ORDER)
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
    return pd.DataFrame(rows, columns=REQUIRED)


def preprocess(df):
    """Clean the data and add hour, is_weekend, slot and is_peak columns."""
    df = df.dropna().copy()
    df["hour"] = pd.to_datetime(df["time"].astype(str).str[:5], format="%H:%M").dt.hour
    df["is_weekend"] = df["day"].isin(["Sat", "Sun"]).astype(int)
    df["slot"] = pd.cut(df["hour"], bins=[-1, 11, 15, 24], labels=["morning", "midday", "evening"])
    df["is_peak"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    return df


def build_features(df):
    """One row per route with 12 engineered features."""
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
    return feat


def evaluate_k(X, n_routes):
    """Try K=2..6; return the K list, inertia, silhouette and the auto-chosen K
    (smallest K whose silhouette is within 0.01 of the maximum)."""
    ks = list(range(2, min(7, n_routes)))
    inertia, sil = [], []
    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        inertia.append(m.inertia_)
        sil.append(silhouette_score(X, m.labels_))
    best_sil = max(sil)
    best_k = next(k for k, sc in zip(ks, sil) if sc >= best_sil - 0.01)
    return ks, inertia, sil, best_k


def name_clusters(feat):
    """Auto-name each cluster from its average profile."""
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
    return names, prof



# ======================================================================
# PREDICTION MODELS (supervised: these DO have accuracy)
# ======================================================================
FULL_BUS_LOAD = 45                   # avg boardings treated as a "full" bus
LEVELS = ["Light", "Moderate", "Crowded"]


def crowding_level(counts):
    """Turn boardings into a crowding level: <40% Light, <75% Moderate, else Crowded."""
    return pd.cut(np.asarray(counts, dtype=float) / FULL_BUS_LOAD,
                  bins=[-np.inf, 0.40, 0.75, np.inf], labels=LEVELS).astype(str)


def train_demand_models(df, seed=SEED):
    """Predict passenger_count from route, stop, day and hour.
    Compares a baseline, Linear Regression and Random Forest on a 20% hold-out test set."""
    cols = ["route", "boarding_location", "day", "hour"]
    X = df[cols].copy()
    X["hour"] = X["hour"].astype(str)                 # hour treated as a category
    y = df["passenger_count"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)

    def pre():
        return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cols)])

    models = {
        "Baseline (average)": make_pipeline(pre(), DummyRegressor()),
        "Linear Regression": make_pipeline(pre(), LinearRegression()),
        "Random Forest": make_pipeline(pre(), RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, random_state=seed, n_jobs=1)),
    }
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        rows.append({"Model": name,
                     "MAE": mean_absolute_error(yte, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
                     "R2": r2_score(yte, p),
                     "Crowding accuracy": accuracy_score(crowding_level(yte), crowding_level(p))})
    results = pd.DataFrame(rows)

    rf = models["Random Forest"]
    cm = confusion_matrix(crowding_level(yte), crowding_level(preds["Random Forest"]), labels=LEVELS)
    names_out = rf[0].get_feature_names_out()
    imp = pd.Series(rf[-1].feature_importances_, index=names_out)
    groups = {c: float(imp[[n for n in names_out if n.startswith("c__" + c + "_")]].sum())
              for c in cols}
    importance = pd.Series(groups).sort_values()

    final = make_pipeline(pre(), RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3, random_state=seed, n_jobs=1)).fit(X, y)
    return {"results": results, "y_test": np.asarray(yte), "y_pred": preds["Random Forest"],
            "X_test": Xte.reset_index(drop=True),
            "confusion": cm, "importance": importance, "final": final}


def build_forecast_lookup(model, routes, stops, days, hours):
    """Predicted average boardings for every (route, day, hour)."""
    import itertools
    rows = list(itertools.product(routes, stops, days, [str(h) for h in hours]))
    X = pd.DataFrame(rows, columns=["route", "boarding_location", "day", "hour"])
    X["pred"] = model.predict(X)
    X["hour"] = X["hour"].astype(int)
    return X.groupby(["route", "day", "hour"]).pred.mean().to_dict()


def predict_with_range(model, route, stop, day, hour):
    """Predicted boardings plus a 10th-90th percentile range across the forest's trees."""
    X = pd.DataFrame([{"route": route, "boarding_location": stop, "day": day, "hour": str(hour)}])
    mean = float(model.predict(X)[0])
    Xt = model[0].transform(X)
    trees = np.array([t.predict(Xt)[0] for t in model[-1].estimators_])
    return mean, float(np.percentile(trees, 10)), float(np.percentile(trees, 90))


def forecast_curve(route, fc, day, hours):
    """Predicted load % by hour for one route on one day."""
    return pd.Series([float(np.clip(fc.get((route, day, h), np.nan) / FULL_BUS_LOAD * 100, 0, 100))
                      for h in hours], index=list(hours))


def light_windows(curve, thr=40):
    """Hour ranges where predicted load stays below thr %."""
    out, start, prev = [], None, None
    for h, v in curve.items():
        if v < thr:
            start = h if start is None else start
            prev = h
        elif start is not None:
            out.append(f"{start:02d}:00-{prev + 1:02d}:00"); start = None
    if start is not None:
        out.append(f"{start:02d}:00-{prev + 1:02d}:00")
    return out


def planted_personality():
    """Hidden route behaviours used by make_simulated_data (ground truth for validation)."""
    routes = [f"R{i}" for i in range(1, 13)]
    return {r: ("morning" if i < 3 else "evening" if i < 6 else "allday" if i < 9 else "low")
            for i, r in enumerate(routes)}


def cluster_recovery(routes, labels):
    """Compare clusters with the planted behaviours (only valid for simulated data)."""
    truth = planted_personality()
    if set(routes) != set(truth):
        return None
    y = [truth[r] for r in routes]
    ari = adjusted_rand_score(y, labels)
    ct = pd.crosstab(pd.Series(list(labels), name="c"), pd.Series(y, name="t"))
    return ari, float(ct.max(axis=1).sum() / ct.values.sum())


# ======================================================================
# LIVE TRACKER HELPERS (simulated GPS on LONG-STRETCH routes)
# ======================================================================
CAMPUS_CENTER = (10.0000, 76.3000)   # (lat, lon) of the campus hub -> EDIT to your campus
HUB = "Campus Main Gate"
STATUS_COLORS = {"Light": [31, 157, 98], "Moderate": [224, 161, 0], "Crowded": [216, 69, 58]}
STATUS_HEX = {"Light": "#1f9d62", "Moderate": "#e0a100", "Crowded": "#d8453a"}
# Corridors leaving the campus: (compass bearing in degrees, [(stop, km from campus), ...]).
# Sample long-distance network (up to 25 km) - rename the stops to your real places.
CORRIDORS = [
    (0,   [("Hostel Colony", 2.5), ("Market Junction", 6), ("Bus Stand", 10)]),
    (60,  [("Medical College", 3), ("Railway Station", 8), ("Airport Road", 16)]),
    (120, [("Tech Park", 4), ("IT Corridor", 9), ("Beach Road", 18)]),
    (180, [("Temple Junction", 3), ("Old Town", 7), ("Town Hall", 12)]),
    (240, [("Sports Village", 2), ("Lake View", 6), ("Industrial Estate", 22)]),
    (300, [("Stadium", 3), ("City Hospital", 8), ("Suburb Terminal", 25)]),
]


def km_to_latlon(km, bearing_deg, center=CAMPUS_CENTER):
    b = np.radians(bearing_deg)
    return (center[0] + km * np.cos(b) / 111.0,
            center[1] + km * np.sin(b) / (111.0 * np.cos(np.radians(center[0]))))


def dist_km(p, q):
    dlat = (p[0] - q[0]) * 111.0
    dlon = (p[1] - q[1]) * 111.0 * np.cos(np.radians(CAMPUS_CENTER[0]))
    return float(np.hypot(dlat, dlon))


def build_network(routes):
    """Give every route a long corridor from the campus hub out to the city.
    Even-numbered routes run the full corridor, odd ones a shorter variant."""
    stop_xy = {HUB: CAMPUS_CENTER}
    route_stops, route_km = {}, {}
    for i, r in enumerate(routes):
        bearing, stops = CORRIDORS[(i // 2) % len(CORRIDORS)]
        use = stops if i % 2 == 0 else stops[:2]
        names = [HUB]
        for nm, km in use:
            stop_xy[nm] = km_to_latlon(km, bearing)
            names.append(nm)
        route_stops[r] = names
        route_km[r] = use[-1][1]
    return stop_xy, route_stops, route_km


def cycle_geometry(pts, stop_xy):
    """Out-and-back cycle of stops, distance to the start of each leg and total km."""
    cyc = pts + pts[-2:0:-1]
    n = len(cyc)
    legs = [dist_km(stop_xy[cyc[j]], stop_xy[cyc[(j + 1) % n]]) for j in range(n)]
    cum = np.concatenate([[0.0], np.cumsum(legs)])
    return cyc, legs, cum, float(cum[-1])


def live_snapshot(routes, segments, stop_xy, route_stops, route_km, fc, overall, sim_min,
                  day, hour_range, kmph=30):
    """Position, ETA and PREDICTED crowding of one bus per route at the simulated time."""
    hour = int(sim_min // 60) % 24
    if not (hour_range[0] <= hour <= hour_range[1]):
        return pd.DataFrame()                      # off duty
    rows = []
    for i, r in enumerate(routes):
        cyc, legs, cum, total = cycle_geometry(route_stops[r], stop_xy)
        d = (sim_min / 60 * kmph + i / len(routes) * total) % total
        k = int(np.searchsorted(cum, d, side="right") - 1)
        frac = (d - cum[k]) / legs[k] if legs[k] > 0 else 0
        a, b = stop_xy[cyc[k]], stop_xy[cyc[(k + 1) % len(cyc)]]
        lat = a[0] + (b[0] - a[0]) * frac
        lon = a[1] + (b[1] - a[1]) * frac
        eta = legs[k] * (1 - frac) / kmph * 60
        base = fc.get((r, day, hour), overall.get(r, 0))
        jitter = np.random.default_rng(int(sim_min) * 100 + i).normal(0, 0.04)
        load = float(np.clip(base / FULL_BUS_LOAD + jitter, 0.03, 1.0))
        status = "Crowded" if load >= 0.75 else "Moderate" if load >= 0.40 else "Light"
        rows.append({"route": r, "segment": segments.get(r, ""), "lat": lat, "lon": lon,
                     "pos_km": d, "route_km": route_km[r],
                     "next_stop": cyc[(k + 1) % len(cyc)], "eta_min": round(eta, 1),
                     "load_pct": round(load * 100), "status": status,
                     "color": STATUS_COLORS[status], "color_hex": STATUS_HEX[status]})
    return pd.DataFrame(rows)


def arrivals_at_stop(snap, route_stops, stop_xy, stop, kmph=30):
    """Running buses that serve `stop`, with minutes until each arrives."""
    rows = []
    for _, bus in snap.iterrows():
        cyc, legs, cum, total = cycle_geometry(route_stops[bus.route], stop_xy)
        if stop in cyc:
            ahead = min((cum[j] - bus.pos_km) % total for j, nm in enumerate(cyc) if nm == stop)
            rows.append({"route": bus.route, "eta_min": round(ahead / kmph * 60, 1),
                         "status": bus.status, "load_pct": bus.load_pct,
                         "route_km": bus.route_km, "segment": bus.segment})
    out = pd.DataFrame(rows, columns=["route", "eta_min", "status", "load_pct",
                                      "route_km", "segment"])
    return out.sort_values("eta_min").reset_index(drop=True)


def best_bus_tip(arr, window=30):
    """Plain-English boarding advice from the arrivals table."""
    cand = arr[arr.eta_min <= window]
    if cand.empty:
        return f"No bus arrives at this stop within {window} minutes."
    first = cand.iloc[0]
    lightest = cand.sort_values(["load_pct", "eta_min"]).iloc[0]
    if (lightest.route != first.route and first.load_pct >= 75
            and lightest.load_pct <= first.load_pct - 20):
        return (f"{first.route} arrives first in {first.eta_min} min but is {first.load_pct}% full. "
                f"Waiting for {lightest.route} ({lightest.eta_min} min, {lightest.load_pct}% full) "
                f"gives a more comfortable ride.")
    return (f"Take {first.route}: arrives in {first.eta_min} min, "
            f"{first.status.lower()} ({first.load_pct}% full).")


# ======================================================================
# MODE 1: ANALYSIS SCRIPT  (python campus_transport_analyzer.py)
# ======================================================================
def run_script():
    print(APP_VERSION)
    # Save outputs next to this script, no matter which folder the terminal is in
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:               # e.g. running inside a notebook cell
        base_dir = os.getcwd()
    OUT = os.path.join(base_dir, "outputs")
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. DATA
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        print(f"Loaded real data: {sys.argv[1]}")
    else:
        df = make_simulated_data()
        print("Using SIMULATED campus data (replace with a real CSV when available)")
    df.to_csv(f"{OUT}/raw_data.csv", index=False)
    print(f"Records: {len(df)} | Routes: {df.route.nunique()} | Stops: {df.boarding_location.nunique()}")

    # ---- 2. PREPROCESSING
    df = preprocess(df)

    # ---- 3. FEATURE ENGINEERING (12 route-level features)
    feat = build_features(df)
    N_FEATURES = feat.shape[1]
    feat.round(2).to_csv(f"{OUT}/route_features.csv")
    print(f"\nEngineered {N_FEATURES} route-level features for {len(feat)} routes")
    X = StandardScaler().fit_transform(feat)

    # ---- 4. CHOOSING K (Elbow + Silhouette)
    ks, inertia, sil, best_k = evaluate_k(X, len(feat))
    print("\nK | Inertia | Silhouette")
    for k, i, s in zip(ks, inertia, sil):
        print(f"{k} | {i:7.2f} | {s:.3f}")
    print(f"\nChosen K = {best_k} (max silhouette {max(sil):.3f}; smallest K within 0.01 of it)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
    ax[1].plot(ks, sil, "o-", color="green"); ax[1].axvline(best_k, ls="--", color="red")
    ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("K")
    plt.tight_layout(); plt.savefig(f"{OUT}/1_elbow_silhouette.png", dpi=150); plt.show()

    # ---- 5. FINAL K-MEANS MODEL
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    feat["cluster"] = km.labels_
    final_sil = silhouette_score(X, km.labels_)

    # ---- 6. INTERPRETATION
    names, prof = name_clusters(feat)
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

    # ---- 7. PLOTS
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

    # ---- 8. SCHEDULING RECOMMENDATIONS
    print("\n=== SCHEDULING RECOMMENDATIONS ===")
    for c, nm in names.items():
        rs = ", ".join(feat[feat.cluster == c].index)
        print(f"- {nm} ({rs}): {TIPS[nm]}")

    # ---- 8b. DEMAND PREDICTION MODELS (supervised, with real accuracy)
    pm = train_demand_models(df)
    print("\n=== DEMAND PREDICTION (80/20 train-test split) ===")
    print(pm["results"].round(3).to_string(index=False))
    rf_row = pm["results"].set_index("Model").loc["Random Forest"]
    print(f"\nRandom Forest: MAE {rf_row.MAE:.2f} passengers, RMSE {rf_row.RMSE:.2f}, "
          f"R2 {rf_row.R2:.3f}, crowding-level accuracy {rf_row['Crowding accuracy']:.1%}")
    print("Note: clustering (K-Means) has no accuracy because it is unsupervised; "
          "its quality is the silhouette score.")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(pm["y_test"], pm["y_pred"], s=8, alpha=0.35)
    lim = [0, max(pm["y_test"].max(), pm["y_pred"].max())]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Actual passengers"); ax[0].set_ylabel("Predicted")
    ax[0].set_title("Random Forest: actual vs predicted")
    sns.heatmap(pm["confusion"], annot=True, fmt="d", cmap="Blues", xticklabels=LEVELS,
                yticklabels=LEVELS, ax=ax[1])
    ax[1].set_xlabel("Predicted level"); ax[1].set_ylabel("Actual level")
    ax[1].set_title("Crowding-level confusion matrix")
    pm["importance"].plot.barh(ax=ax[2], color="teal"); ax[2].set_title("What drives demand")
    plt.tight_layout(); plt.savefig(f"{OUT}/6_prediction_accuracy.png", dpi=150); plt.show()
    pm["results"].round(3).to_csv(f"{OUT}/prediction_results.csv", index=False)

    if len(sys.argv) == 1:                       # simulated data has hidden ground truth
        rec = cluster_recovery(list(feat.index), km.labels_)
        if rec:
            print(f"\nCluster check vs hidden route behaviours (simulated only): "
                  f"ARI {rec[0]:.2f}, purity {rec[1]:.0%}")

    # ---- 9. RESUME NUMBERS
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
    print("Optional Bullet 4: Trained and compared Linear Regression and Random Forest to "
          "forecast hourly boardings (MAE {:.1f} passengers, R2 {:.2f}, {:.0%} accuracy on "
          "crowding level) on a held-out test set.".format(
              rf_row.MAE, rf_row.R2, rf_row["Crowding accuracy"]))
    print(f"\nAll files saved in: {OUT}")
    print("Tip: for the interactive dashboard run:  "
          "python -m streamlit run campus_transport_analyzer.py")


# ======================================================================
# MODE 2: DASHBOARD "CampusPulse"  (python -m streamlit run campus_transport_analyzer.py)
# Original design: a departure-board + route-lane control room (not a map-first taxi app).
# ======================================================================
import html as _html

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');
.cp-banner{background:linear-gradient(120deg,#0b3c49 0%,#0e7c86 60%,#2bb3a3 100%);color:#fff;padding:22px 28px;border-radius:18px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px}
.cp-banner h1{margin:0;padding:0;font-family:'Space Grotesk',system-ui,sans-serif;font-size:34px;letter-spacing:-0.5px;color:#fff}
.cp-banner p{margin:4px 0 0;opacity:.88;font-size:14px;color:#fff}
.cp-chip{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.4);padding:5px 13px;border-radius:999px;font-size:13px;color:#fff;margin:3px 0 3px 8px;white-space:nowrap}
.cp-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:6px 0 16px}
.cp-kpi{background:#fff;border:1px solid #e2dccf;border-radius:14px;padding:14px 16px;color:#1d2b36}
.cp-kpi .l{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:#6b7782}
.cp-kpi .v{font-size:28px;font-weight:700;font-family:'Space Grotesk',system-ui,sans-serif;color:#0b3c49}
.cp-kpi .s{font-size:12px;color:#6b7782}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700;color:#fff}
.pill.Light{background:#1f9d62}.pill.Moderate{background:#e0a100;color:#222}.pill.Crowded{background:#d8453a}
.board{background:#101820;border-radius:14px;padding:12px 16px;color:#ffd166;font-family:'Courier New',monospace}
.board .bt{font-size:13px;letter-spacing:.12em;color:#8fb8c2;margin-bottom:6px}
.board .br{display:grid;grid-template-columns:72px 56px 1fr 64px;gap:8px;padding:7px 2px;border-bottom:1px dashed #2c3a47;font-size:15px;align-items:center}
.board .bh{color:#8fb8c2;font-size:11px;letter-spacing:.1em}
.tip{background:#e7f4f1;border-left:5px solid #0e7c86;border-radius:10px;padding:10px 14px;margin-top:10px;color:#12343b;font-size:14px}
.lanes{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:12px;margin:6px 0 14px}
.lane{background:#fff;border:1px solid #e2dccf;border-radius:14px;padding:14px 18px 12px;color:#1d2b36}
.lane.sel{border:2px solid #0e7c86;box-shadow:0 0 0 4px rgba(14,124,134,.15)}
.lane .lh{display:flex;justify-content:space-between;align-items:center;gap:8px}
.lane .rt{font-family:'Space Grotesk',system-ui,sans-serif;font-size:20px;font-weight:700;margin-right:6px;color:#0b3c49}
.lane .mu{font-size:12px;color:#6b7782}
.track{position:relative;height:6px;background:#e5e0d4;border-radius:3px;margin:30px 46px 40px}
.stopdot{position:absolute;top:-5px;width:14px;height:14px;border-radius:50%;background:#fff;border:3px solid #0b3c49;transform:translateX(-50%)}
.stoplbl{position:absolute;top:16px;transform:translateX(-50%);font-size:11px;color:#55616c;white-space:nowrap}
.busdot{position:absolute;top:-15px;transform:translateX(-50%);font-size:11px;font-weight:700;color:#fff;padding:3px 9px;border-radius:10px;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.3)}
.loadbar{height:8px;background:#e5e0d4;border-radius:4px;overflow:hidden}
.loadbar div{height:100%}
.lane .lf{display:flex;justify-content:space-between;font-size:12px;color:#55616c;margin-top:8px;gap:8px;flex-wrap:wrap}
.alert{background:#fdecea;border-left:5px solid #d8453a;border-radius:10px;padding:10px 14px;margin:6px 0;color:#4a1612;font-size:14px}
</style>"""


def kpi_html(label, value, sub=""):
    return (f'<div class="cp-kpi"><div class="l">{_html.escape(str(label))}</div>'
            f'<div class="v">{_html.escape(str(value))}</div>'
            f'<div class="s">{_html.escape(str(sub))}</div></div>')


def board_html(stop, arr, route_stops):
    """Station-style departure board for one stop."""
    e = lambda x: _html.escape(str(x))
    rows = ""
    for _, a in arr.head(8).iterrows():
        eta = "Due" if a.eta_min < 1 else f"{a.eta_min:.0f} min"
        rows += (f'<div class="br"><span>{eta}</span><span>{e(a.route)}</span>'
                 f'<span>to {e(route_stops[a.route][-1])}</span>'
                 f'<span class="pill {a.status}">{a.load_pct}%</span></div>')
    if not rows:
        rows = '<div class="br"><span>No bus serves this stop right now</span></div>'
    return (f'<div class="board"><div class="bt">DEPARTURES | {e(stop).upper()}</div>'
            f'<div class="br bh"><span>ARRIVES</span><span>ROUTE</span><span>TOWARDS</span>'
            f'<span>LOAD</span></div>{rows}</div>')


def lane_html(r, row, route_stops, stop_xy, route_km, fc, day, hour, selected):
    """One route as a horizontal lane: stops on a line, the bus as a moving badge."""
    e = lambda x: _html.escape(str(x))
    pts = route_stops[r]
    L = route_km[r]
    _, _, _, total = cycle_geometry(pts, stop_xy)
    one_way = total / 2
    d = row.pos_km
    x, arrow = (d / one_way, "\u2192") if d <= one_way else ((total - d) / one_way, "\u2190")
    x = min(max(x, 0.0), 1.0) * 100
    marks = ""
    for nm in pts:
        f = dist_km(stop_xy[HUB], stop_xy[nm]) / L * 100
        marks += (f'<div class="stopdot" style="left:{f:.1f}%"></div>'
                  f'<div class="stoplbl" style="left:{f:.1f}%">{e(nm)}</div>')
    nxt = fc.get((r, day, hour + 1))
    if nxt is None or nxt != nxt:
        fc_txt = "n/a"
    else:
        pct = float(np.clip(nxt / FULL_BUS_LOAD * 100, 0, 100))
        trend = "\u2191" if pct > row.load_pct + 5 else "\u2193" if pct < row.load_pct - 5 else "\u2192"
        fc_txt = f"{pct:.0f}% {trend}"
    col = STATUS_HEX[row.status]
    return (f'<div class="lane{" sel" if selected else ""}"><div class="lh"><div>'
            f'<span class="rt">{e(r)}</span><span class="pill {row.status}">{row.status} {row.load_pct}%</span>'
            f'</div><div class="mu">{e(row.segment)} | {L:g} km</div></div>'
            f'<div class="track">{marks}<div class="busdot" style="left:{x:.1f}%;background:{col}">'
            f'{e(r)} {arrow}</div></div>'
            f'<div class="loadbar"><div style="width:{row.load_pct}%;background:{col}"></div></div>'
            f'<div class="lf"><span>Next: {e(row.next_stop)} in {row.eta_min} min</span>'
            f'<span>Forecast next hour: {fc_txt}</span></div></div>')


def _alt():
    import altair as alt
    return alt


def show_chart(st, make, key=None, select=None, fallback=None):
    """Draw an interactive (hover / tap / zoom) Altair chart. Never crashes the page.
    If `select` is a selection name, returns the rows the user tapped."""
    try:
        chart = make()
        attempts = [{"width": "stretch"}, {"use_container_width": True}, {}]
        if select:
            for kw in attempts:
                try:
                    ev = st.altair_chart(chart, key=key, on_select="rerun", **kw)
                except TypeError:
                    continue
                try:
                    sel = ev.selection if hasattr(ev, "selection") else ev["selection"]
                    got = sel.get(select) if hasattr(sel, "get") else sel[select]
                except Exception:
                    got = None
                return got if isinstance(got, list) else []
        for kw in attempts:
            try:
                st.altair_chart(chart, **kw)
                return []
            except TypeError:
                continue
        return []
    except Exception:
        if fallback is not None:
            st.caption("Interactive chart not available here, showing the data instead.")
            st.dataframe(fallback)
        return []


def picked_value(rows, field="route"):
    if rows and isinstance(rows[0], dict):
        return rows[0].get(field)
    return None


def chart_network(snap, stop_xy, route_stops, route_km, sel_route, sel_stop):
    """Touch map: tap a bus to follow its route. Hover shows details."""
    alt = _alt()
    paths = pd.DataFrame([{"route": r, "order": i, "lon": stop_xy[nm][1], "lat": stop_xy[nm][0],
                           "km": route_km[r]}
                          for r, pts in route_stops.items() for i, nm in enumerate(pts)])
    stops = pd.DataFrame([{"name": n, "lat": xy[0], "lon": xy[1], "hl": int(n == sel_stop),
                           "hls": str(int(n == sel_stop))} for n, xy in stop_xy.items()])
    buses = snap[["route", "segment", "lat", "lon", "status", "load_pct", "next_stop",
                  "eta_min", "route_km"]].copy()
    kw = {"value": [{"route": sel_route}]} if sel_route else {}
    pick = alt.selection_point(name="pick", fields=["route"], **kw)
    x = alt.X("lon:Q", axis=None, scale=alt.Scale(zero=False, padding=30))
    y = alt.Y("lat:Q", axis=None, scale=alt.Scale(zero=False, padding=30))
    lines = alt.Chart(paths).mark_line(strokeWidth=3).encode(
        x=x, y=y, detail="route:N", order="order:Q",
        color=alt.condition(pick, alt.value("#0e7c86"), alt.value("#b9b3a6")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.55)),
        tooltip=[alt.Tooltip("route:N", title="route"), alt.Tooltip("km:Q", title="length km")])
    dots = alt.Chart(stops).mark_circle(stroke="white", strokeWidth=1.5).encode(
        x=x, y=y,
        size=alt.Size("hl:Q", scale=alt.Scale(domain=[0, 1], range=[70, 420]), legend=None),
        color=alt.Color("hls:N", scale=alt.Scale(domain=["0", "1"], range=["#1d2b36", "#e67e22"]),
                        legend=None),
        tooltip=[alt.Tooltip("name:N", title="stop")])
    names_ = alt.Chart(stops).mark_text(dy=-13, fontSize=10, color="#55616c").encode(
        x=x, y=y, text="name:N")
    bus = alt.Chart(buses).mark_circle(size=420, stroke="white", strokeWidth=2).encode(
        x=x, y=y,
        color=alt.Color("status:N", scale=alt.Scale(domain=LEVELS, range=[STATUS_HEX[l] for l in LEVELS]),
                        legend=alt.Legend(title="Load", orient="bottom")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.4)),
        tooltip=[alt.Tooltip("route:N"), alt.Tooltip("segment:N"), alt.Tooltip("status:N"),
                 alt.Tooltip("load_pct:Q", title="load %"), alt.Tooltip("next_stop:N", title="next stop"),
                 alt.Tooltip("eta_min:Q", title="ETA min"), alt.Tooltip("route_km:Q", title="route km")]
    ).add_params(pick)
    lab = alt.Chart(buses).mark_text(fontSize=10, fontWeight="bold", color="white").encode(
        x=x, y=y, text="route:N")
    return alt.layer(lines, dots, names_, bus, lab).properties(height=470).configure_view(strokeWidth=0)


def chart_k(ks, inertia, sil, K):
    alt = _alt()
    d = pd.DataFrame({"K": ks, "Inertia": inertia, "Silhouette": sil})
    rule = alt.Chart(pd.DataFrame({"K": [K]})).mark_rule(color="#d8453a", strokeDash=[4, 4]).encode(x="K:O")
    a = alt.Chart(d).mark_line(point=True).encode(
        x="K:O", y=alt.Y("Inertia:Q"), tooltip=["K", alt.Tooltip("Inertia:Q", format=".1f")]
    ).properties(title="Elbow (inertia)", height=260)
    b = alt.Chart(d).mark_line(point=True, color="#1f9d62").encode(
        x="K:O", y=alt.Y("Silhouette:Q"), tooltip=["K", alt.Tooltip("Silhouette:Q", format=".3f")]
    ).properties(title="Silhouette score", height=260)
    return alt.hconcat(a + rule, b + rule)


def chart_pca(P, feat, names, var):
    """Tap a route dot to see its profile; hover for details."""
    alt = _alt()
    d = pd.DataFrame({"route": list(feat.index), "PC1": P[:, 0], "PC2": P[:, 1],
                      "segment": [names[c] for c in feat.cluster],
                      "avg_passengers": feat.avg_passengers.round(1).values})
    pick = alt.selection_point(name="pick", fields=["route"])
    pts = alt.Chart(d).mark_circle(size=380, stroke="white", strokeWidth=2).encode(
        x=alt.X("PC1:Q", title="PC1"), y=alt.Y("PC2:Q", title=f"PC2 ({var:.0%} of variance shown)"),
        color=alt.Color("segment:N", legend=alt.Legend(orient="bottom")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.4)),
        tooltip=["route", "segment", alt.Tooltip("avg_passengers:Q", title="avg passengers")]
    ).add_params(pick)
    lab = alt.Chart(d).mark_text(dy=-16, fontSize=11).encode(x="PC1:Q", y="PC2:Q", text="route:N")
    return (pts + lab).properties(height=380).interactive()


def chart_heat(d, x, y, v, vtitle, order=None, domain=None, height=330):
    alt = _alt()
    scale = alt.Scale(scheme="yelloworangered", domain=domain) if domain else alt.Scale(scheme="yelloworangered")
    return alt.Chart(d).mark_rect().encode(
        x=alt.X(f"{x}:O", title=x), y=alt.Y(f"{y}:N", sort=order, title=y),
        color=alt.Color(f"{v}:Q", scale=scale, title=vtitle),
        tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:O"), alt.Tooltip(f"{v}:Q", format=".1f")]
    ).properties(height=height)


def chart_lines(d, x, y, color, ytitle, rule_x=None, height=320):
    """Click a legend entry to show or hide a line; hover for values."""
    alt = _alt()
    leg = alt.selection_point(fields=[color], bind="legend")
    c = alt.Chart(d).mark_line(point=True).encode(
        x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:Q", title=ytitle), color=f"{color}:N",
        opacity=alt.condition(leg, alt.value(1), alt.value(0.15)),
        tooltip=[alt.Tooltip(f"{color}:N"), alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q", format=".1f")]
    ).add_params(leg).properties(height=height)
    if rule_x is not None:
        c = c + alt.Chart(pd.DataFrame({x: [rule_x]})).mark_rule(
            color="#d8453a", strokeDash=[5, 4]).encode(x=f"{x}:Q")
    return c


def chart_bar(d, x, y, ytitle, horizontal=False, color="#0e7c86", height=300):
    alt = _alt()
    if horizontal:
        enc = dict(y=alt.Y(f"{x}:N", sort="-x", title=None), x=alt.X(f"{y}:Q", title=ytitle))
    else:
        enc = dict(x=alt.X(f"{x}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                   y=alt.Y(f"{y}:Q", title=ytitle))
    return alt.Chart(d).mark_bar(color=color).encode(
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=".2f")], **enc).properties(height=height)


def chart_scatter_pred(d):
    alt = _alt()
    lim = float(max(d.actual.max(), d.predicted.max()))
    pts = alt.Chart(d).mark_circle(size=22, opacity=0.35, color="#0e7c86").encode(
        x=alt.X("actual:Q", title="Actual passengers"), y=alt.Y("predicted:Q", title="Predicted"),
        tooltip=["route", "day", "hour", "actual", alt.Tooltip("predicted:Q", format=".1f")])
    diag = alt.Chart(pd.DataFrame({"a": [0, lim], "b": [0, lim]})).mark_line(
        color="#d8453a", strokeDash=[5, 4]).encode(x="a:Q", y="b:Q")
    return (pts + diag).properties(height=320).interactive()


def chart_confusion(cm):
    alt = _alt()
    d = pd.DataFrame([{"actual": LEVELS[i], "predicted": LEVELS[j], "count": int(cm[i][j])}
                      for i in range(len(LEVELS)) for j in range(len(LEVELS))])
    base = alt.Chart(d).encode(x=alt.X("predicted:N", sort=LEVELS), y=alt.Y("actual:N", sort=LEVELS))
    rect = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                   tooltip=["actual", "predicted", "count"])
    txt = base.mark_text(fontSize=16).encode(text="count:Q")
    return (rect + txt).properties(height=320)


def run_dashboard():
    import streamlit as st

    st.set_page_config(page_title="CampusPulse", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    cached_sim = st.cache_data(make_simulated_data)
    cached_train = st.cache_data(train_demand_models)

    # ---------------------------------------------------------- SIDEBAR
    st.sidebar.title("Controls")
    st.sidebar.caption(APP_VERSION)
    source = st.sidebar.radio("Data source", ["Simulated data", "Upload CSV"])
    if source == "Upload CSV":
        up = st.sidebar.file_uploader("CSV with columns: " + ", ".join(REQUIRED), type="csv")
        if up is None:
            st.info("Upload a CSV in the sidebar, or switch to Simulated data.")
            st.stop()
        raw = pd.read_csv(up)
        missing = [c for c in REQUIRED if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
    else:
        raw = cached_sim()

    df = preprocess(raw)
    df["slot"] = df["slot"].astype(str)
    feat = build_features(df)
    if len(feat) < 4:
        st.error("Need at least 4 routes to cluster.")
        st.stop()
    X = StandardScaler().fit_transform(feat)

    ks, inertia, sil, auto_k = evaluate_k(X, len(feat))
    auto = st.sidebar.checkbox("Choose K automatically", value=True)
    K = auto_k if auto else st.sidebar.slider("Number of clusters (K)", ks[0], ks[-1], auto_k)

    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
    feat["cluster"] = km.labels_
    final_sil = silhouette_score(X, km.labels_)
    names, _ = name_clusters(feat)
    feat["segment"] = feat.cluster.map(names)
    df = df.merge(feat[["cluster", "segment"]], left_on="route", right_index=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Live board")
    play = st.sidebar.checkbox("Live updates (every 2 s)", value=True)
    speed = st.sidebar.slider("Simulation speed (minutes per update)", 1, 15, 5)
    kmph = st.sidebar.slider("Average bus speed (km/h)", 20, 60, 30)
    live_day = st.sidebar.selectbox("Day", DAY_ORDER, index=0)
    start_hour = st.sidebar.slider("Start hour", 6, 20, 7)
    if st.session_state.get("start_key") != (start_hour, live_day):
        st.session_state["start_key"] = (start_hour, live_day)
        st.session_state["sim_min"] = start_hour * 60

    # ------------------------------------------- prediction model + network
    models = cached_train(df)
    rf = models["results"].set_index("Model").loc["Random Forest"]
    routes_sorted = sorted(feat.index, key=lambda r: (len(r), r))
    stop_xy, route_stops, route_km = build_network(routes_sorted)
    stop_names = [HUB] + sorted(n for n in stop_xy if n != HUB)
    data_stops = sorted(df.boarding_location.unique())
    data_days = [d for d in DAY_ORDER if d in set(df.day)]
    hour_range = (int(df.hour.min()), int(df.hour.max()))
    hours = list(range(hour_range[0], hour_range[1] + 1))
    fc = build_forecast_lookup(models["final"], routes_sorted, data_stops, data_days, hours)
    overall = df.groupby("route").passenger_count.mean().to_dict()
    seg_map = feat.segment.to_dict()
    net_curve = pd.concat([forecast_curve(r, fc, live_day, hours) for r in routes_sorted],
                          axis=1).mean(axis=1)
    peak_txt, peak_sub = "n/a", "no data for this day"
    if not net_curve.isna().all():
        peak_txt = f"{int(net_curve.idxmax()):02d}:00"
        peak_sub = f"{net_curve.max():.0f}% network load forecast"
    is_sim = source == "Simulated data"
    seg_names = sorted(set(names.values()))

    # ------------------------------------------------------------ BANNER
    chips = (f'<span class="cp-chip">{"Simulated data" if is_sim else "Your uploaded data"}</span>'
             f'<span class="cp-chip">{K} demand segments</span>'
             f'<span class="cp-chip">silhouette {final_sil:.2f}</span>'
             f'<span class="cp-chip">forecast accuracy {rf["Crowding accuracy"]:.0%}</span>'
             f'<span class="cp-chip">v3 interactive</span>')
    st.markdown('<div class="cp-banner"><div><h1>CampusPulse</h1>'
                '<p>Tap, hover and explore: live demand, crowding and forecasts for long-stretch campus routes</p>'
                f'</div><div>{chips}</div></div>', unsafe_allow_html=True)

    tab_live, tab_pred, tab_acc, tab_cl, tab_dem, tab_data = st.tabs(
        ["Live Board", "Predict", "Accuracy", "Clusters", "Demand patterns", "Data & Downloads"])

    # -------------------------------------------------------- LIVE BOARD
    with tab_live:
        c1, c2, c3 = st.columns([2, 2, 3])
        sel_route = c1.selectbox("Follow a route", ["All routes"] + routes_sorted)
        sel_stop = c2.selectbox("Departures from", ["Network view"] + stop_names)
        extra = c3.slider("What-if: add extra buses to the followed route", 0, 3, 0)
        sel_route = None if sel_route == "All routes" else sel_route
        sel_stop = None if sel_stop == "Network view" else sel_stop
        j0, j1, j2, j3, j4 = st.columns([2, 1, 1, 1, 2])
        j0.markdown("**Jump to a time of day**")
        if j1.button("Morning peak"):
            st.session_state["sim_min"] = 8 * 60
        if j2.button("Midday"):
            st.session_state["sim_min"] = 13 * 60
        if j3.button("Evening peak"):
            st.session_state["sim_min"] = 17 * 60
        street = j4.toggle("Street map view")

        def draw_street_map(snap, eff_route):
            hi = [14, 124, 134]
            if sel_stop:
                (lat0, lon0), span = stop_xy[sel_stop], 6.0
            elif eff_route:
                pts = [stop_xy[x] for x in route_stops[eff_route]]
                lat0 = float(np.mean([q[0] for q in pts])); lon0 = float(np.mean([q[1] for q in pts]))
                span = max(route_km[eff_route], 4.0)
            else:
                (lat0, lon0), span = CAMPUS_CENTER, max(route_km.values())
            zoom = float(np.clip(np.log2(57700 / span), 9, 14))
            try:
                import pydeck as pdk
                paths = pd.DataFrame({
                    "name": routes_sorted,
                    "path": [[[stop_xy[x][1], stop_xy[x][0]] for x in route_stops[r]] for r in routes_sorted],
                    "color": [hi + [255] if r == eff_route else [120, 120, 120, 40 if eff_route else 110]
                              for r in routes_sorted],
                    "width": [6 if r == eff_route else 2 for r in routes_sorted]})
                stops_df = pd.DataFrame({
                    "name": stop_names, "lat": [stop_xy[x][0] for x in stop_names],
                    "lon": [stop_xy[x][1] for x in stop_names],
                    "radius": [14 if x == sel_stop else 6 for x in stop_names],
                    "color": [hi + [255] if x == sel_stop else [30, 30, 30, 255] for x in stop_names]})
                bus = snap.copy()
                bus["rgba"] = [STATUS_COLORS[st_] + [255 if (eff_route is None or r == eff_route) else 70]
                               for st_, r in zip(bus.status, bus.route)]
                bus["radius"] = [14 if r == eff_route else 9 for r in bus.route]
                layers = [
                    pdk.Layer("PathLayer", paths, get_path="path", get_color="color",
                              get_width="width", width_units="pixels"),
                    pdk.Layer("ScatterplotLayer", stops_df, get_position="[lon, lat]",
                              get_fill_color="color", get_radius="radius", radius_units="pixels"),
                    pdk.Layer("TextLayer", stops_df, get_position="[lon, lat]", get_text="name",
                              get_size=12, get_color=[40, 40, 40], get_pixel_offset=[0, 16]),
                    pdk.Layer("ScatterplotLayer", bus, get_position="[lon, lat]", get_fill_color="rgba",
                              get_radius="radius", radius_units="pixels", pickable=True),
                    pdk.Layer("TextLayer", bus, get_position="[lon, lat]", get_text="route",
                              get_size=14, get_color=[0, 0, 0], get_pixel_offset=[0, -18])]
                st.pydeck_chart(pdk.Deck(
                    layers=layers, map_provider="carto", map_style="light",
                    initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom),
                    tooltip={"text": "{route} | {status} {load_pct}%\nNext: {next_stop} in {eta_min} min"}))
            except Exception:
                st.map(snap, latitude="lat", longitude="lon", color="color_hex", size=60)

        def live_panel():
            if play:
                st.session_state["sim_min"] = st.session_state["sim_min"] + speed
            sim_min = st.session_state["sim_min"]
            hh, mm = int(sim_min // 60) % 24, int(sim_min % 60)
            snap = live_snapshot(routes_sorted, seg_map, stop_xy, route_stops, route_km, fc,
                                 overall, sim_min, live_day, hour_range, kmph)
            clock = kpi_html("Simulated time", f"{live_day} {hh:02d}:{mm:02d}",
                             "clock running" if play else "paused")
            if snap.empty:
                st.markdown(f'<div class="cp-kpis">{clock}</div>', unsafe_allow_html=True)
                st.warning("Outside service hours for this data. Use a Jump button or raise the start hour.")
                return

            # what-if: extra buses share the load of the followed route
            follow = sel_route or st.session_state.get("picked_route")
            note = ""
            if follow and extra > 0 and follow in set(snap.route):
                i = snap.index[snap.route == follow][0]
                old = int(snap.at[i, "load_pct"])
                new = int(round(old / (1 + extra)))
                lvl = "Crowded" if new >= 75 else "Moderate" if new >= 40 else "Light"
                snap.at[i, "load_pct"] = new
                snap.at[i, "status"] = lvl
                snap.at[i, "color_hex"] = STATUS_HEX[lvl]
                note = f"What-if: +{extra} bus(es) on {follow} lowers its load from {old}% to {new}% ({lvl})."

            kp = clock + "".join([
                kpi_html("Buses running", len(snap), f"{kmph} km/h average"),
                kpi_html("Crowded now", int((snap.status == "Crowded").sum()), "above 75% load"),
                kpi_html("Network load", f"{snap.load_pct.mean():.0f}%", "average over routes"),
                kpi_html("Predicted peak", peak_txt, peak_sub),
                kpi_html("Forecast accuracy", f"{rf['Crowding accuracy']:.0%}",
                         f"MAE {rf.MAE:.1f} | R\u00b2 {rf.R2:.2f}")])
            st.markdown(f'<div class="cp-kpis">{kp}</div>', unsafe_allow_html=True)
            if note:
                st.markdown(f'<div class="tip">{_html.escape(note)}</div>', unsafe_allow_html=True)

            left, right = st.columns([3, 2])
            eff = sel_route
            with left:
                if street:
                    draw_street_map(snap, follow)
                else:
                    rows = show_chart(
                        st, lambda: chart_network(snap, stop_xy, route_stops, route_km, sel_route, sel_stop),
                        key="net_map", select="pick",
                        fallback=snap[["route", "status", "load_pct", "next_stop", "eta_min"]])
                    got = picked_value(rows)
                    st.session_state["picked_route"] = got
                    eff = sel_route or got
                    st.caption("Tap a bus to follow its route. Hover any bus, stop or line for details. "
                               "Orange stop = your chosen departure stop.")
            with right:
                bstop = sel_stop or HUB
                arr = arrivals_at_stop(snap, route_stops, stop_xy, bstop, kmph)
                st.markdown(board_html(bstop, arr, route_stops), unsafe_allow_html=True)
                if not arr.empty:
                    st.markdown(f'<div class="tip">{_html.escape(best_bus_tip(arr))}</div>',
                                unsafe_allow_html=True)

            st.markdown("##### Route lanes")
            by_route = snap.set_index("route")
            order = ([eff] if eff else []) + [r for r in by_route.index if r != eff]
            lanes = "".join(lane_html(r, by_route.loc[r], route_stops, stop_xy, route_km, fc,
                                      live_day, int(sim_min // 60) % 24, r == eff) for r in order)
            st.markdown(f'<div class="lanes">{lanes}</div>', unsafe_allow_html=True)

            crowded = snap[snap.status == "Crowded"].sort_values("load_pct", ascending=False)
            if not crowded.empty:
                st.markdown("##### Action needed")
                st.markdown("".join(
                    f'<div class="alert"><b>{_html.escape(str(r.route))}</b> ({r.route_km:g} km) is '
                    f'{r.load_pct}% full, next stop {_html.escape(str(r.next_stop))} in {r.eta_min} min. '
                    f'{_html.escape(str(r.segment))}: {_html.escape(TIPS.get(r.segment, "").capitalize())}.</div>'
                    for _, r in crowded.head(5).iterrows()), unsafe_allow_html=True)

        if hasattr(st, "fragment"):
            st.fragment(run_every=2 if play else None)(live_panel)()
        else:
            live_panel()
            st.button("Refresh")

    # ----------------------------------------------------------- PREDICT
    with tab_pred:
        st.subheader("Prediction tool: how crowded will it be?")
        st.caption("Random Forest forecast. Change any control and the answer updates instantly.")
        p1, p2, p3, p4 = st.columns(4)
        pr = p1.selectbox("Route", routes_sorted, key="p_route")
        ps = p2.selectbox("Boarding stop", data_stops, key="p_stop")
        pday = p3.selectbox("Day", DAY_ORDER, index=DAY_ORDER.index(live_day), key="p_day")
        ph = p4.slider("Hour of day", hour_range[0], hour_range[1], min(max(8, hour_range[0]), hour_range[1]),
                       key="p_hour")
        mean, lo, hi = predict_with_range(models["final"], pr, ps, pday, int(ph))
        level = str(crowding_level([mean])[0])
        loadp = float(np.clip(mean / FULL_BUS_LOAD * 100, 0, 100))
        cards = "".join([
            kpi_html("Predicted boardings", f"{mean:.0f}", f"likely range {lo:.0f} to {hi:.0f}"),
            kpi_html("Predicted load", f"{loadp:.0f}%", f"{pr} at {ps}"),
            kpi_html("Crowding level", level, f"{pday} {int(ph):02d}:00"),
            kpi_html("Model typical error", f"{rf.MAE:.1f}", "passengers (MAE on test data)")])
        st.markdown(f'<div class="cp-kpis">{cards}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tip">Forecast for <b>{_html.escape(pr)}</b> '
                    f'({_html.escape(str(seg_map.get(pr, "")))}): <span class="pill {level}">{level}</span> '
                    f'at {int(ph):02d}:00 on {pday}.</div>', unsafe_allow_html=True)

        st.markdown("##### Compare days (click a legend entry to hide a line)")
        cmp_days = st.multiselect("Days to compare", DAY_ORDER, default=[pday, "Sat" if pday != "Sat" else "Mon"],
                                  key="p_cmp")
        rows_ = [{"hour": h, "day": d_, "load": v}
                 for d_ in (cmp_days or [pday]) for h, v in forecast_curve(pr, fc, d_, hours).items()]
        curve_df = pd.DataFrame(rows_).dropna()
        show_chart(st, lambda: chart_lines(curve_df, "hour", "load", "day", "Predicted load %", rule_x=int(ph)),
                   fallback=curve_df)
        wins = light_windows(forecast_curve(pr, fc, pday, hours))
        st.info(f"{pr} on {pday}: " + (f"lightest travel times are {', '.join(wins)}." if wins
                                       else "no hour is predicted to be light."))
        st.markdown(f"##### Predicted load for every route on {pday} (hover a cell)")
        grid = pd.DataFrame({r: forecast_curve(r, fc, pday, hours) for r in routes_sorted})
        heat_df = grid.T.stack().reset_index()
        heat_df.columns = ["route", "hour", "load"]
        show_chart(st, lambda: chart_heat(heat_df.dropna(), "hour", "route", "load", "Load %",
                                          order=routes_sorted, domain=[0, 100]), fallback=heat_df)

    # ---------------------------------------------------------- ACCURACY
    with tab_acc:
        st.subheader("Prediction accuracy (20% hold-out test set)")
        res = models["results"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Error (MAE)", f"{rf.MAE:.2f} passengers")
        m2.metric("RMSE", f"{rf.RMSE:.2f}")
        m3.metric("R\u00b2", f"{rf.R2:.3f}")
        m4.metric("Crowding-level accuracy", f"{rf['Crowding accuracy']:.1%}")
        metric = st.radio("Compare the models by", ["MAE", "RMSE", "R2", "Crowding accuracy"], horizontal=True)
        bars = res[["Model", metric]].rename(columns={metric: "value"})
        show_chart(st, lambda: chart_bar(bars, "Model", "value", metric), fallback=res)
        st.caption("Lower is better for MAE and RMSE. Higher is better for R2 and accuracy. "
                   "The baseline always guesses the average, so beating it shows the model learned something.")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("##### Actual vs predicted (zoom and hover)")
            xt = models["X_test"].copy()
            xt["actual"] = models["y_test"]
            xt["predicted"] = models["y_pred"]
            xt["hour"] = xt["hour"].astype(int)
            sample = xt.sample(min(1500, len(xt)), random_state=SEED)
            show_chart(st, lambda: chart_scatter_pred(sample), fallback=sample.head(20))
        with a2:
            st.markdown("##### Crowding-level confusion matrix")
            show_chart(st, lambda: chart_confusion(models["confusion"]),
                       fallback=pd.DataFrame(models["confusion"], index=LEVELS, columns=LEVELS))
        st.markdown("##### What drives demand")
        imp = models["importance"].reset_index()
        imp.columns = ["factor", "importance"]
        show_chart(st, lambda: chart_bar(imp, "factor", "importance", "importance", horizontal=True),
                   fallback=imp)

        st.markdown("##### Test it yourself: random held-out cases")
        seed_ = st.session_state.get("spot_seed", 0)
        if st.button("Draw 10 new random test cases"):
            seed_ += 1
            st.session_state["spot_seed"] = seed_
        idx = np.random.default_rng(seed_).choice(len(models["y_test"]), 10, replace=False)
        spot = models["X_test"].iloc[idx].copy().reset_index(drop=True)
        spot["actual"] = models["y_test"][idx]
        spot["predicted"] = models["y_pred"][idx].round(1)
        spot["error"] = (spot.predicted - spot.actual).round(1)
        spot["actual level"] = crowding_level(spot.actual)
        spot["predicted level"] = crowding_level(spot.predicted)
        spot["match"] = np.where(spot["actual level"] == spot["predicted level"], "yes", "no")
        st.dataframe(spot)
        s1, s2 = st.columns(2)
        s1.metric("This sample: average error", f"{spot.error.abs().mean():.1f} passengers")
        s2.metric("This sample: levels correct", f"{int((spot.match == 'yes').sum())} of 10")

        st.markdown("##### Clustering quality (K-Means has no accuracy)")
        q1, q2 = st.columns(2)
        q1.metric("Silhouette score", f"{final_sil:.3f}")
        rec = cluster_recovery(list(feat.index), km.labels_) if is_sim else None
        q2.metric("Match with hidden behaviours (ARI)", f"{rec[0]:.2f}" if rec else "n/a")
        st.caption("ARI is valid only for simulated data, where the true route behaviours are known.")
        st.warning("Honest note: the simulated data follows simple rules, so scores are high. "
                   "Real campus data will be noisier and score lower.")

    # ---------------------------------------------------------- CLUSTERS
    with tab_cl:
        st.subheader("Choosing K (hover the points)")
        show_chart(st, lambda: chart_k(ks, inertia, sil, K),
                   fallback=pd.DataFrame({"K": ks, "Inertia": inertia, "Silhouette": sil}))
        st.info(f"Auto rule: smallest K within 0.01 of the best silhouette ({max(sil):.3f}). "
                f"Auto K = {auto_k}. Using K = {K}. Change K in the sidebar and everything updates.")
        st.subheader("Route clusters: tap a route")
        P = PCA(2, random_state=SEED).fit_transform(X)
        var = PCA(2, random_state=SEED).fit(X).explained_variance_ratio_.sum()
        rows = show_chart(st, lambda: chart_pca(P, feat, names, var), key="pca_pick", select="pick",
                          fallback=feat[["segment", "avg_passengers"]])
        chosen_r = picked_value(rows) or list(feat.index)[0]
        row = feat.loc[chosen_r]
        st.markdown(f"##### Route {chosen_r}: {row.segment}")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Avg passengers", f"{row.avg_passengers:.1f}")
        r2.metric("Morning avg", f"{row.morning_avg:.1f}")
        r3.metric("Evening avg", f"{row.evening_avg:.1f}")
        r4.metric("Weekend ratio", f"{row.weekend_ratio:.2f}")
        st.success(f"Advice: {TIPS[row.segment].capitalize()}.")
        seg_pick = st.multiselect("Filter the route table by segment", seg_names, default=seg_names)
        st.dataframe(feat[feat.segment.isin(seg_pick)][["cluster", "segment", "avg_passengers",
                                                         "morning_avg", "evening_avg"]].round(1)
                     .sort_values("cluster"))
        st.subheader("Cluster feature profile (hover a cell)")
        z = pd.DataFrame(X, columns=feat.drop(columns=["cluster", "segment"]).columns,
                         index=feat.index).groupby(feat.cluster).mean()
        z.index = [names[c] for c in z.index]
        zl = z.stack().reset_index()
        zl.columns = ["segment", "feature", "z"]
        show_chart(st, lambda: chart_heat(zl, "feature", "segment", "z", "z-score", height=240),
                   fallback=z.round(2))
        st.subheader("Scheduling recommendations")
        for c, nm in sorted(names.items()):
            st.success(f"**{nm}** ({', '.join(feat[feat.cluster == c].index)}): {TIPS[nm].capitalize()}")

    # --------------------------------------------------- DEMAND PATTERNS
    with tab_dem:
        st.subheader("Explore demand: choose days, hours and segments")
        f1, f2, f3 = st.columns(3)
        dsel = f1.multiselect("Days", DAY_ORDER, default=data_days, key="d_days")
        hsel = f2.slider("Hours", hour_range[0], hour_range[1], hour_range, key="d_hours")
        ssel = f3.multiselect("Segments", seg_names, default=seg_names, key="d_segs")
        f = df[df.day.isin(dsel) & df.hour.between(hsel[0], hsel[1]) & df.segment.isin(ssel)]
        if f.empty:
            st.warning("No data for this selection.")
        else:
            order_r = [r for r in feat.sort_values("cluster").index if r in set(f.route)]
            hm = f.pivot_table(index="route", columns="hour", values="passenger_count",
                               aggfunc="mean").stack().reset_index()
            hm.columns = ["route", "hour", "avg"]
            st.markdown("##### Average passengers: route x hour (hover a cell)")
            show_chart(st, lambda: chart_heat(hm, "hour", "route", "avg", "Avg passengers", order=order_r),
                       fallback=hm)
            st.markdown("##### Hourly demand by segment (click a legend entry)")
            hs = f.groupby(["hour", "segment"]).passenger_count.mean().reset_index(name="avg")
            show_chart(st, lambda: chart_lines(hs, "hour", "avg", "segment", "Avg passengers"), fallback=hs)
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("##### Average boardings by stop")
                bs = f.groupby("boarding_location").passenger_count.mean().reset_index(name="avg")
                show_chart(st, lambda: chart_bar(bs, "boarding_location", "avg", "Avg passengers",
                                                 horizontal=True), fallback=bs)
            with b2:
                st.markdown("##### Average boardings by day")
                bd = f.groupby("day").passenger_count.mean().reindex(
                    [d for d in DAY_ORDER if d in set(f.day)]).reset_index(name="avg")
                show_chart(st, lambda: chart_bar(bd, "day", "avg", "Avg passengers"), fallback=bd)

    # ------------------------------------------------- DATA & DOWNLOADS
    with tab_data:
        st.subheader("Data explorer")
        g1, g2 = st.columns(2)
        rsel = g1.multiselect("Routes", routes_sorted, default=routes_sorted[:3], key="x_routes")
        dsel2 = g2.multiselect("Days", DAY_ORDER, default=data_days, key="x_days")
        view_df = raw[raw.route.isin(rsel) & raw.day.isin(dsel2)]
        st.caption(f"{len(view_df):,} of {len(raw):,} records shown (click a column header to sort).")
        st.dataframe(view_df.head(500))
        st.subheader("Downloads")
        st.download_button("Route clusters (CSV)", feat.round(2).to_csv().encode(), "route_clusters.csv", "text/csv")
        st.download_button("Raw data (CSV)", raw.to_csv(index=False).encode(), "raw_data.csv", "text/csv")
        st.download_button("Forecast accuracy (CSV)", models["results"].round(3).to_csv(index=False).encode(),
                           "prediction_results.csv", "text/csv")


# ======================================================================
# ENTRY POINT: dashboard if launched by Streamlit, otherwise the script
# ======================================================================
def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:                      # older Streamlit versions
            return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    run_script()