"""
Campus Transport Route Analyzer - Simple Dashboard (Team 15)
------------------------------------------------------------
Two models only:
  1. K-Means        -> groups the bus routes into demand segments   (unsupervised)
  2. Random Forest  -> predicts passengers and crowding level       (supervised)

Run (in the VS Code terminal, inside this folder):
    python -m pip install streamlit numpy pandas scikit-learn matplotlib seaborn
    python -m streamlit run dashboard.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

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
FULL_BUS_LOAD = 45            # average boardings treated as a "full" bus
LEVELS = ["Light", "Moderate", "Crowded"]
sns.set_theme(style="whitegrid")


# ======================================================================
# DATA, FEATURES AND K-MEANS HELPERS
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
# RANDOM FOREST (predict passengers from route, stop, day and hour)
# ======================================================================
def crowding_level(counts):
    """<40% of a full bus = Light, <75% = Moderate, otherwise Crowded."""
    return pd.cut(np.asarray(counts, dtype=float) / FULL_BUS_LOAD,
                  bins=[-np.inf, 0.40, 0.75, np.inf], labels=LEVELS).astype(str)


def train_random_forest(df):
    cols = ["route", "boarding_location", "day", "hour"]
    X = df[cols].copy()
    X["hour"] = X["hour"].astype(str)                      # hour as a category
    y = df["passenger_count"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    def build(estimator):
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cols)])
        return make_pipeline(pre, estimator)

    def forest():
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                     random_state=SEED, n_jobs=1)

    rf = build(forest()).fit(Xtr, ytr)
    baseline = build(DummyRegressor()).fit(Xtr, ytr)

    rows = []
    for name, model in [("Average-only baseline", baseline), ("Random Forest", rf)]:
        p = model.predict(Xte)
        rows.append({"Model": name,
                     "MAE": mean_absolute_error(yte, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
                     "R2": r2_score(yte, p),
                     "Crowding accuracy": accuracy_score(crowding_level(yte), crowding_level(p))})
    pred = rf.predict(Xte)

    names = rf[0].get_feature_names_out()
    imp = pd.Series(rf[-1].feature_importances_, index=names)
    importance = pd.Series({c: float(imp[[n for n in names if n.startswith("c__" + c + "_")]].sum())
                            for c in cols}).sort_values()
    return {"table": pd.DataFrame(rows),
            "y_test": np.asarray(yte), "y_pred": pred,
            "confusion": confusion_matrix(crowding_level(yte), crowding_level(pred), labels=LEVELS),
            "importance": importance,
            "final": build(forest()).fit(X, y)}          # final model trained on all data


# ======================================================================
# DASHBOARD
# ======================================================================
st.set_page_config(page_title="Campus Transport Route Analyzer", layout="wide")
make_data = st.cache_data(make_simulated_data)
train_rf = st.cache_data(train_random_forest)

# ---- sidebar: data source and K
st.sidebar.title("Settings")
source = st.sidebar.radio("Data source", ["Simulated data", "Upload CSV"])
if source == "Upload CSV":
    up = st.sidebar.file_uploader("CSV columns: " + ", ".join(REQUIRED), type="csv")
    if up is None:
        st.info("Upload a CSV in the sidebar, or switch to Simulated data.")
        st.stop()
    raw = pd.read_csv(up)
    if any(c not in raw.columns for c in REQUIRED):
        st.error("The CSV must contain: " + ", ".join(REQUIRED))
        st.stop()
else:
    raw = make_data()

# ---- prepare data, K-Means
df = preprocess(raw)
df["slot"] = df["slot"].astype(str)
feat = build_features(df)
if len(feat) < 4:
    st.error("At least 4 routes are needed for clustering.")
    st.stop()
X = StandardScaler().fit_transform(feat)
ks, inertia, sil, auto_k = evaluate_k(X, len(feat))
auto = st.sidebar.checkbox("Choose K automatically", value=True)
K = auto_k if auto else st.sidebar.slider("Number of clusters (K)", ks[0], ks[-1], auto_k)
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
feat["cluster"] = km.labels_
names, _ = name_clusters(feat)
feat["segment"] = feat.cluster.map(names)
final_sil = silhouette_score(X, km.labels_)
df = df.merge(feat[["segment"]], left_on="route", right_index=True)

# ---- Random Forest
rfm = train_rf(df)
rf_row = rfm["table"].set_index("Model").loc["Random Forest"]

# ---- header
st.title("Campus Transport Route Analyzer")
st.caption("Team 15 | K-Means groups the routes, Random Forest predicts crowding"
           + (" | simulated data" if source == "Simulated data" else ""))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Records", f"{len(df):,}")
m2.metric("Routes", feat.shape[0])
m3.metric("K-Means silhouette", f"{final_sil:.3f}")
m4.metric("Random Forest error (MAE)", f"{rf_row.MAE:.1f} passengers")
m5.metric("Crowding accuracy", f"{rf_row['Crowding accuracy']:.1%}")

tab_data, tab_km, tab_rf = st.tabs(["1. Data", "2. K-Means: Route Clusters",
                                    "3. Random Forest: Prediction and Accuracy"])

# ----------------------------------------------------------------- DATA
with tab_data:
    st.subheader("Data sample")
    st.dataframe(raw.head(10))
    a, b = st.columns(2)
    with a:
        st.subheader("Average passengers by hour")
        st.line_chart(df.groupby("hour").passenger_count.mean())
    with b:
        st.subheader("Average passengers by day")
        by_day = df.groupby("day").passenger_count.mean()
        st.bar_chart(by_day.reindex([d for d in DAY_ORDER if d in by_day.index]))

# --------------------------------------------------------------- K-MEANS
with tab_km:
    st.write("K-Means is **unsupervised**: it has no correct answers to learn from, "
             "so it is judged by the silhouette score, not accuracy.")
    a, b = st.columns(2)
    with a:
        st.subheader("Choosing K: silhouette score")
        st.line_chart(pd.DataFrame({"Silhouette": sil}, index=ks))
        st.caption(f"Best score {max(sil):.3f}. Smallest K within 0.01 of it = {auto_k}. "
                   f"Using K = {K}.")
    with b:
        st.subheader("Choosing K: inertia (elbow)")
        st.line_chart(pd.DataFrame({"Inertia": inertia}, index=ks))

    a, b = st.columns(2)
    with a:
        st.subheader("Route clusters (PCA view)")
        pca = PCA(2, random_state=SEED)
        P = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c in sorted(feat.cluster.unique()):
            idx = feat.cluster.values == c
            ax.scatter(P[idx, 0], P[idx, 1], s=150, edgecolor="k", label=names[c])
        for i, r in enumerate(feat.index):
            ax.annotate(r, (P[i, 0] + 0.07, P[i, 1] + 0.07))
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with b:
        st.subheader("Route and segment")
        st.dataframe(feat[["segment", "avg_passengers", "morning_avg", "evening_avg"]]
                     .round(1).sort_values("segment"))

    st.subheader("Average passengers: route x hour")
    order = feat.sort_values("cluster").index
    heat = df.pivot_table(index="route", columns="hour", values="passenger_count",
                          aggfunc="mean").loc[order]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(heat, cmap="YlOrRd", cbar_kws={"label": "Avg passengers"}, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Scheduling recommendations")
    for c, nm in sorted(names.items()):
        st.success(f"**{nm}** ({', '.join(feat[feat.cluster == c].index)}): "
                   f"{TIPS[nm].capitalize()}")

# --------------------------------------------------------- RANDOM FOREST
with tab_rf:
    st.write("Random Forest is **supervised**: it learns from past records to predict "
             "passengers, so it has real accuracy numbers.")
    st.subheader("Try the model")
    hmin, hmax = int(df.hour.min()), int(df.hour.max())
    stops = sorted(df.boarding_location.unique())
    c1, c2, c3, c4 = st.columns(4)
    route = c1.selectbox("Route", sorted(feat.index, key=lambda r: (len(r), r)))
    stop = c2.selectbox("Boarding stop", stops)
    day = c3.selectbox("Day", [d for d in DAY_ORDER if d in set(df.day)])
    hour = c4.slider("Hour", hmin, hmax, min(max(8, hmin), hmax))

    one = pd.DataFrame([{"route": route, "boarding_location": stop, "day": day, "hour": str(hour)}])
    pred_n = float(rfm["final"].predict(one)[0])
    level = str(crowding_level([pred_n])[0])
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted passengers", f"{pred_n:.0f}")
    r2.metric("Load of a full bus", f"{min(pred_n / FULL_BUS_LOAD * 100, 100):.0f}%")
    r3.metric("Crowding level", level)
    {"Light": st.success, "Moderate": st.warning, "Crowded": st.error}[level](
        f"{route} at {stop} on {day}, {hour:02d}:00 is predicted to be **{level}**.")

    hours = list(range(hmin, hmax + 1))
    day_grid = pd.DataFrame({"route": route, "boarding_location": stop, "day": day,
                             "hour": [str(h) for h in hours]})
    st.subheader(f"Predicted passengers through the day: {route}, {stop}, {day}")
    st.line_chart(pd.DataFrame({"Predicted passengers": rfm["final"].predict(day_grid)}, index=hours))

    st.subheader("Accuracy (20% of the data kept aside for testing)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Average error (MAE)", f"{rf_row.MAE:.2f}")
    a2.metric("RMSE", f"{rf_row.RMSE:.2f}")
    a3.metric("R2 score", f"{rf_row.R2:.2f}")
    a4.metric("Crowding-level accuracy", f"{rf_row['Crowding accuracy']:.1%}")
    st.dataframe(rfm["table"].round(3))
    st.caption("Baseline = always predict the average. Beating it shows the forest learned the patterns. "
               "Crowding accuracy = how often Light, Moderate or Crowded was predicted correctly.")

    a, b = st.columns(2)
    with a:
        st.subheader("Actual vs predicted")
        n = min(1500, len(rfm["y_test"]))
        idx = np.random.default_rng(SEED).choice(len(rfm["y_test"]), n, replace=False)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(rfm["y_test"][idx], rfm["y_pred"][idx], s=8, alpha=0.3)
        lim = [0, max(rfm["y_test"].max(), rfm["y_pred"].max())]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Actual passengers"); ax.set_ylabel("Predicted passengers")
        st.pyplot(fig)
        plt.close(fig)
    with b:
        st.subheader("Crowding-level confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(rfm["confusion"], annot=True, fmt="d", cmap="Blues", xticklabels=LEVELS,
                    yticklabels=LEVELS, cbar=False, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("What drives demand?")
    st.bar_chart(rfm["importance"])
    if source == "Simulated data":
        st.info("The simulated data follows simple rules, so scores are high. "
                "Real campus data will be noisier and score lower.")