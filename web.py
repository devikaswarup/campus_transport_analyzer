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
# LIVE TRACKER HELPERS (simulated GPS, Ola-style)
# ======================================================================
CAMPUS_CENTER = (10.0000, 76.3000)   # (lat, lon) -> EDIT to your campus location
LOOP_MIN = 24                        # minutes one bus takes for a full loop
FULL_BUS_LOAD = 45                   # avg boardings treated as a "full" bus
STATUS_COLORS = {"Light": [40, 167, 69], "Moderate": [255, 193, 7], "Crowded": [220, 53, 69]}
STATUS_HEX = {"Light": "#28a745", "Moderate": "#ffc107", "Crowded": "#dc3545"}


def make_stop_coords(stop_names, center=CAMPUS_CENTER, radius=0.0022):
    """Place stops on a ring around the campus centre (sample layout)."""
    stops = sorted(stop_names)
    xy = {}
    for i, name in enumerate(stops):
        ang = 2 * np.pi * i / len(stops)
        xy[name] = (center[0] + radius * np.sin(ang), center[1] + radius * np.cos(ang))
    return xy


def make_route_stops(routes, stop_names, n_stops=4):
    """Each route serves n_stops consecutive stops (rotating start)."""
    stops = sorted(stop_names)
    n = min(n_stops, len(stops))
    return {r: [stops[(i + j) % len(stops)] for j in range(n)] for i, r in enumerate(routes)}


def build_demand_lookup(df):
    """Average boardings per (route, hour, weekend flag) from the data."""
    by = df.groupby(["route", "hour", "is_weekend"]).passenger_count.mean().to_dict()
    overall = df.groupby("route").passenger_count.mean().to_dict()
    return by, overall


def live_snapshot(routes, segments, stop_xy, route_stops, demand, sim_min, is_weekend,
                  hour_range):
    """Position, ETA and estimated crowding of one bus per route at sim time."""
    by, overall = demand
    hour = int(sim_min // 60) % 24
    if not (hour_range[0] <= hour <= hour_range[1]):
        return pd.DataFrame()                      # off duty
    rows = []
    for i, r in enumerate(routes):
        pts = route_stops[r]
        n = len(pts)
        prog = ((sim_min / LOOP_MIN) + i / len(routes)) % 1.0
        seg = prog * n
        k, frac = int(seg) % n, seg - int(seg)
        a, b = stop_xy[pts[k]], stop_xy[pts[(k + 1) % n]]
        lat = a[0] + (b[0] - a[0]) * frac
        lon = a[1] + (b[1] - a[1]) * frac
        eta = (1 - frac) * LOOP_MIN / n
        base = by.get((r, hour, is_weekend), overall.get(r, 0))
        jitter = np.random.default_rng(int(sim_min) * 100 + i).normal(0, 0.05)
        load = float(np.clip(base / FULL_BUS_LOAD + jitter, 0.03, 1.0))
        status = "Crowded" if load >= 0.75 else "Moderate" if load >= 0.40 else "Light"
        rows.append({"route": r, "segment": segments.get(r, ""), "lat": lat, "lon": lon,
                     "seg_pos": seg,
                     "next_stop": pts[(k + 1) % n], "eta_min": round(eta, 1),
                     "load_pct": round(load * 100), "status": status,
                     "color": STATUS_COLORS[status], "color_hex": STATUS_HEX[status]})
    return pd.DataFrame(rows)


def arrivals_at_stop(snap, route_stops, stop):
    """Running buses that serve `stop`, with minutes until each arrives."""
    rows = []
    for _, b in snap.iterrows():
        pts = route_stops[b.route]
        if stop in pts:
            n = len(pts)
            delta = (pts.index(stop) - b.seg_pos) % n
            rows.append({"route": b.route, "eta_min": round(delta * LOOP_MIN / n, 1),
                         "status": b.status, "load_pct": b.load_pct, "segment": b.segment})
    out = pd.DataFrame(rows, columns=["route", "eta_min", "status", "load_pct", "segment"])
    return out.sort_values("eta_min").reset_index(drop=True)


def best_bus_tip(arr, window=15):
    """Plain-English boarding advice from the arrivals table."""
    cand = arr[arr.eta_min <= window]
    if cand.empty:
        return "No bus arrives at this stop within 15 minutes."
    first = cand.iloc[0]
    lightest = cand.sort_values(["load_pct", "eta_min"]).iloc[0]
    if (lightest.route != first.route and first.load_pct >= 75
            and lightest.load_pct <= first.load_pct - 20):
        return (f"{first.route} arrives first in {first.eta_min} min but is {first.load_pct}% full. "
                f"Waiting for {lightest.route} ({lightest.eta_min} min, {lightest.load_pct}% full) "
                f"gives a more comfortable ride.")
    return (f"Take {first.route}: arrives in {first.eta_min} min, "
            f"{first.status.lower()} ({first.load_pct}% full).")


def expected_load_curve(route, demand, hours):
    """Expected load % by hour for a route on weekdays and weekends."""
    by, overall = demand
    def pct(h, wk):
        v = by.get((route, h, wk), np.nan)
        return float(np.clip(v / FULL_BUS_LOAD * 100, 0, 100)) if v == v else np.nan
    return pd.DataFrame({"Weekday": [pct(h, 0) for h in hours],
                         "Weekend": [pct(h, 1) for h in hours]}, index=list(hours))


# ======================================================================
# MODE 1: ANALYSIS SCRIPT  (python campus_transport_analyzer.py)
# ======================================================================
def run_script():
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
    print(f"\nAll files saved in: {OUT}")
    print("Tip: for the interactive dashboard run:  "
          "python -m streamlit run campus_transport_analyzer.py")


# ======================================================================
# MODE 2: DASHBOARD  (python -m streamlit run campus_transport_analyzer.py)
# ======================================================================
def run_dashboard():
    import streamlit as st

    st.set_page_config(page_title="Campus Transport Route Analyzer", layout="wide")
    cached_sim = st.cache_data(make_simulated_data)

    # ---------------------------------------------------------- SIDEBAR
    st.sidebar.title("Controls")
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

    all_routes = sorted(feat.index, key=lambda r: (len(r), r))
    chosen = st.sidebar.multiselect("Routes to show in charts", all_routes, default=all_routes)
    if not chosen:
        chosen = all_routes
    view = df[df.route.isin(chosen)]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Live tracker")
    play = st.sidebar.checkbox("Live updates (every 2 s)", value=True)
    speed = st.sidebar.slider("Simulation speed (minutes per update)", 1, 10, 2)
    live_day = st.sidebar.selectbox("Day", DAY_ORDER, index=0)
    start_hour = st.sidebar.slider("Start hour", 6, 20, 7)
    if st.session_state.get("start_key") != (start_hour, live_day):
        st.session_state["start_key"] = (start_hour, live_day)
        st.session_state["sim_min"] = start_hour * 60

    # ----------------------------------------------------------- HEADER
    st.title("Campus Transport Route Analyzer")
    st.caption("K-Means clustering of campus bus routes by demand pattern | Team 15"
               + (" | simulated data" if source == "Simulated data" else ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Routes", feat.shape[0])
    c3.metric("Clusters (K)", K)
    c4.metric("Silhouette score", f"{final_sil:.3f}")

    tab_live, tab_over, tab_k, tab_cl, tab_dem, tab_rec, tab_dl = st.tabs(
        ["Live Tracker", "Overview", "Choose K", "Clusters", "Demand patterns",
         "Recommendations", "Downloads"])

    # ------------------------------------------------------ LIVE TRACKER
    with tab_live:
        st.subheader("Live bus tracker (simulated)")
        st.caption("Bus positions are simulated on a sample campus layout. Crowding is "
                   "estimated from your data's average demand for the current hour and day.")
        routes_sorted = sorted(feat.index, key=lambda r: (len(r), r))
        stop_names = sorted(df.boarding_location.unique())
        stop_xy = make_stop_coords(stop_names)
        route_stops = make_route_stops(routes_sorted, stop_names)
        demand = build_demand_lookup(df)
        hour_range = (int(df.hour.min()), int(df.hour.max()))
        seg_map = feat.segment.to_dict()
        is_wk = int(live_day in ("Sat", "Sun"))

        # ---- choose what to follow (click-style selection)
        sc1, sc2 = st.columns(2)
        sel_route = sc1.selectbox("Track a route", ["All routes"] + routes_sorted)
        sel_stop = sc2.selectbox("Choose your stop", ["None"] + stop_names)
        sel_route = None if sel_route == "All routes" else sel_route
        sel_stop = None if sel_stop == "None" else sel_stop

        def draw_map(snap):
            hi = [0, 110, 255]
            try:
                import pydeck as pdk
                paths = pd.DataFrame({
                    "name": routes_sorted,
                    "path": [[[stop_xy[x][1], stop_xy[x][0]] for x in route_stops[r]]
                             + [[stop_xy[route_stops[r][0]][1], stop_xy[route_stops[r][0]][0]]]
                             for r in routes_sorted],
                    "color": [hi + [255] if r == sel_route
                              else [120, 120, 120, 30 if sel_route else 90]
                              for r in routes_sorted],
                    "width": [7 if r == sel_route else 2 for r in routes_sorted]})
                stops_df = pd.DataFrame({"name": stop_names,
                                         "lat": [stop_xy[x][0] for x in stop_names],
                                         "lon": [stop_xy[x][1] for x in stop_names],
                                         "radius": [34 if x == sel_stop else 14 for x in stop_names],
                                         "color": [hi + [255] if x == sel_stop else [30, 30, 30, 255]
                                                   for x in stop_names]})
                bus = snap.copy()
                bus["rgba"] = [c + [255 if (sel_route is None or r == sel_route) else 70]
                               for c, r in zip(bus.color, bus.route)]
                bus["radius"] = [40 if r == sel_route else 26 for r in bus.route]
                layers = [
                    pdk.Layer("PathLayer", paths, get_path="path", get_color="color",
                              get_width="width", width_units="pixels"),
                    pdk.Layer("ScatterplotLayer", stops_df, get_position="[lon, lat]",
                              get_fill_color="color", get_radius="radius"),
                    pdk.Layer("TextLayer", stops_df, get_position="[lon, lat]", get_text="name",
                              get_size=13, get_color=[40, 40, 40], get_pixel_offset=[0, 18]),
                    pdk.Layer("ScatterplotLayer", bus, get_position="[lon, lat]",
                              get_fill_color="rgba", get_radius="radius", pickable=True),
                    pdk.Layer("TextLayer", bus, get_position="[lon, lat]", get_text="route",
                              get_size=15, get_color=[0, 0, 0], get_pixel_offset=[0, -20]),
                ]
                lat0, lon0 = stop_xy[sel_stop] if sel_stop else CAMPUS_CENTER
                deck = pdk.Deck(
                    layers=layers, map_provider="carto", map_style="light",
                    initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0,
                                                     zoom=16.2 if sel_stop else 15.3),
                    tooltip={"text": "{route} | {status} {load_pct}%\nNext: {next_stop} "
                                     "in {eta_min} min"})
                st.pydeck_chart(deck)
            except Exception:
                st.map(snap, latitude="lat", longitude="lon", color="color_hex", size=20)

        def live_panel():
            if play:
                st.session_state["sim_min"] = st.session_state["sim_min"] + speed
            sim_min = st.session_state["sim_min"]
            hh, mm = int(sim_min // 60) % 24, int(sim_min % 60)
            snap = live_snapshot(routes_sorted, seg_map, stop_xy, route_stops, demand,
                                 sim_min, is_wk, hour_range)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Simulated time", f"{live_day} {hh:02d}:{mm:02d}")
            if snap.empty:
                m2.metric("Buses running", 0)
                st.warning("Outside service hours for this data. Raise the start hour or "
                           "let the clock run on.")
                return
            m2.metric("Buses running", len(snap))
            m3.metric("Crowded buses", int((snap.status == "Crowded").sum()))
            m4.metric("Average load", f"{snap.load_pct.mean():.0f}%")
            left, right = st.columns([3, 2])
            with left:
                draw_map(snap)
                st.caption("Green = light, amber = moderate, red = crowded. "
                           "Blue = your selected route or stop.")
            with right:
                st.dataframe(snap[["route", "status", "load_pct", "next_stop", "eta_min"]]
                             .sort_values("load_pct", ascending=False)
                             .rename(columns={"load_pct": "load %", "eta_min": "ETA (min)"}),
                             hide_index=True)

            if sel_stop:                                   # ---- stop selected
                st.markdown(f"#### Buses arriving at {sel_stop}")
                arr = arrivals_at_stop(snap, route_stops, sel_stop)
                if arr.empty:
                    st.info("No running bus serves this stop right now.")
                else:
                    st.dataframe(arr.rename(columns={"eta_min": "arrives in (min)",
                                                     "load_pct": "load %"}), hide_index=True)
                    st.success(best_bus_tip(arr))

            if sel_route:                                  # ---- route selected
                row = snap[snap.route == sel_route].iloc[0]
                st.markdown(f"#### Route {sel_route} ({row.segment})")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Status", row.status)
                r2.metric("Load", f"{row.load_pct}%")
                r3.metric("Next stop", row.next_stop)
                r4.metric("ETA", f"{row.eta_min} min")
                st.write("Stops: " + " \u2192 ".join(route_stops[sel_route]) + " \u21BB")
                st.info(f"Planning advice: {TIPS.get(row.segment, '').capitalize()}.")

            crowded = snap[snap.status == "Crowded"].sort_values("load_pct", ascending=False)
            if crowded.empty:
                st.success("No crowded buses right now.")
            for _, row in crowded.head(5).iterrows():
                st.error(f"**{row.route}** is {row.load_pct}% full, next stop "
                         f"{row.next_stop} in {row.eta_min} min. "
                         f"{row.segment}: {TIPS.get(row.segment, '').capitalize()}.")

        if hasattr(st, "fragment"):
            st.fragment(run_every=2 if play else None)(live_panel)()
        else:                                   # older Streamlit: manual refresh
            live_panel()
            st.button("Refresh")

        if sel_route:                                      # typical demand curve
            st.markdown(f"#### Typical load by hour: route {sel_route}")
            st.line_chart(expected_load_curve(sel_route, demand,
                                              range(hour_range[0], hour_range[1] + 1)))

    # --------------------------------------------------------- OVERVIEW
    with tab_over:
        st.subheader("Data sample")
        st.dataframe(raw.head(15))
        a, b = st.columns(2)
        with a:
            st.subheader("Average boardings by hour")
            hourly = view.groupby("hour").passenger_count.mean()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(hourly.index, hourly.values, marker="o")
            ax.set_xlabel("Hour"); ax.set_ylabel("Avg passengers")
            st.pyplot(fig); plt.close(fig)
        with b:
            st.subheader("Average boardings by day")
            bd = view.groupby("day").passenger_count.mean()
            bd = bd.reindex([d for d in DAY_ORDER if d in bd.index])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(bd.index, bd.values, color="steelblue")
            ax.set_ylabel("Avg passengers")
            st.pyplot(fig); plt.close(fig)
        st.subheader("Average boardings by stop")
        bs = view.groupby("boarding_location").passenger_count.mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(bs.index, bs.values, color="teal")
        ax.set_xlabel("Avg passengers")
        st.pyplot(fig); plt.close(fig)

    # --------------------------------------------------------- CHOOSE K
    with tab_k:
        st.subheader("Elbow method and silhouette score")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow (inertia)")
        ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
        ax[1].plot(ks, sil, "o-", color="green"); ax[1].axvline(K, ls="--", color="red")
        ax[1].set_title("Silhouette score"); ax[1].set_xlabel("K")
        st.pyplot(fig); plt.close(fig)
        st.dataframe(pd.DataFrame({"K": ks, "Inertia": np.round(inertia, 2),
                                   "Silhouette": np.round(sil, 3)}))
        st.info(f"Auto rule: the smallest K whose silhouette is within 0.01 of the best "
                f"(best = {max(sil):.3f}). Auto K = {auto_k}. Currently using K = {K}.")

    # --------------------------------------------------------- CLUSTERS
    with tab_cl:
        left, right = st.columns(2)
        with left:
            st.subheader("PCA view of route clusters")
            pca = PCA(2, random_state=SEED)
            P = pca.fit_transform(X)
            fig, ax = plt.subplots(figsize=(6, 5))
            for c in sorted(feat.cluster.unique()):
                idx = feat.cluster.values == c
                ax.scatter(P[idx, 0], P[idx, 1], s=150, edgecolor="k", label=names[c])
            for i, r in enumerate(feat.index):
                ax.annotate(r, (P[i, 0] + 0.07, P[i, 1] + 0.07))
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8)
            ax.set_title(f"{pca.explained_variance_ratio_.sum():.0%} of variance shown")
            st.pyplot(fig); plt.close(fig)
        with right:
            st.subheader("Route to segment")
            st.dataframe(feat[["cluster", "segment", "avg_passengers", "morning_avg",
                               "evening_avg"]].round(1).sort_values("cluster"))
        st.subheader("Cluster feature profile (standardized)")
        z = pd.DataFrame(X, columns=feat.drop(columns=["cluster", "segment"]).columns,
                         index=feat.index).groupby(feat.cluster).mean()
        z.index = [names[c] for c in z.index]
        fig, ax = plt.subplots(figsize=(12, 3.8))
        sns.heatmap(z, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig); plt.close(fig)

    # --------------------------------------------------- DEMAND PATTERNS
    with tab_dem:
        st.subheader("Average passengers: route x hour")
        order = [r for r in feat.sort_values("cluster").index if r in chosen]
        heat = view.pivot_table(index="route", columns="hour", values="passenger_count",
                                aggfunc="mean").reindex(order)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.heatmap(heat, cmap="YlOrRd", cbar_kws={"label": "Avg passengers"}, ax=ax)
        st.pyplot(fig); plt.close(fig)
        a, b = st.columns(2)
        with a:
            st.subheader("Routes per segment")
            fig, ax = plt.subplots(figsize=(6, 3.8))
            feat.segment.value_counts().plot.bar(ax=ax, color="steelblue")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig); plt.close(fig)
        with b:
            st.subheader("Hourly demand by segment")
            hs = df.groupby(["hour", "segment"]).passenger_count.mean().unstack()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            hs.plot(ax=ax, marker="o")
            ax.set_ylabel("Avg passengers"); ax.legend(fontsize=7)
            st.pyplot(fig); plt.close(fig)

    # ----------------------------------------------------- RECOMMENDATIONS
    with tab_rec:
        st.subheader("Scheduling recommendations")
        for c, nm in sorted(names.items()):
            rs = ", ".join(feat[feat.cluster == c].index)
            st.success(f"**{nm}** ({rs}): {TIPS[nm].capitalize()}")
        st.caption("Segments and advice are rule-based from each cluster's average profile.")

    # ------------------------------------------------------------ DOWNLOADS
    with tab_dl:
        st.subheader("Download results")
        st.download_button("Route clusters (CSV)", feat.round(2).to_csv().encode(),
                           "route_clusters.csv", "text/csv")
        st.download_button("Raw data (CSV)", raw.to_csv(index=False).encode(),
                           "raw_data.csv", "text/csv")


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