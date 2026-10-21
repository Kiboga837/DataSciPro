import time
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# XGBoost is optional: the app works without it.
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Smart Data Science Studio", layout="wide")
st.title("Smart Data Science Studio")
st.caption("Upload a dataset, explore it, analyse sales trends, and compare machine learning models.")


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read a CSV or Excel file from raw bytes."""
    buffer = BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1")
    return pd.read_excel(buffer)


def make_sample_data(n: int = 800) -> pd.DataFrame:
    """Synthetic sales dataset so the app can be tried without a file."""
    rng = np.random.default_rng(42)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    month = dates.month.to_numpy()
    seasonal = 1 + 0.35 * np.isin(month, [11, 12]) + 0.15 * np.isin(month, [3, 9])
    category = rng.choice(["Furniture", "Technology", "Office Supplies"], n)
    region = rng.choice(["East", "West", "Central", "South"], n)
    quantity = rng.integers(1, 10, n)
    discount = rng.choice([0, 0.1, 0.2, 0.3, 0.5], n)
    sales = rng.gamma(2.0, 120, n) * quantity / 3 * seasonal
    profit = sales * (0.25 - discount) + rng.normal(0, 15, n)
    return pd.DataFrame(
        {
            "Order Date": dates.sort_values(),
            "Category": category,
            "Region": region,
            "Quantity": quantity,
            "Discount": discount,
            "Sales": sales.round(2),
            "Profit": profit.round(2),
        }
    )


def detect_date_columns(df: pd.DataFrame) -> list:
    """Columns that already are, or can be parsed as, dates."""
    found = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            found.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().head(200)
            if sample.empty:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                found.append(col)
    return found


def prepare_features(df: pd.DataFrame, feature_cols: list, max_categories: int = 20):
    """Keep numeric columns, one-hot encode low-cardinality text columns, drop
    high-cardinality text columns (IDs, names). Returns (X, dropped_columns)."""
    X = df[feature_cols].copy()
    dropped = [
        c for c in X.select_dtypes(exclude=[np.number]).columns if X[c].nunique() > max_categories
    ]
    X = X.drop(columns=dropped)
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, dropped


# ----------------------------------------------------------------------------
# Modelling helpers
# ----------------------------------------------------------------------------
def get_models(problem: str, n_estimators: int, max_depth: int) -> dict:
    depth = None if max_depth == 0 else max_depth
    if problem == "Regression":
        models = {
            "Linear Regression": Pipeline(
                [("scale", StandardScaler()), ("model", LinearRegression())]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=n_estimators, max_depth=depth, random_state=42, n_jobs=-1
                        ),
                    )
                ]
            ),
            "Gradient Boosting": Pipeline(
                [
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=n_estimators, max_depth=depth or 3, random_state=42
                        ),
                    )
                ]
            ),
        }
        if HAS_XGB:
            models["XGBoost"] = Pipeline(
                [
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=n_estimators, max_depth=depth or 6, random_state=42, n_jobs=-1
                        ),
                    )
                ]
            )
    else:
        models = {
            "Logistic Regression": Pipeline(
                [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=n_estimators, max_depth=depth, random_state=42, n_jobs=-1
                        ),
                    )
                ]
            ),
            "Gradient Boosting": Pipeline(
                [
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=n_estimators, max_depth=depth or 3, random_state=42
                        ),
                    )
                ]
            ),
        }
        if HAS_XGB:
            models["XGBoost"] = Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=depth or 6,
                            random_state=42,
                            n_jobs=-1,
                            eval_metric="logloss",
                        ),
                    )
                ]
            )
    return models


def evaluate(problem: str, y_true, y_pred) -> dict:
    if problem == "Regression":
        return {
            "R²": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def get_importance(pipe: Pipeline, columns):
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(np.atleast_2d(est.coef_)).mean(axis=0)
    else:
        return None
    return pd.Series(vals, index=columns).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Sidebar: upload + task selection
# ----------------------------------------------------------------------------
st.sidebar.header("1. Upload Data")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
use_sample = st.sidebar.checkbox("Use sample sales data", value=False)

df = None
if file:
    try:
        df = load_data(file.getvalue(), file.name)
        st.sidebar.success("Data Loaded Successfully")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
elif use_sample:
    df = make_sample_data()
    st.sidebar.info("Using synthetic sample data")

if df is None:
    st.info("Upload a CSV or Excel file in the sidebar (or tick the sample data box) to get started.")
    st.stop()

st.sidebar.header("2. Select Task")
task = st.sidebar.radio(
    "Choose an option", ["Exploratory Analysis", "Sales Trends", "Predictive Model"]
)

# ----------------------------------------------------------------------------
# Dataset preview (always shown)
# ----------------------------------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# ----------------------------------------------------------------------------
# Task 1: Exploratory Data Analysis
# ----------------------------------------------------------------------------
if task == "Exploratory Analysis":
    st.subheader("Exploratory Data Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate rows", int(df.duplicated().sum()))

    left, right = st.columns(2)
    with left:
        st.write("**Missing Values**")
        missing = df.isnull().sum().rename("Missing").to_frame()
        missing["% Missing"] = (missing["Missing"] / len(df) * 100).round(2)
        st.dataframe(missing)
    with right:
        st.write("**Column Types**")
        st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame())

    if num_cols:
        st.write("**Summary Statistics**")
        st.dataframe(df[num_cols].describe().T)

        st.subheader("Numeric Columns Distribution")
        selected_col = st.selectbox("Select Column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="skyblue")
        ax.set_xlabel(selected_col)
        st.pyplot(fig)

        st.subheader("Outlier Check")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_col].dropna(), ax=ax, color="lightcoral")
        st.pyplot(fig)

        if len(num_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            st.subheader("Scatter Plot")
            sc1, sc2 = st.columns(2)
            x_axis = sc1.selectbox("X axis", num_cols, index=0, key="sx")
            y_axis = sc2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1), key="sy")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, alpha=0.6, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("No numeric columns found in this dataset.")

    if cat_cols:
        st.subheader("Categorical Columns")
        cat_choice = st.selectbox("Select Categorical Column", cat_cols)
        counts = df[cat_choice].value_counts().head(15)
        fig, ax = plt.subplots()
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="steelblue")
        ax.set_xlabel("Count")
        ax.set_ylabel(cat_choice)
        st.pyplot(fig)

        if num_cols:
            st.write("**Average of a numeric column by category**")
            metric_col = st.selectbox("Numeric column", num_cols, key="grp_metric")
            grouped = df.groupby(cat_choice)[metric_col].mean().sort_values(ascending=False).head(15)
            st.bar_chart(grouped)

# ----------------------------------------------------------------------------
# Task 2: Sales Trends (time series)
# ----------------------------------------------------------------------------
elif task == "Sales Trends":
    st.subheader("Date-Based Sales Trends")

    date_cols = detect_date_columns(df)
    if not date_cols or not num_cols:
        st.warning(
            "This view needs at least one date column (e.g. Order Date) and one numeric column."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", date_cols)
    default_idx = num_cols.index("Sales") if "Sales" in num_cols else 0
    value_col = c2.selectbox("Value column", num_cols, index=default_idx)
    agg_label = c3.selectbox("Aggregation", ["Sum", "Mean", "Count"])
    freq_label = c4.selectbox("Period", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=2)

    agg_fn = {"Sum": "sum", "Mean": "mean", "Count": "count"}[agg_label]
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}[freq_label]

    ts = df[[date_col, value_col]].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    if ts.empty:
        st.error("No valid dates found in that column.")
        st.stop()

    ts["period"] = ts[date_col].dt.to_period(freq).dt.to_timestamp()
    series = ts.groupby("period")[value_col].agg(agg_fn).sort_index()

    st.caption(f"Data range: {ts[date_col].min().date()} to {ts[date_col].max().date()}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Periods", len(series))
    k2.metric("Best period", series.idxmax().strftime("%Y-%m-%d"), f"{series.max():,.0f}")
    k3.metric("Weakest period", series.idxmin().strftime("%Y-%m-%d"), f"{series.min():,.0f}")
    if len(series) > 1 and series.iloc[-2] != 0:
        growth = (series.iloc[-1] / series.iloc[-2] - 1) * 100
        k4.metric("Latest period change", f"{series.iloc[-1]:,.0f}", f"{growth:+.1f}%")
    else:
        k4.metric("Latest period", f"{series.iloc[-1]:,.0f}")

    window = st.slider("Moving average window (periods)", 1, 12, 3)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, marker="o", markersize=3, label=f"{agg_label} of {value_col}")
    if window > 1 and len(series) >= window:
        ax.plot(
            series.index,
            series.rolling(window).mean().values,
            color="crimson",
            linewidth=2,
            label=f"{window}-period moving average",
        )
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Seasonality")
    s1, s2 = st.columns(2)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    by_month = ts.groupby(ts[date_col].dt.month)[value_col].agg(agg_fn).reindex(range(1, 13))
    by_month.index = month_names
    with s1:
        st.write("**By month of year**")
        st.bar_chart(by_month.fillna(0))
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = ts.groupby(ts[date_col].dt.dayofweek)[value_col].agg(agg_fn).reindex(range(7))
    by_dow.index = dow_names
    with s2:
        st.write("**By day of week**")
        st.bar_chart(by_dow.fillna(0))

    group_options = [c for c in cat_cols if c != date_col and df[c].nunique() <= 15]
    if group_options:
        st.subheader("Trend by Category")
        group_col = st.selectbox("Break down by", ["None"] + group_options)
        if group_col != "None":
            tmp = df[[date_col, value_col, group_col]].copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            tmp["period"] = tmp[date_col].dt.to_period(freq).dt.to_timestamp()
            pivot = tmp.pivot_table(
                index="period", columns=group_col, values=value_col, aggfunc=agg_fn
            ).fillna(0)
            st.line_chart(pivot)

# ----------------------------------------------------------------------------
# Task 3: Predictive Model (regression + classification + comparison)
# ----------------------------------------------------------------------------
elif task == "Predictive Model":
    st.subheader("Predictive Modeling")

    problem = st.radio("Problem type", ["Regression", "Classification"], horizontal=True)

    # ---- target selection -------------------------------------------------
    if problem == "Regression":
        if not num_cols:
            st.error("You need at least one numeric column to use as the target.")
            st.stop()
        target = st.selectbox("Select Target Column", num_cols)
        y_raw = df[target]
    else:
        candidates = [c for c in df.columns if c in num_cols or df[c].nunique() <= 20]
        if not candidates:
            st.error("No suitable target column found for classification.")
            st.stop()
        target = st.selectbox("Select Target Column", candidates)
        y_raw = df[target]
        if target in num_cols and df[target].nunique() > 20:
            threshold = st.number_input(
                f"Label rows 'High' when {target} is above (otherwise 'Low')",
                value=float(df[target].median()),
            )
            y_raw = pd.Series(np.where(df[target] > threshold, "High", "Low"), index=df.index).where(
                df[target].notna()
            )

    # ---- feature + model settings ----------------------------------------
    available = [c for c in df.columns if c != target]
    default_features = [c for c in num_cols if c != target]
    features = st.multiselect("Select Feature Columns", available, default=default_features)

    col_a, col_b, col_c = st.columns(3)
    test_size = col_a.slider("Test Size", 0.1, 0.5, 0.2)
    n_estimators = col_b.slider("Number of Trees / Rounds", 50, 500, 100, step=50)
    max_depth = col_c.slider("Max Depth (0 = default)", 0, 30, 0)

    all_models = list(get_models(problem, n_estimators, max_depth).keys())
    chosen_models = st.multiselect(
        "Models to train and compare", all_models, default=all_models
    )
    if not HAS_XGB:
        st.caption("Tip: run `pip install xgboost` to add XGBoost to the comparison.")

    # ---- training ---------------------------------------------------------
    if st.button("Train & Compare Models", type="primary"):
        if not features:
            st.error("Select at least one feature.")
            st.stop()
        if not chosen_models:
            st.error("Select at least one model.")
            st.stop()

        mask = y_raw.notna()
        data = df.loc[mask]
        y = y_raw.loc[mask]
        X, dropped = prepare_features(data, features)

        if X.shape[1] == 0:
            st.error("No usable feature columns after preprocessing.")
            st.stop()
        if dropped:
            st.warning(f"Dropped high-cardinality text columns: {', '.join(dropped)}")

        classes = None
        if problem == "Classification":
            encoder = LabelEncoder()
            y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
            classes = list(encoder.classes_)
            if len(classes) < 2:
                st.error("The target needs at least two classes.")
                st.stop()
            stratify = y if y.value_counts().min() >= 2 else None
        else:
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        models = get_models(problem, n_estimators, max_depth)
        fitted, predictions, rows = {}, {}, []
        progress = st.progress(0.0, text="Training models...")
        for i, name in enumerate(chosen_models):
            try:
                start = time.time()
                pipe = models[name]
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                elapsed = time.time() - start
                fitted[name] = pipe
                predictions[name] = preds
                rows.append({"Model": name, **evaluate(problem, y_test, preds), "Train time (s)": elapsed})
            except Exception as e:
                st.warning(f"{name} failed: {e}")
            progress.progress((i + 1) / len(chosen_models), text=f"Trained {name}")
        progress.empty()

        if not fitted:
            st.error("No model could be trained with these settings.")
            st.stop()

        # metadata for the single-row prediction form
        kept_cols = [c for c in features if c not in dropped]
        input_meta = {}
        for col in kept_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                med = data[col].median()
                med = 0.0 if pd.isna(med) else med
                is_int = pd.api.types.is_integer_dtype(data[col])
                input_meta[col] = {
                    "type": "numeric",
                    "default": int(round(med)) if is_int else float(med),
                    "is_int": is_int,
                }
            else:
                input_meta[col] = {
                    "type": "categorical",
                    "options": sorted(data[col].dropna().astype(str).unique().tolist()),
                }

        st.session_state["result"] = {
            "sig": (tuple(df.columns), df.shape),
            "problem": problem,
            "target": target,
            "metrics": pd.DataFrame(rows).set_index("Model"),
            "models": fitted,
            "preds": predictions,
            "y_test": y_test.values,
            "classes": classes,
            "X_columns": list(X.columns),
            "input_cols": kept_cols,
            "input_meta": input_meta,
        }

    # ---- results ----------------------------------------------------------
    res = st.session_state.get("result")
    if res and res["sig"] == (tuple(df.columns), df.shape):
        st.success("Models trained successfully")
        metrics = res["metrics"]
        primary = "R²" if res["problem"] == "Regression" else "F1"
        best_name = metrics[primary].idxmax()

        st.subheader("Model Comparison")
        st.dataframe(metrics.round(4))
        st.caption(f"Best model by {primary}: **{best_name}**")
        st.bar_chart(metrics[primary])

        trained_names = list(res["models"].keys())
        inspect = st.selectbox(
            "Inspect model", trained_names, index=trained_names.index(best_name)
        )
        preds = res["preds"][inspect]
        y_test = res["y_test"]

        g1, g2 = st.columns(2)
        with g1:
            if res["problem"] == "Regression":
                st.write("**Actual vs Predicted**")
                fig, ax = plt.subplots()
                ax.scatter(y_test, preds, alpha=0.5)
                lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
                ax.plot(lims, lims, "r--")
                ax.set_xlabel(f"Actual {res['target']}")
                ax.set_ylabel(f"Predicted {res['target']}")
                st.pyplot(fig)
            else:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, preds, labels=range(len(res["classes"])))
                fig, ax = plt.subplots()
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=res["classes"],
                    yticklabels=res["classes"],
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
        with g2:
            importance = get_importance(res["models"][inspect], res["X_columns"])
            if importance is not None:
                st.write("**Top Feature Importances**")
                top = importance.head(10)
                fig, ax = plt.subplots()
                sns.barplot(x=top.values, y=top.index, ax=ax, color="seagreen")
                ax.set_xlabel("Importance")
                st.pyplot(fig)

        if res["problem"] == "Classification":
            actual_lbl = [res["classes"][i] for i in y_test]
            pred_lbl = [res["classes"][i] for i in preds]
            results_df = pd.DataFrame({"Actual": actual_lbl, "Predicted": pred_lbl})
        else:
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": preds}).round(2)
        st.download_button(
            "Download predictions (CSV)",
            results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{inspect.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

        # ---- single-row prediction ----------------------------------------
        st.subheader(f"Predict a Single Row ({inspect})")
        with st.form("single_prediction"):
            inputs = {}
            form_cols = st.columns(3)
            for i, col in enumerate(res["input_cols"]):
                meta = res["input_meta"][col]
                with form_cols[i % 3]:
                    if meta["type"] == "numeric":
                        inputs[col] = st.number_input(col, value=meta["default"])
                    else:
                        inputs[col] = st.selectbox(col, meta["options"])
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = pd.get_dummies(pd.DataFrame([inputs]))
            row = row.reindex(columns=res["X_columns"], fill_value=0).astype(float)
            model = res["models"][inspect]
            pred = model.predict(row)[0]
            if res["problem"] == "Regression":
                st.metric(f"Predicted {res['target']}", f"{float(pred):,.2f}")
            else:
                st.metric(f"Predicted {res['target']}", res["classes"][int(pred)])
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(row)[0]
                    st.write("**Class probabilities**")
                    st.dataframe(
                        pd.DataFrame({"Class": res["classes"], "Probability": proba.round(3)}).set_index(
                            "Class"
                        )
                    )
