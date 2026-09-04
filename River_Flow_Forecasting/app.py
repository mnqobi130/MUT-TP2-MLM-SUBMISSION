import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="River Flow Forecasting Dashboard",
    page_icon="💧",
    layout="wide"
)

st.title("💧 River Flow Forecasting Dashboard")

st.write(
    "This dashboard uses historical river flow and water-level data "
    "together with a Random Forest machine-learning model to forecast "
    "river flow for the next month."
)

# ---------------------------------------------------------
# FILE NAMES
# ---------------------------------------------------------

DATA_FILE = "W1H004PK.CSV"
MODEL_FILE = "river_flow_rf_model.joblib"
FEATURE_FILE = "river_flow_feature_cols.joblib"

HIGH_FLOW_THRESHOLD = 2.0


# ---------------------------------------------------------
# LOAD RAW DATA
# This follows the same method used in your Jupyter Notebook
# ---------------------------------------------------------

@st.cache_data
def load_raw(path):

    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    # Find where the real CSV header starts
    header_idx = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("Year,Date,Time")
    )

    records = []

    for line in lines[header_idx + 1:]:

        line = line.rstrip("\n")

        if (
            not line.strip()
            or line.strip().startswith("Explanation")
            or "\x0c" in line
        ):
            continue

        parts = line.split(",")

        if len(parts) < 5:
            continue

        records.append({
            "date_raw": parts[1].strip(),
            "level_raw": parts[3].strip(),
            "flow_raw": parts[4].strip(),
            "flag": parts[5].strip() if len(parts) > 5 else "",
        })

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(
        df["date_raw"],
        format="%Y%m%d"
    )

    df["level"] = pd.to_numeric(
        df["level_raw"],
        errors="coerce"
    )

    df["flow"] = pd.to_numeric(
        df["flow_raw"],
        errors="coerce"
    )

    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "level", "flow", "flag"]]


# ---------------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    feature_columns = joblib.load(FEATURE_FILE)

    return model, feature_columns


# ---------------------------------------------------------
# CREATE MODEL FEATURES
# Same feature engineering as your notebook
# ---------------------------------------------------------

def create_features(df):

    # Remove flagged observations
    clean = df[df["flag"] == ""].copy()

    clean["ym"] = clean["date"].dt.to_period("M")

    grid = clean.set_index("ym").sort_index()

    # Create complete monthly timeline
    full_range = pd.period_range(
        grid.index.min(),
        grid.index.max(),
        freq="M"
    )

    grid = grid.reindex(full_range)

    grid.index.name = "ym"

    # Log transformation of flow
    grid["log_flow"] = np.log1p(grid["flow"])

    # Date features
    grid["year"] = grid.index.year
    grid["month"] = grid.index.month

    grid["month_sin"] = np.sin(
        2 * np.pi * grid["month"] / 12
    )

    grid["month_cos"] = np.cos(
        2 * np.pi * grid["month"] / 12
    )

    # Lag features
    for lag in [1, 2, 3, 6, 12]:

        grid[f"flow_lag_{lag}"] = (
            grid["log_flow"].shift(lag)
        )

        grid[f"level_lag_{lag}"] = (
            grid["level"].shift(lag)
        )

    # Rolling statistics
    for window in [3, 6]:

        grid[f"flow_rolling_mean_{window}"] = (
            grid["log_flow"]
            .shift(1)
            .rolling(window)
            .mean()
        )

        grid[f"flow_rolling_max_{window}"] = (
            grid["log_flow"]
            .shift(1)
            .rolling(window)
            .max()
        )

        grid[f"flow_rolling_std_{window}"] = (
            grid["log_flow"]
            .shift(1)
            .rolling(window)
            .std()
        )

    return grid


# ---------------------------------------------------------
# CREATE FEATURES FOR NEXT MONTH
# ---------------------------------------------------------

def create_next_month_features(grid, feature_columns):

    latest_month = grid.index.max()

    next_month = latest_month + 1

    new_grid = grid.copy()

    # Add empty next month
    new_grid.loc[next_month, "flow"] = np.nan
    new_grid.loc[next_month, "level"] = np.nan

    new_grid.loc[next_month, "year"] = next_month.year
    new_grid.loc[next_month, "month"] = next_month.month

    new_grid.loc[next_month, "month_sin"] = np.sin(
        2 * np.pi * next_month.month / 12
    )

    new_grid.loc[next_month, "month_cos"] = np.cos(
        2 * np.pi * next_month.month / 12
    )

    # Lag features
    for lag in [1, 2, 3, 6, 12]:

        previous_month = next_month - lag

        if previous_month in new_grid.index:

            previous_flow = new_grid.loc[
                previous_month,
                "flow"
            ]

            previous_level = new_grid.loc[
                previous_month,
                "level"
            ]

            if pd.notna(previous_flow):

                new_grid.loc[
                    next_month,
                    f"flow_lag_{lag}"
                ] = np.log1p(previous_flow)

            else:

                new_grid.loc[
                    next_month,
                    f"flow_lag_{lag}"
                ] = np.nan

            new_grid.loc[
                next_month,
                f"level_lag_{lag}"
            ] = previous_level

    # Rolling flow statistics
    for window in [3, 6]:

        previous_months = [
            next_month - i
            for i in range(1, window + 1)
        ]

        flows = []

        for month in previous_months:

            if month in new_grid.index:

                flow_value = new_grid.loc[
                    month,
                    "flow"
                ]

                if pd.notna(flow_value):
                    flows.append(
                        np.log1p(flow_value)
                    )

        if len(flows) == window:

            new_grid.loc[
                next_month,
                f"flow_rolling_mean_{window}"
            ] = np.mean(flows)

            new_grid.loc[
                next_month,
                f"flow_rolling_max_{window}"
            ] = np.max(flows)

            new_grid.loc[
                next_month,
                f"flow_rolling_std_{window}"
            ] = np.std(
                flows,
                ddof=1
            )

        else:

            new_grid.loc[
                next_month,
                f"flow_rolling_mean_{window}"
            ] = np.nan

            new_grid.loc[
                next_month,
                f"flow_rolling_max_{window}"
            ] = np.nan

            new_grid.loc[
                next_month,
                f"flow_rolling_std_{window}"
            ] = np.nan

    next_features = new_grid.loc[
        [next_month],
        feature_columns
    ]

    return next_month, next_features


# ---------------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------------

try:

    df = load_raw(DATA_FILE)

    model, feature_columns = load_model()

    grid = create_features(df)

except FileNotFoundError as error:

    st.error(
        "One of the required files could not be found."
    )

    st.write(
        "Make sure app.py, W1H004PK.CSV, "
        "river_flow_rf_model.joblib and "
        "river_flow_feature_cols.joblib "
        "are all in the same folder."
    )

    st.stop()

except Exception as error:

    st.error("The dashboard could not load.")

    st.write(error)

    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("💧 Dashboard Menu")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "Overview",
        "Historical Data",
        "Flow Forecast",
        "Model Information"
    ]
)


# ---------------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------------

if page == "Overview":

    st.header("🌊 River Monitoring Overview")

    valid_flow = df["flow"].dropna()

    valid_level = df["level"].dropna()

    latest_valid = df.dropna(
        subset=["flow", "level"]
    ).iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Latest Flow",
            f"{latest_valid['flow']:.3f} cumec"
        )

    with col2:

        st.metric(
            "Latest Water Level",
            f"{latest_valid['level']:.3f} m"
        )

    with col3:

        st.metric(
            "Maximum Flow",
            f"{valid_flow.max():.3f} cumec"
        )

    with col4:

        st.metric(
            "Average Flow",
            f"{valid_flow.mean():.3f} cumec"
        )

    st.write(
        f"Latest available observation: "
        f"**{latest_valid['date'].date()}**"
    )

    st.subheader("Monthly River Flow")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["date"],
        df["flow"]
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Flow (cumec)")

    ax.set_title(
        "Historical Monthly Maximum River Flow"
    )

    ax.grid(True)

    st.pyplot(fig)


# ---------------------------------------------------------
# HISTORICAL DATA PAGE
# ---------------------------------------------------------

elif page == "Historical Data":

    st.header("📊 Historical River Data")

    st.write(
        "The dataset contains monthly river-flow "
        "and water-level observations."
    )

    # Year selection
    minimum_year = int(df["date"].dt.year.min())

    maximum_year = int(df["date"].dt.year.max())

    year_range = st.slider(
        "Select year range",
        minimum_year,
        maximum_year,
        (
            max(minimum_year, maximum_year - 20),
            maximum_year
        )
    )

    filtered_df = df[
        (
            df["date"].dt.year
            >= year_range[0]
        )
        &
        (
            df["date"].dt.year
            <= year_range[1]
        )
    ]

    st.subheader("River Flow")

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        filtered_df["date"],
        filtered_df["flow"]
    )

    ax1.set_xlabel("Date")

    ax1.set_ylabel("Flow (cumec)")

    ax1.set_title(
        "Historical River Flow"
    )

    ax1.grid(True)

    st.pyplot(fig1)

    st.subheader("Water Level")

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        filtered_df["date"],
        filtered_df["level"]
    )

    ax2.set_xlabel("Date")

    ax2.set_ylabel("Water Level (m)")

    ax2.set_title(
        "Historical Water Level"
    )

    ax2.grid(True)

    st.pyplot(fig2)

    st.subheader("Dataset")

    display_df = filtered_df[
        [
            "date",
            "level",
            "flow",
            "flag"
        ]
    ]

    st.dataframe(
        display_df,
        use_container_width=True
    )


# ---------------------------------------------------------
# FORECAST PAGE
# ---------------------------------------------------------

elif page == "Flow Forecast":

    st.header("🔮 One-Month-Ahead River Flow Forecast")

    st.write(
        "The Random Forest model uses previous "
        "river-flow values, previous water levels, "
        "rolling statistics, seasonality and year "
        "to forecast river flow for the next month."
    )

    next_month, next_features = (
        create_next_month_features(
            grid,
            feature_columns
        )
    )

    latest_month = grid.index.max()

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"Latest data month: "
            f"{latest_month.strftime('%B %Y')}"
        )

    with col2:

        st.info(
            f"Forecast month: "
            f"{next_month.strftime('%B %Y')}"
        )

    if next_features.isna().any().any():

        st.warning(
            "The latest data does not contain enough "
            "continuous historical observations to "
            "create all model features."
        )

        missing_features = next_features.columns[
            next_features.isna().any()
        ].tolist()

        st.write(
            "Missing features:",
            missing_features
        )

    else:

        if st.button(
            "Predict Next Month's River Flow",
            type="primary"
        ):

            # Model predicts log1p(flow)
            prediction_log = model.predict(
                next_features
            )[0]

            predicted_flow = np.expm1(
                np.clip(
                    prediction_log,
                    -10,
                    10
                )
            )

            predicted_flow = max(
                0,
                predicted_flow
            )

            st.subheader(
                "Forecast Result"
            )

            st.metric(
                f"Predicted Flow for "
                f"{next_month.strftime('%B %Y')}",
                f"{predicted_flow:.3f} cumec"
            )

            # Risk classification
            if predicted_flow >= HIGH_FLOW_THRESHOLD:

                st.error(
                    "🚨 HIGH-FLOW WARNING"
                )

                st.write(
                    "The predicted flow is above "
                    f"the model's high-flow threshold "
                    f"of {HIGH_FLOW_THRESHOLD} cumec."
                )

            else:

                st.success(
                    "✅ NORMAL / LOWER FLOW CONDITION"
                )

                st.write(
                    "The predicted flow is below "
                    f"the high-flow threshold of "
                    f"{HIGH_FLOW_THRESHOLD} cumec."
                )

            # Recent historical data + prediction
            recent = grid.tail(24).copy()

            recent_dates = (
                recent.index.to_timestamp()
            )

            recent_flows = recent["flow"]

            prediction_date = (
                next_month.to_timestamp()
            )

            fig3, ax3 = plt.subplots(
                figsize=(12, 5)
            )

            ax3.plot(
                recent_dates,
                recent_flows,
                marker="o",
                label="Historical Flow"
            )

            ax3.scatter(
                prediction_date,
                predicted_flow,
                s=100,
                label="Forecast"
            )

            ax3.set_xlabel("Date")

            ax3.set_ylabel(
                "Flow (cumec)"
            )

            ax3.set_title(
                "Recent River Flow and "
                "Next-Month Forecast"
            )

            ax3.legend()

            ax3.grid(True)

            st.pyplot(fig3)


# ---------------------------------------------------------
# MODEL INFORMATION PAGE
# ---------------------------------------------------------

elif page == "Model Information":

    st.header("🤖 Machine Learning Model")

    st.write(
        """
        The forecasting model used in this dashboard
        is a tuned **Random Forest Regressor**.

        The model predicts the logarithm of river flow
        and the prediction is converted back to the
        original flow measurement in cumec.
        """
    )

    st.subheader("Model Features")

    feature_table = pd.DataFrame({
        "Feature": feature_columns
    })

    st.dataframe(
        feature_table,
        use_container_width=True
    )

    st.subheader("Feature Groups")

    st.write(
        """
        The model uses:

        - River-flow lag values from 1, 2, 3, 6 and 12 months earlier
        - Water-level lag values from 1, 2, 3, 6 and 12 months earlier
        - 3-month rolling mean, maximum and standard deviation
        - 6-month rolling mean, maximum and standard deviation
        - Month seasonality
        - Year
        """
    )

    st.subheader("High-Flow Threshold")

    st.metric(
        "High Flow Threshold",
        f"{HIGH_FLOW_THRESHOLD} cumec"
    )

    st.write(
        "A predicted river flow equal to or greater "
        "than 2.0 cumec is marked as a high-flow event."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "River Flow Forecasting using Machine Learning | "
    "Random Forest Regression"
)