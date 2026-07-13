"""
COVID-19 India: Trend Analysis, Rate Calculation, and 7-Day Forecast
Dataset: covid_19_india.csv
"""

import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import logging

# Hide Prophet warnings
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# ============================================================
# STEP 1: Load and Clean Data
# ============================================================

df = pd.read_csv(r"C:\Users\sazed\Downloads\covid_19\covid_19_india.csv")

df["Date"] = pd.to_datetime(df["Date"])

state_fixes = {
    "Telengana": "Telangana",
    "Bihar****": "Bihar",
    "Madhya Pradesh***": "Madhya Pradesh",
    "Himanchal Pradesh": "Himachal Pradesh",
    "Karanataka": "Karnataka",
    "Maharashtra***": "Maharashtra",
    "Nagaland#": "Nagaland",
}

df["State/UnionTerritory"] = df["State/UnionTerritory"].replace(state_fixes)

df = df[
    ~df["State/UnionTerritory"].isin(
        ["Unassigned", "Cases being reassigned to states"]
    )
]

df = df[
    ["Date", "State/UnionTerritory", "Cured", "Deaths", "Confirmed"]
]

df.columns = [
    "Date",
    "State",
    "Cured",
    "Deaths",
    "Confirmed",
]

# ============================================================
# STEP 2: National Daily Statistics
# ============================================================

national = (
    df.groupby("Date")[["Cured", "Deaths", "Confirmed"]]
    .sum()
    .reset_index()
)

national = national.sort_values("Date")

national["New_Confirmed"] = (
    national["Confirmed"].diff().fillna(0).clip(lower=0)
)

national["New_Deaths"] = (
    national["Deaths"].diff().fillna(0).clip(lower=0)
)

national["New_Cured"] = (
    national["Cured"].diff().fillna(0).clip(lower=0)
)

national["Active"] = (
    national["Confirmed"]
    - national["Cured"]
    - national["Deaths"]
)

national["Recovery_Rate_%"] = (
    national["Cured"] / national["Confirmed"] * 100
).round(2)

national["Death_Rate_%"] = (
    national["Deaths"] / national["Confirmed"] * 100
).round(2)

# ============================================================
# STEP 3: Charts
# ============================================================

# Chart 1
fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=national["Date"],
        y=national["Confirmed"],
        name="Confirmed",
    )
)

fig1.add_trace(
    go.Scatter(
        x=national["Date"],
        y=national["Cured"],
        name="Recovered",
    )
)

fig1.add_trace(
    go.Scatter(
        x=national["Date"],
        y=national["Deaths"],
        name="Deaths",
    )
)

fig1.update_layout(
    title="India: Cumulative COVID-19 Cases",
    xaxis_title="Date",
    yaxis_title="Cases",
)

fig1.write_html("chart1_cumulative.html")

# ------------------------------------------------------------

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=national["Date"],
        y=national["New_Confirmed"],
        name="Daily New Cases",
    )
)

fig2.update_layout(
    title="India: Daily New Confirmed Cases",
    xaxis_title="Date",
    yaxis_title="New Cases",
)

fig2.write_html("chart2_daily_new.html")

# ------------------------------------------------------------

fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(
        x=national["Date"],
        y=national["Recovery_Rate_%"],
        name="Recovery Rate %",
    )
)

fig3.add_trace(
    go.Scatter(
        x=national["Date"],
        y=national["Death_Rate_%"],
        name="Death Rate %",
    )
)

fig3.update_layout(
    title="Recovery Rate vs Death Rate",
    xaxis_title="Date",
    yaxis_title="Percentage",
)

fig3.write_html("chart3_rates.html")

# ------------------------------------------------------------

latest_date = df["Date"].max()

latest = (
    df[df["Date"] == latest_date]
    .sort_values("Confirmed", ascending=False)
    .head(10)
)

fig4 = go.Figure()

fig4.add_trace(
    go.Bar(
        x=latest["State"],
        y=latest["Confirmed"],
    )
)

fig4.update_layout(
    title=f"Top 10 States by Confirmed Cases ({latest_date.date()})",
    xaxis_title="State",
    yaxis_title="Confirmed Cases",
)

fig4.write_html("chart4_top_states.html")

# ============================================================
# STEP 4: Prophet Forecast
# ============================================================

prophet_df = national[
    ["Date", "New_Confirmed"]
].rename(
    columns={
        "Date": "ds",
        "New_Confirmed": "y",
    }
)

prophet_df["y"] = prophet_df["y"].clip(lower=0)

recent = prophet_df.tail(90).reset_index(drop=True)

model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=False,
    changepoint_prior_scale=0.1,
)

model.fit(recent)

future = model.make_future_dataframe(periods=7)

forecast = model.predict(future)

forecast["yhat"] = forecast["yhat"].clip(lower=0)
forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

last_actual = recent["ds"].max()

historical = forecast[forecast["ds"] <= last_actual]
future_forecast = forecast[forecast["ds"] > last_actual]

print("\n7-Day Forecast\n")

print(
    future_forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        ]
    ]
)

# ============================================================
# STEP 5: Forecast Chart
# ============================================================

fig5 = go.Figure()

fig5.add_trace(
    go.Scatter(
        x=recent["ds"],
        y=recent["y"],
        mode="lines+markers",
        name="Actual",
    )
)

fig5.add_trace(
    go.Scatter(
        x=historical["ds"],
        y=historical["yhat"],
        name="Model Fit",
    )
)

fig5.add_trace(
    go.Scatter(
        x=future_forecast["ds"],
        y=future_forecast["yhat"],
        name="Forecast",
    )
)

fig5.add_trace(
    go.Scatter(
        x=list(future_forecast["ds"])
        + list(future_forecast["ds"][::-1]),
        y=list(future_forecast["yhat_upper"])
        + list(future_forecast["yhat_lower"][::-1]),
        fill="toself",
        fillcolor="rgba(255,0,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval",
    )
)

fig5.update_layout(
    title="India Daily New Cases with 7-Day Forecast",
    xaxis_title="Date",
    yaxis_title="New Confirmed Cases",
)

fig5.write_html("chart5_forecast.html")

# ============================================================
# STEP 6: Export CSV
# ============================================================

national.to_csv("national_daily.csv", index=False)

print("\nAnalysis completed successfully!")
print("Generated files:")
print("✓ chart1_cumulative.html")
print("✓ chart2_daily_new.html")
print("✓ chart3_rates.html")
print("✓ chart4_top_states.html")
print("✓ chart5_forecast.html")
print("✓ national_daily.csv")