import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

# Load dataset from CSV
df = pd.read_csv('dorm_energy.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 1. Moving Average Smoothing
df['moving_avg'] = df['consumption'].rolling(window=3).mean().bfill()

# Split into historical (7 days) and current (today)
df_hist = df.iloc[:7 * 24].copy()
df_today_full = df.iloc[7 * 24:].copy()

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dormitory Peak Hour Electricity Dashboard", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    html.P("Live stream of today's consumption. Historical data is smoothed using Moving Average and Linear Regression is used to predict the evening peak.", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    dcc.Graph(id='live-graph', style={'height': '70vh'}),
    dcc.Interval(
        id='interval-component',
        interval=1500, # 1.5 seconds per hour
        n_intervals=0
    )
])

# 2. Linear Regression to Predict Evening Peaks
# Fit model once on historical data to capture the daily pattern
# We one-hot encode the hour to allow linear regression to model the non-linear daily shape
X_hist = pd.get_dummies(df_hist['hour'], prefix='hr').join(df_hist[['is_weekend']])
y_hist = df_hist['consumption'] # We can also target smoothed 'moving_avg', but let's target actual consumption
model = LinearRegression()
model.fit(X_hist, y_hist)

# Predict for today using the same features
X_today = pd.get_dummies(df_today_full['hour'], prefix='hr').join(df_today_full[['is_weekend']])
# Ensure same columns
missing_cols = set(X_hist.columns) - set(X_today.columns)
for c in missing_cols:
    X_today[c] = 0
X_today = X_today[X_hist.columns]

df_today_full['predicted'] = model.predict(X_today)

@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    # Simulate time passing today: reveal 1 hour per tick (n=0 to 23)
    current_hour_idx = n % 24
    
    # We only have actual data up to current_hour_idx for today
    df_today_live = df_today_full.iloc[:current_hour_idx + 1]
    
    # 3. Visualize trends on a live Plotly dashboard
    fig = go.Figure()
    
    # Plot past 24h historical (just for context)
    recent_hist = df_hist.iloc[-24:]
    fig.add_trace(go.Scatter(
        x=recent_hist['timestamp'], y=recent_hist['moving_avg'],
        mode='lines', name='Yesterday (Smoothed)', line=dict(color='gray', dash='dash')
    ))
    
    # Plot today's actual data so far
    fig.add_trace(go.Scatter(
        x=df_today_live['timestamp'], y=df_today_live['consumption'],
        mode='lines+markers', name='Today Actual (Live)', line=dict(color='#00d2ff', width=3)
    ))

    # Plot today's Moving Average
    fig.add_trace(go.Scatter(
        x=df_today_live['timestamp'], y=df_today_live['moving_avg'],
        mode='lines', name='Today Moving Avg (w=3)', line=dict(color='#3a7bd5', width=2)
    ))
    
    # Plot today's Prediction (Whole day)
    fig.add_trace(go.Scatter(
        x=df_today_full['timestamp'], y=df_today_full['predicted'],
        mode='lines', name='Predicted by LinReg', line=dict(color='#ff7e5f', dash='dot')
    ))
    
    # Highlight Predicted Evening Peak (18:00 - 22:00)
    evening_hours = df_today_full[(df_today_full['hour'] >= 18) & (df_today_full['hour'] <= 22)]
    pred_peak_idx = evening_hours['predicted'].idxmax()
    pred_peak_time = df_today_full.loc[pred_peak_idx, 'timestamp']
    pred_peak_val = df_today_full.loc[pred_peak_idx, 'predicted']
    
    fig.add_trace(go.Scatter(
        x=[pred_peak_time], y=[pred_peak_val],
        mode='markers', name='Predicted Evening Peak', 
        marker=dict(color='#feb47b', size=14, symbol='star', line=dict(color='white', width=1))
    ))

    # Add annotation for the peak
    fig.add_annotation(
        x=pred_peak_time, y=pred_peak_val,
        text=f"Predicted Evening Peak: {pred_peak_val:.1f} kW",
        showarrow=True, arrowhead=2, ax=-60, ay=-40,
        font=dict(color='white'),
        bgcolor='rgba(255, 126, 95, 0.5)'
    )

    fig.update_layout(
        title="Live Hourly Meter Data & Evening Peak Prediction",
        xaxis_title="Time",
        yaxis_title="Power Consumption (kW)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(range=[recent_hist['timestamp'].min(), df_today_full['timestamp'].max() + pd.Timedelta(hours=2)])
    fig.update_yaxes(range=[0, df['consumption'].max() + 10])
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
