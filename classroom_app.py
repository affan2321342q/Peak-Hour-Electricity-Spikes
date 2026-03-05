import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

# Load dataset from CSV
df = pd.read_csv('classroom_wifi_energy.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Classroom electricity Usage Forecasting", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    html.P("ARIMA model forecasting the next few hours of room electricity draw based on Wi-Fi occupancy logs.", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
    
    html.Div([
        html.Button("Refresh Forecast", id="refresh-button", n_clicks=0, style={
            'padding': '10px 20px', 'fontSize': '16px', 'backgroundColor': '#00d2ff', 'border': 'none', 
            'color': '#fff', 'borderRadius': '5px', 'cursor': 'pointer', 'display': 'block', 'margin': '0 auto'
        }),
    ], style={'padding': '20px'}),

    dcc.Graph(id='live-forecast-graph', style={'height': '70vh'})
])

@app.callback(
    Output('live-forecast-graph', 'figure'),
    [Input('refresh-button', 'n_clicks')]
)
def update_forecast(n_clicks):
    # Use the last 72 hours for training to keep it fast while capturing daily seasonality
    train_data = df.iloc[-72:]
    
    # Exogenous variable: occupancy
    exog_train = train_data['occupancy']
    endog_train = train_data['electricity']
    
    # Train ARIMAX (ARIMA with exogenous variable)
    # Using order (2,1,2) for illustration. In production, this can be auto-tuned.
    model = ARIMA(endog_train, exog=exog_train, order=(2, 1, 2))
    fit_model = model.fit()
    
    # Forecast the next 6 hours
    num_steps = 6
    # Assume occupancy for the next hours will be similar to the same time yesterday
    last_occupancy = df['occupancy'].iloc[-24: -24 + num_steps].values
    
    forecast = fit_model.get_forecast(steps=num_steps, exog=last_occupancy)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    
    future_dates = pd.date_range(start=train_data.index[-1] + pd.Timedelta(hours=1), periods=num_steps, freq='H')
    
    fig = go.Figure()
    
    # Historical Electricity
    fig.add_trace(go.Scatter(
        x=train_data.index, y=train_data['electricity'],
        mode='lines+markers', name='Actual Electricity (kW)', line=dict(color='#00d2ff', width=3)
    ))
    
    # Historical Occupancy (scaled for visualization)
    fig.add_trace(go.Scatter(
        x=train_data.index, y=train_data['occupancy'] * 0.2, # Scaled down
        mode='lines', name='Wi-Fi Occupancy (Scaled)', line=dict(color='gray', dash='dot')
    ))
    
    # Forecast Mean
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast_mean,
        mode='lines+markers', name='Forecasted Electricity', line=dict(color='#ff7e5f', width=3, dash='dash')
    ))
    
    # Confidence Intervals
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0])[::-1],
        fill='toself',
        fillcolor='rgba(255, 126, 95, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title="Predictive Electricity Draw vs Wi-Fi Occupancy (ARIMAX)",
        xaxis_title="Time",
        yaxis_title="Power Consumption (kW)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)
