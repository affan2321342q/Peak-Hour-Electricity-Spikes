import pandas as pd
import numpy as np

def generate_dorm_energy():
    np.random.seed(42)
    days = 14
    hours = days * 24
    dates = pd.date_range(end=pd.Timestamp.now().floor('D') + pd.Timedelta(days=1) - pd.Timedelta(hours=1), periods=hours, freq='h')
    
    # Base pattern: low at night, peak in evening (18-22)
    base_pattern = [20, 18, 17, 16, 17, 19, 25, 30, 35, 38, 40, 42, 40, 38, 36, 38, 45, 55, 60, 65, 58, 45, 30, 25]
    base_data = np.tile(base_pattern, days)
    noise = np.random.normal(0, 5, hours)
    consumption = np.maximum(base_data + noise, 0)
    
    df = pd.DataFrame({'timestamp': dates, 'consumption': consumption})
    df.to_csv('dorm_energy.csv', index=False)
    print("Created dorm_energy.csv")

def generate_classroom_wifi_energy():
    np.random.seed(42)
    days = 14
    hours = days * 24
    dates = pd.date_range(end=pd.Timestamp.now().floor('h'), periods=hours, freq='h')
    
    occupancy_base = np.tile([5, 2, 1, 0, 1, 5, 20, 50, 80, 100, 110, 105, 95, 100, 110, 80, 50, 30, 20, 15, 10, 8, 5, 5], days)
    occupancy = np.maximum(occupancy_base + np.random.normal(0, 5, hours), 0)
    
    base_load = 10
    electricity_draw = base_load + (occupancy * 0.3) + np.random.normal(0, 2, hours)
    electricity_draw = np.maximum(electricity_draw, 0)
    
    df = pd.DataFrame({'timestamp': dates, 'occupancy': occupancy, 'electricity': electricity_draw})
    df.to_csv('classroom_wifi_energy.csv', index=False)
    print("Created classroom_wifi_energy.csv")

if __name__ == "__main__":
    generate_dorm_energy()
    generate_classroom_wifi_energy()
