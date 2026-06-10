import os
import sys
import torch
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
TIMESFM_SRC = os.path.join(ROOT_DIR, "timesfm", "src")
if TIMESFM_SRC not in sys.path:
    sys.path.append(TIMESFM_SRC)

import timesfm

class AdaptiveForecastingEngine:
    def __init__(self):
        self.model = None
        
        # Exact Week-Level Festival Map matching the 'Week Start Date' timestamps
        # Mapping format: (Year, WeekNumber) -> Festival Name
        self.weekly_festival_calendar = {
            # 2023 Weekly Benchmarks
            (2023, 2): "Makar Sankranti / Pongal", (2023, 4): "Republic Day week",
            (2023, 6): "Valentine's Week", (2023, 10): "Holi / Gudi Padwa",
            (2023, 16): "Akshaya Tritiya", (2023, 35): "Raksha Bandhan",
            (2023, 36): "Janmashtami", (2023, 38): "Ganesh Chaturthi",
            (2023, 42): "Navratri", (2023, 43): "Dussehra",
            (2023, 45): "Dhanteras / Diwali", (2023, 46): "Diwali Padwa / Tulsi Vivah",
            (2023, 52): "Christmas / Year End New Year Peak",

            # 2024 Weekly Benchmarks
            (2024, 2): "Makar Sankranti / Pongal", (2024, 4): "Republic Day week",
            (2024, 7): "Valentine's Week", (2024, 12): "Holi / Gudi Padwa",
            (2024, 19): "Akshaya Tritiya", (2024, 33): "Raksha Bandhan",
            (2024, 35): "Janmashtami", (2024, 36): "Ganesh Chaturthi",
            (2024, 41): "Navratri", (2024, 42): "Dussehra",
            (2024, 44): "Dhanteras / Diwali", (2024, 45): "Diwali Padwa / Tulsi Vivah",
            (2024, 52): "Christmas / Year End New Year Peak",

            # 2025 Weekly Benchmarks
            (2025, 3): "Makar Sankranti / Pongal", (2025, 4): "Republic Day week",
            (2025, 7): "Valentine's Week", (2025, 11): "Holi / Gudi Padwa",
            (2025, 18): "Akshaya Tritiya", (2025, 32): "Raksha Bandhan",
            (2025, 33): "Janmashtami", (2025, 36): "Ganesh Chaturthi",
            (2025, 40): "Navratri", (2025, 41): "Dussehra",
            (2025, 43): "Dhanteras / Diwali", (2025, 44): "Diwali Padwa / Tulsi Vivah",
            (2025, 52): "Christmas / Year End New Year Peak",

            # 2026 Future Projections Window
            (2026, 3): "Makar Sankranti / Pongal", (2026, 4): "Republic Day week",
            (2026, 7): "Valentine's Week", (2026, 11): "Holi / Gudi Padwa",
            (2026, 16): "Akshaya Tritiya", (2026, 35): "Raksha Bandhan",
            (2026, 36): "Janmashtami", (2026, 38): "Ganesh Chaturthi",
            (2026, 41): "Navratri", (2026, 42): "Dussehra",
            (2026, 44): "Dhanteras / Diwali", (2026, 45): "Diwali Padwa / Tulsi Vivah",
            (2026, 52): "Christmas / Year End New Year Peak",

            # 2027 Projections Window
            (2027, 2): "Makar Sankranti / Pongal", (2027, 4): "Republic Day week",
            (2027, 7): "Valentine's Week", (2027, 12): "Holi / Gudi Padwa",
            (2027, 19): "Akshaya Tritiya", (2027, 34): "Raksha Bandhan",
            (2027, 35): "Janmashtami", (2027, 37): "Ganesh Chaturthi",
            (2027, 41): "Navratri", (2027, 42): "Dussehra",
            (2027, 44): "Dhanteras / Diwali", (2027, 45): "Diwali Padwa / Tulsi Vivah",
            (2027, 52): "Christmas / Year End New Year Peak",

            # 2028 Projections Window
            (2028, 2): "Makar Sankranti / Pongal", (2028, 4): "Republic Day week",
            (2028, 7): "Valentine's Week", (2028, 11): "Holi / Gudi Padwa",
            (2028, 17): "Akshaya Tritiya", (2028, 35): "Raksha Bandhan",
            (2028, 36): "Janmashtami", (2028, 37): "Ganesh Chaturthi",
            (2028, 42): "Navratri", (2028, 43): "Dussehra",
            (2028, 45): "Dhanteras / Diwali", (2028, 46): "Diwali Padwa / Tulsi Vivah",
            (2028, 52): "Christmas / Year End New Year Peak"
        }

    def initialize_and_compile(self):
        if self.model is None:
            import torch
            
            # Dynamically determine the host PC's best available hardware acceleration
            if torch.cuda.is_available():
                device_backend = "gpu"
                torch.set_float32_matmul_precision("high")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # For Apple Silicon Mac architectures (M1/M2/M3), route safely via CPU boundaries
                device_backend = "cpu"
            else:
                device_backend = "cpu"
                
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=1024, max_horizon=512, normalize_inputs=True,
                    use_continuous_quantile_head=True, return_backcast=True
                )
            )
        return self.model

    def execute_modular_forecast(self, dataframe, date_col, target_col, selected_features, horizon):
        self.initialize_and_compile()
        
        historical_df = dataframe.dropna(subset=[target_col]).copy()
        historical_sales = historical_df[target_col].values.astype(np.float32)
        history_length = len(historical_sales)
        required_total_length = history_length + horizon 
        
        # Clean timestamps and explicitly resolve calendar weeks
        historical_df[date_col] = pd.to_datetime(historical_df[date_col])
        dataframe[date_col] = pd.to_datetime(dataframe[date_col])
        
        last_date = historical_df[date_col].iloc[-1]
        try:
            inferred_freq = pd.infer_freq(historical_df[date_col].tail(10))
            if not inferred_freq: inferred_freq = 'W-MON'
        except:
            inferred_freq = 'W-MON'
            
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=inferred_freq)[1:]
        full_timeline = pd.concat([historical_df[date_col], pd.Series(future_dates)]).reset_index(drop=True)
        
        dynamic_numerical = {}
        dynamic_categorical = {}
        
        unique_festivals = [
            "Normal Week", "Makar Sankranti / Pongal", "Republic Day week", "Valentine's Week", 
            "Holi / Gudi Padwa", "Akshaya Tritiya", "Raksha Bandhan", "Janmashtami", 
            "Ganesh Chaturthi", "Navratri", "Dussehra", "Dhanteras / Diwali", 
            "Diwali Padwa / Tulsi Vivah", "Christmas / Year End New Year Peak"
        ]
        fest_to_code = {name: idx for idx, name in enumerate(unique_festivals)}
        
        for feature in selected_features:
            if "festival" in feature.lower() or "event" in feature.lower():
                final_codes = []
                for current_timestamp in full_timeline:
                    # Get exact ISO year and calendar week number
                    year, week, _ = current_timestamp.isocalendar()
                    fest_name = self.weekly_festival_calendar.get((year, week), "Normal Week")
                    final_codes.append(fest_to_code.get(fest_name, 0))
                
                final_array = np.array(final_codes, dtype=np.int32)[:required_total_length]
                is_categorical = True
            else:
                feature_series = dataframe[feature]
                if not pd.api.types.is_numeric_dtype(feature_series):
                    coded_series, _ = pd.factorize(feature_series.astype(str).replace(["None", "none", "nan", "NaN"], "Normal"))
                    feature_values = np.array(coded_series, dtype=np.int32)
                    is_categorical = True
                else:
                    is_categorical = feature_series.nunique() <= 2
                    if is_categorical:
                        feature_values = feature_series.values.astype(np.int32)
                    else:
                        feature_values = feature_series.values.astype(np.float32)
                
                if len(feature_values) < required_total_length:
                    missing_count = required_total_length - len(feature_values)
                    pad_val = 0 if is_categorical else feature_values[-1]
                    extended_part = np.full(missing_count, pad_val, dtype=feature_values.dtype)
                    final_array = np.concatenate([feature_values, extended_part])
                else:
                    final_array = feature_values[:required_total_length]
            
            if is_categorical:
                dynamic_categorical[feature] = [final_array.astype(int).tolist()]
            else:
                dynamic_numerical[feature] = [final_array.astype(float).tolist()]
                
        point_outputs, quantile_outputs = self.model.forecast_with_covariates(
            inputs=[historical_sales.tolist()],
            dynamic_numerical_covariates=dynamic_numerical if dynamic_numerical else None,
            dynamic_categorical_covariates=dynamic_categorical if dynamic_categorical else None,
            xreg_mode="xreg + timesfm"
        )
        
        return historical_df, future_dates, point_outputs[0], quantile_outputs[0]