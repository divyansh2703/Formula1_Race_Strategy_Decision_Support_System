# Model 1 README: Lap Time, Pace, Tyre Degradation and Uncertainty

## 1. Purpose

Model 1 is the foundation of the Formula 1 race strategy pipeline. It predicts expected lap time, fresh-tyre pace, tyre degradation, and lap-time uncertainty. These outputs are required because every strategy decision depends on how fast the driver is expected to be on the current tyre and how much performance could be gained from switching to a fresh tyre.

Model 1 answers this question:

> How fast is the car expected to be now, how much are the tyres degrading, and how uncertain is the pace prediction?

---

## 2. Main Script

Recommended script name:

```text
train_model1_lap_time_degradation.py
```

If your local script has a different name, update this README to match it.

---

## 3. Inputs

Model 1 uses cleaned lap-level Formula 1 race data. Important input features include:

```text
race_id
driver_id
team_id
circuit_id
lap_number
laps_remaining
tyre_compound
tyre_age
stint_number
previous_lap_time
track_temperature
air_temperature
rain_flag
race_phase
safety car / virtual safety car indicators
```

The model uses only information that would be available at or before the current lap. Future race outcome information is not used as input.

---

## 4. Outputs

Model 1 produces these main outputs:

```text
expected_lap_time
expected_lap_time_fresh
tyre_degradation_delta
tyre_degradation_per_lap
lap_time_sigma
```

`expected_lap_time` is the predicted lap time for the driver in the current race state.

`expected_lap_time_fresh` is the expected lap time if the tyre was fresh. This helps estimate the benefit of pitting for new tyres.

`tyre_degradation_delta` measures how much slower the current tyre is compared with a fresh tyre estimate.

`tyre_degradation_per_lap` measures how much performance is being lost per lap as the tyre ages.

`lap_time_sigma` represents uncertainty in lap-time prediction and is used by the simulator to generate realistic variation.

---

## 5. Machine Learning Method

Model 1 uses regression methods. The final model is based mainly on XGBoost regression because lap time and tyre degradation are non-linear. Tyre degradation depends on compound, tyre age, track temperature, circuit, driver, team, stint length, and race phase. XGBoost can learn these interactions better than a simple linear model.

A Ridge regression model is used as a baseline. Ridge is useful because it provides a simple and stable comparison. If the XGBoost model performs better than Ridge, it shows that the non-linear model is adding value.

---

## 6. Evaluation Metrics

Model 1 is evaluated using regression metrics:

```text
MAE
RMSE
R²
```

MAE measures the average lap-time error in seconds. RMSE penalises large errors more strongly. R² measures how much variation in lap time is explained by the model.

---

## 7. Connection to Other Models

Model 1 feeds directly into Models 5A and 5B through the final master file:

```text
outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv
```

Model 5B uses Model 1 outputs to simulate race pace, tyre degradation, fresh tyre advantage, and uncertainty.

Model 6 indirectly uses Model 1 through Model 5B results.

---

## 8. Why Model 1 Matters

Without Model 1, the system would not know whether staying out or pitting is faster. Model 1 provides the pace engine of the entire pipeline. Strategy decisions such as pit now, wait three laps, or stay out depend heavily on the expected pace and degradation values produced by this model.
