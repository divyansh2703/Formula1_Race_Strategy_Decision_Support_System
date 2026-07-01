# 2026-mcm-SportsAnalytics
# Formula 1 Race Strategy Decision Support System

This project is a machine learning based Formula 1 race strategy decision support system. The aim of the project is to analyse historical Formula 1 race data and support race strategy decisions such as whether a driver should stay out, pit now, wait a few laps, or change to another tyre compound. The system is built as a complete pipeline from data collection to final risk-aware strategy recommendation.

The data was collected using the FastF1 API and processed at lap level. Each row in the dataset represents one driver at one lap of a race. The collected data includes lap times, tyre compound, tyre age, stint details, pit stop information, driver position, gaps to other cars, weather data, safety car and virtual safety car information, race status, circuit information and driver/team details. After extraction, the data was cleaned, standardised and merged into a single master dataset.

The project is divided into six main models. Model 1 predicts lap time, expected pace, tyre degradation and uncertainty. Model 2 predicts future race interruptions such as Safety Car, Virtual Safety Car and Red Flag probabilities. Model 3 predicts pit stop probability and tyre compound choice. Model 4 models on-track interaction such as clean air, position changes, traffic and rejoin penalty. Model 5A estimates local pit and wait strategy costs, while Model 5B uses all previous model outputs to simulate different race strategy scenarios. Finally, Model 6 acts as the risk-aware recommendation layer and gives the final strategy decision.

The final master file used by the strategy simulator is:

`outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv`

This file contains the original cleaned race data along with the generated outputs from Models 1 to 4. Model 5A also produces a strategy table:

`outputs/model5a_local_strategy_FINAL_strategy_table_20260609_101034.csv`

Model 5B uses these files to simulate possible race outcomes for different strategies. It compares options such as staying out, pitting now, waiting one lap, waiting three laps, waiting five laps and switching to Soft, Medium, Hard, Intermediate or Wet tyres. It also supports different scenarios such as base race conditions, Safety Car situations, no-neutralisation cases and rain/wet-track conditions.

Model 6 is the final pipeline runner. When Model 6 is executed, it can automatically run Model 5B first, store the Model 5B simulation output, and then generate the final risk-aware recommendation. The recommendation considers expected finishing position, expected points, top 10 probability, top 5 probability, podium probability, win probability, downside risk, compound confidence and the driver’s race situation.

The final Model 6 output is saved in a separate run folder:

`outputs/model6_runs/<race_driver_lap_scenario_risk_timestamp>/`

Each Model 6 run contains two main folders. The `model5b_run` folder stores the raw simulation output from Model 5B, including results, metrics, configuration and report files. The `model6_result` folder stores the final Model 6 recommendation, ranked strategies, JSON output, technical report, simple report and combined report.

The complete pipeline works in the following order:

Data collection using FastF1
Data cleaning and merging
Model 1 pace and degradation prediction
Model 2 race interruption probability prediction
Model 3 pit and compound prediction
Model 4 traffic and rejoin modelling
Model 5A local strategy cost estimation
Model 5B full scenario simulation
Model 6 final risk-aware recommendation

The main scripts used in the project are:

`train_model1_lap_time_degradation.py`
`train_model2_run_final9.py`
`train_model3a_pit_propensity.py`
`train_model3b_compound_choice.py`
`apply_model3c_wet_compound_handler.py`
`train_model4a_clean_air.py`
`train_model4b_position_delta.py`
`train_model4c_rejoin_penalty.py`
`run_model5a_local_strategy.py`
`run_model5b_scenario_simulator.py`
`run_model6.py`

The final pipeline can be run using Model 6. Example command:

```bash
python run_model6.py \
  --model5b_script run_model5b_scenario_simulator.py \
  --input_path outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv \
  --model5a_strategy_table outputs/model5a_local_strategy_FINAL_strategy_table_20260609_101034.csv \
  --outputs_dir outputs \
  --race_id 2019_02_R \
  --driver_id norris \
  --decision_lap 28 \
  --scenario base \
  --strategies auto \
  --n_sim 5000 \
  --seed 42 \
  --risk_appetite balanced
```

The project prevents data leakage by using race-based train, validation and test splits instead of random lap-level splitting. Future race information is not used as an input when making predictions at a current lap. Each target is created using only the correct future horizon, while model features represent information available at the decision point.

This project is designed as an academic machine learning research system for Formula 1 strategy. It combines predictive modelling, probabilistic simulation, baseline evaluation, risk-aware recommendation and real-world scenario validation. The final output helps explain not only which strategy is recommended, but also why it is recommended and how risky the decision is.
