# Formula 1 Race Strategy Decision Support with Machine Learning and Monte Carlo Simulation

A collaborative MSc research project that connects predictive models with race strategy decisions. The system estimates pace, disruption risk, pit behaviour and traffic effects, then compares candidate strategies through Monte Carlo simulation and a final recommendation layer. The research paper is titled “An Integrated Risk Aware Decision Support System for Formula 1 Race Strategy Using Machine Learning and Monte Carlo Simulation.”

## The question

At a given race lap, should a driver stay out, pit now or delay a stop, and how does the recommendation change with uncertainty, track position, traffic and weather?

## Tools and methods

Python, FastF1, pandas, scikit learn, XGBoost, Optuna, Monte Carlo simulation, Decision support.

## Work in this repository

1. Collected and integrated historical timing, tyre, weather, event and race context data from the 2019 to 2025 seasons.
2. Developed six connected model stages: pace and degradation; disruption probabilities; pit and compound behaviour; traffic and rejoin effects; strategy simulation; and final recommendation.
3. Used race based splitting in the modelling design so training and evaluation do not randomly mix laps from the same race.
4. Compared stay out, pit and wait strategies under configurable scenarios, including neutralisation and wet conditions.
5. Produced ranked strategies and reports that consider expected points, finishing position, outcome probabilities and downside risk.

## Evidence and scope

| Measure | Recorded value |
| --- | --- |
| Committed master dataset | 154,606 driver lap rows |
| Master dataset columns at this stage | 70 |
| Races | 151 |
| Driver identifiers | 39 |
| Circuit identifiers | 34 |
| Seasons | 2019 to 2025 |
| Paper reported lap time MAE | 5.7465 seconds for the Ridge baseline; 1.0548 seconds for the final model |
| Paper reported rejoin penalty MAE | 8.7578 seconds for the historical baseline; 3.4895 seconds for the final model |

## Repository guide

| File or folder | Purpose |
| --- | --- |
| [build/Race Strategy Decision Support System for Formula 1/Dataset/lap_level_model_table_master.csv](https://github.com/divyansh2703/Formula1_Race_Strategy_Decision_Support_System/blob/main/build/Race%20Strategy%20Decision%20Support%20System%20for%20Formula%201/Dataset/lap_level_model_table_master.csv) | Master dataset audited for row and entity counts |
| [build/Race Strategy Decision Support System for Formula 1/Documentation/Reseach Paper.pdf](https://github.com/divyansh2703/Formula1_Race_Strategy_Decision_Support_System/blob/main/build/Race%20Strategy%20Decision%20Support%20System%20for%20Formula%201/Documentation/Reseach%20Paper.pdf) | Research paper and reported evaluation results |
| [build/Race Strategy Decision Support System for Formula 1/Data Collection and Cleaning/Readme.md](https://github.com/divyansh2703/Formula1_Race_Strategy_Decision_Support_System/blob/main/build/Race%20Strategy%20Decision%20Support%20System%20for%20Formula%201/Data%20Collection%20and%20Cleaning/Readme.md) | Data engineering documentation |
| [build/Race Strategy Decision Support System for Formula 1/Model Building/README_MAIN.md](https://github.com/divyansh2703/Formula1_Race_Strategy_Decision_Support_System/blob/main/build/Race%20Strategy%20Decision%20Support%20System%20for%20Formula%201/Model%20Building/README_MAIN.md) | Model guide |
| [build/Race Strategy Decision Support System for Formula 1/Model Building/Model 1 - Lap Time and Tyre Degradation Prediction/train_model1_run6FINAL.py](https://github.com/divyansh2703/Formula1_Race_Strategy_Decision_Support_System/blob/main/build/Race%20Strategy%20Decision%20Support%20System%20for%20Formula%201/Model%20Building/Model%201%20-%20Lap%20Time%20and%20Tyre%20Degradation%20Prediction/train_model1_run6FINAL.py) | Pace model training |
| [build/Race Strategy Decision Support System for Formula 1/Model Building/Model 5 - Monte Carlo Strategy Simulation/run_model5b_scenario_simulator.py](https://github.com/divyansh2703/Formula1_Race_Strategy_Decision_Support_System/blob/main/build/Race%20Strategy%20Decision%20Support%20System%20for%20Formula%201/Model%20Building/Model%205%20-%20Monte%20Carlo%20Strategy%20Simulation/run_model5b_scenario_simulator.py) | Scenario simulator |
| [build/Race Strategy Decision Support System for Formula 1/Model Building/Model 6 - Risk Aware Strategy Recommendation/run_model6.py](https://github.com/divyansh2703/Formula1_Race_Strategy_Decision_Support_System/blob/main/build/Race%20Strategy%20Decision%20Support%20System%20for%20Formula%201/Model%20Building/Model%206%20-%20Risk%20Aware%20Strategy%20Recommendation/run_model6.py) | Final recommendation runner |

## Getting started

Start with the model guide and research paper. Scripts live in the nested model folders shown below; earlier documentation used several simplified filenames that are not the committed filenames. Review imports and local paths before building a dedicated environment.

The recommendation runner requires the generated combined outputs from Models 1 to 4 and the Model 5A strategy table. These generated inputs are referenced by the code but are not included in the current public tree. The committed master dataset is an earlier stage and cannot be substituted without generating the required model outputs.

The following command demonstrates the verified argument names and repository script paths. Run it only after generating the two `outputs/` inputs:

```bash
python "build/Race Strategy Decision Support System for Formula 1/Model Building/Model 6 - Risk Aware Strategy Recommendation/run_model6.py" \
  --model5b_script "build/Race Strategy Decision Support System for Formula 1/Model Building/Model 5 - Monte Carlo Strategy Simulation/run_model5b_scenario_simulator.py" \
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

This is an example configuration, not a newly executed result.

## Current limitations

1. The error values are reported in the committed research paper. Model training was not repeated for this documentation update.
2. Dataset counts describe the committed 70 column master table, not a later feature table or the rows retained by each individual model.
3. Simulated outcomes depend on assumptions and scenario settings. They are not observed race gains or a claim that an F1 team uses this system.
4. The public repository is an academic research implementation. Live timing integration, a deployed race operations service and a validated real time product are not established by the committed files.

## Next steps

1. Publish the generated simulator input tables or a reproducible generation manifest.
2. Provide a complete dependency file and portable configuration.
3. Validate live ingestion separately from the historical research workflow.

## Authors and reuse

Divyansh Doshi and Amisha Sanjay Kadukar.

Documentation reviewed against the public repository on 7 September 2026. Counts are taken from the named saved artifacts or directly inspected CSVs; this review did not rerun model training or validate a complete deployment. No source code licence was found in the reviewed project tree. Data and third party material may have separate terms.
