Formula 1 Race Strategy Decision Support System – Data Engineering Pipeline
Overview

This folder contains the complete data engineering pipeline used in the development of the Formula 1 Race Strategy Decision Support System. The pipeline is responsible for extracting raw Formula 1 race data using the FastF1 API, cleaning and validating the collected data, integrating multiple race information sources, engineering modelling features, and producing the final master dataset used throughout the project.

The objective of this pipeline is to transform raw race telemetry and timing information into a structured lap level dataset suitable for machine learning, probabilistic modelling, simulation, and strategy optimisation.

The resulting dataset serves as the foundation for all downstream models including lap time prediction, tyre degradation estimation, Safety Car probability prediction, pit stop modelling, traffic analysis, rejoin modelling, and race strategy simulation.

Data Source

All primary race data is collected using the FastF1 API.

FastF1 provides access to official Formula 1 timing and session data, including:

Race sessions
Lap timing information
Driver information
Team information
Tyre compounds
Tyre stints
Pit stop information
Track status data
Race control messages
Weather information
Session metadata

The project uses Formula 1 race weekends from the 2019–2025 seasons.

Purpose of the Pipeline

Raw Formula 1 data is distributed across multiple sources and contains inconsistencies, missing values, and differing granularities. The purpose of this pipeline is to:

Collect historical Formula 1 race data.
Standardise race information across seasons.
Clean and validate raw data.
Process tyre and pit stop information.
Convert race control messages into machine learning labels.
Integrate weather and race event information.
Create race level and lap level features.
Build a unified master dataset.
Provide reproducible inputs for all project models.
Pipeline Architecture
FastF1 API
     ↓
Raw Data Extraction
     ↓
Data Cleaning
     ↓
Session Standardisation
     ↓
Pit Stop Processing
     ↓
Tyre Stint Processing
     ↓
Weather Processing
     ↓
Race Event Processing
     ↓
Feature Engineering
     ↓
Dataset Integration
     ↓
Master Lap-Level Dataset
     ↓
Machine Learning Models
Data Extraction Process

The extraction phase collects race information season by season and race by race.

For every race session, the pipeline retrieves:

Session Metadata
Season
Round
Race name
Circuit
Session type
Event date
Driver Information
Driver ID
Driver name
Team
Team ID
Lap Timing Information
Lap number
Lap time
Sector 1 time
Sector 2 time
Sector 3 time
Position
Track status
Tyre Information
Compound
Tyre age
Stint number
Stint length
Pit Stop Information
Pit in lap
Pit out lap
Number of stops
Compound changes
Weather Information
Air temperature
Track temperature
Rain indicators
Humidity
Wind speed
Race Control Messages
Safety Car
Virtual Safety Car
Red Flag
Yellow Flag
Session interruptions
Data Cleaning Process

The raw FastF1 data contains several issues that must be addressed before modelling.

The cleaning stage includes:

Missing Value Handling

Managing:

Missing weather records
Missing tyre information
Missing timing information
Missing stint information

Appropriate imputation and validation rules are applied where required.

Duplicate Removal

Duplicate observations are identified and removed using:

race_id
driver_id
lap_number

as the unique race-lap identifier.

Standardisation

Standardisation is applied to:

Driver names
Team names
Circuit names
Tyre compound labels
Session identifiers

to ensure consistency across seasons.

Data Type Validation

All variables are converted into appropriate formats:

Numerical:

Lap times
Temperatures
Gaps
Tyre age

Categorical:

Compound
Team
Driver
Circuit

Datetime:

Session dates
Event timestamps
Pit Stop and Tyre Processing

Pit stop information is transformed into strategy related variables.

Generated variables include:

Pit stop count
Current stint number
Stint length
Compound changes
Pit stop lap
Tyre age
Tyre age after stop

This information is critical for strategy modelling and degradation analysis.

Race Event Processing

Race control messages are processed and converted into lap level event indicators.

Examples include:

Safety Car flags
Virtual Safety Car flags
Red Flag indicators
Yellow Flag indicators

These event markers are later used for hazard modelling and race interruption prediction.

Weather Processing

Weather information is aligned with race timing data and merged at the lap level.

Generated weather variables include:

Air temperature
Track temperature
Rain indicators
Humidity
Wind speed

Weather features play an important role in tyre behaviour and race strategy.

Dataset Integration

After cleaning and processing, all individual data sources are merged into a unified dataset.

The master dataset is indexed by:

race_id
driver_id
lap_number

This means each row represents:

One driver, in one race, on one specific lap.

This structure reflects the real decision making unit used by Formula 1 strategy engineers.

Feature Engineering

A large number of race strategy features are generated from the cleaned data.

Examples include:

Pace Features
Previous lap time
Rolling pace
Pace trends
Tyre Features
Tyre age
Stint length
Compound type
Degradation measures
Race Context Features
Position
Gap ahead
Gap behind
Laps remaining
Race phase
Traffic Features
Cars ahead count
Cars behind count
Traffic density
Clean air indicators
Event Features
Safety Car indicators
Virtual Safety Car indicators
Incident history
Weather Features
Air temperature
Track temperature
Rain conditions

These engineered features provide the inputs required for downstream modelling.

Final Dataset

The output of the pipeline is a consolidated lap level master dataset.

Typical columns include:

race_id
season
round
circuit_id
driver_id
team_id
lap_number
lap_time
position
tyre_compound
tyre_age
stint_number
pit_stop_number
gap_ahead
gap_behind
laps_remaining
race_phase
air_temperature
track_temperature
rain_flag
sc_flag
vsc_flag
red_flag

The final dataset forms the central modelling table used throughout the project.

Data Leakage Prevention

A major design objective of the pipeline is preventing future information from entering model inputs.

The pipeline ensures:

Features are derived only from current and historical race information.
Future lap times are not used as predictors.
Future pit stops are not used as current features.
Future incidents are not used as current predictors.
Final race results are not used during feature creation.
Temporal ordering is preserved throughout processing.

This allows the machine learning models to operate under realistic race conditions.