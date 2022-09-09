# The online companion to the paper "Predictive Prescription of Unit Commitment Decisions Under Net Load Uncertainty" submitted to the IEEE PES ISGT NA 2023 #

This repository contains 
1. the dataset used in the simulation studies
2. the source code of the experiments

# 1. Dataset #

The data for the experiments is organized in JSON file format as per the input file format of the UnitCommitment.jl package. The data for the generators is taken from the IEEE 14-bus test system. We use the net load measurements recorded in the California ISO system on June 1, 2018--August 31, 2019 to construct the net load data for the experiments. As covariates, we use 24 lagged realizations of net load, as well as the 24 lagged realizations of the daily, weekly, and monthly moving average of net load. We also define categorical variables to indicate whether a day falls on a weekend and on a public holiday and use one-hot encoding for their representation. Further, we assess the population and population density of the counties of California and use temperature measurements recorded in the following counties
1. Los Angeles
2. San Diego
3. Concord
4. San Jose

By taking into account the spatial distribution of solar installations, we harvest global horizontal irradiance measurements collected in
1. Antelope Valley
2. Desert Sunlight
3. Mount Signal
4. Solar Star

To capture the influence of wind speed on wind power generation, we study the spatial distribution of wind installations in California and use wind speed measurements recorded in
1. Alta
2. San Gorgonio
3. Shiloh

The map below depicts the locations of the weather stations from which we utilize temperature, GHI, and wind speed measurements

# 2. Source code #
We provide the source code for training the RF model in and the source code for solving the UC instances in.
