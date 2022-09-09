# The online companion to the paper "" submitted to the ISGT #

This repository contains 
1. the dataset used in the simulation studies
2. the source code of the experiments

# 1. Dataset #

The data for the experiments is organized in JSON file format as per the input file format of the UnitCommitment.jl package. We use the net load measurements recorded in the California ISO system to construct the net load data for the experiments. As covariates, we use lagged observations of net load. Further, we use temperature measurements recorded in the following counties
1. Los Angeles
2. San Diego
3. Concord
4. San Jose

We harvest global horizontal irradiance measurements collected in
1. Antelope Valley
2. Desert Sunlight
3. Mount Signal
4. Solar Star

We use wind speed measurements recorded in
1. Alta
2. San Gorgonio
3. Shiloh

The data for the generators is taken from the IEEE 14-bus test system,        

# 2. Dataset #
