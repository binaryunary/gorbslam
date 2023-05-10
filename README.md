# Introduction
This repo is part of my master's thesis.

TODO: Add link to thesis itself.

`gorbslam` stands for Georeferenced ORB-SLAM.

This repo contains the code to:
- build and train the coordinate transformation model (FCNN)
- validate the model against other regression methods
- plot and visualize the results

Repo layout:
- `gorbslam/` - the `gorbslam` Python module
- `data/` - mapping and localization results for sequences 01, 02, 03, including trained models
- `notbooks/` - various notebooks to interactively explore the data and results

# Running
1. Install the environment
```
conda env create -f environment.yml
```
NOTE: It contains some `pip` installed deps as well, you may need to install them manually.

2. Decompress the data
```
cd data/
unzip sequences.zip
```

Go to `notebooks/plot_results.ipynb` and run it.


# Technical details
- requires Python >= 3.10.11
- all dependencies are listed in `environment.yml`
