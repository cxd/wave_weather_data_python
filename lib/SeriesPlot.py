import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class SeriesPlot:

    def __init__(self):
        None

    def plot_series(self, label_index_pairs, x_series, obs_series, sim_series, obs_color='blue', sim_color='red'):
        # Plot the two series for a given observation and simulation over the supplied x-series.
        # The label_index_pairs represent the labelname of the test data and the index of the sim data.
        # Each index must correspond with the same position in the test data as in the simulated data.
        for pair in label_index_pairs[0:len(label_index_pairs)]:
            name = pair[0]
            idx = pair[1]
            plt.scatter(x_series, obs_series[name].values, label="Observed "+name, color=obs_color)
            plt.scatter(x_series, sim_series[:, idx], color=sim_color, label="Simulated "+name)
            plt.legend()
            plt.show()

    def plot_histograms(self, label_index_pairs, obs_series, sim_series, obs_color='blue', sim_color='red'):
        ## Plot histograms for the two series for a given observation and simulation.
        # The label_index_pairs represent the labelname of the test data and the index of the sim data.
        # Each index must correspond with the same position in the test data as in the simulated data.
        for pair in label_index_pairs[0:len(label_index_pairs)]:
            name = pair[0]
            idx = pair[1]
            plt.hist(obs_series[name].values, label="Observed "+name, lw=1, alpha=0.6, edgecolor='black', color=obs_color)
            plt.hist(sim_series[:, idx], color=sim_color, label="Simulated "+name, lw=1, alpha=0.6, edgecolor='black')
            plt.legend()
            plt.show()

