import pandas as pd
import numpy as np

class ModelMetrics:

    def __init__(self):
        None

    def r_squared(self, obs, sim):
        # Calculate the R^2 measure for observation and simulated data.
        obs1 = np.array(obs)
        sim1 = np.array(sim)
        muO = obs1.mean()
        muS = sim1.mean()
        deltaO = obs1 - muO
        deltaS = sim1 - muS
        ssO = deltaO.transpose().dot(deltaO)
        ssS = deltaS.transpose().dot(deltaS)
        prod = deltaO.transpose().dot(deltaS)
        r = prod/(np.sqrt(ssO*ssS))
        R2 = r*r
        return R2

    def agreement(self, obs, sim):
        # Compute wilmotts index of agreement.
        obs1 = np.array(obs)
        sim1 = np.array(sim)
        muO = obs1.mean()
        delta = obs1 - sim
        MSE = delta.transpose().dot(delta)
        deltaS = np.abs(sim1 - muO)
        deltaO = np.abs(obs1 - muO)
        total = deltaS + deltaO
        PE = total.transpose().dot(total)
        n = len(obs1)
        d = 1.0 - (MSE/PE)
        return d

    def efficiency(self, obs, sim):
        # Compute Nash Sutcliffe efficiency metric
        obs1 = np.array(obs)
        sim1 = np.array(sim)
        muO = obs1.mean()
        delta = obs1 - sim1
        MSE = delta.transpose().dot(delta)
        deltaO = obs1 - muO
        varO = deltaO.transpose().dot(deltaO)
        p = MSE/varO
        E = 1.0 - p
        return E

    def percent_peak_deviation(self, obs, sim):
        # Compute the percent peak deviation measure.
        obs1 = np.array(obs)
        sim1 = np.array(sim)
        maxObs = obs1.max()
        maxSim = sim1.max()
        pdv = 100.0 * (maxSim - maxObs)/maxObs
        return pdv

    def root_mean_square_error(self, obs, sim):
        # Compute the root mean square error
        obs1 = np.array(obs)
        sim1 = np.array(sim)
        n = len(obs1)
        delta = sim1 - obs1
        ss = delta.transpose().dot(delta)
        rmse = np.sqrt(1.0/n * ss)
        return rmse

    def mean_absolute_error(self, obs, sim):
        # Compute the mean absolute error.
        obs1 = np.array(obs)
        sim1 = np.array(sim)
        n = len(obs1)
        delta = np.abs(sim1 - obs1)
        mae = 1.0/n * np.sum(delta)
        return mae

    def report_all_metrics(self, model_name, label_index_pairs, test_data, sim_data):
        # Report all metrics and return a pandas dataframe.
        # The label_index_pairs represent the labelname of the test data and the index of the sim data.
        # Each index must correspond with the same position in the test data as in the simulated data.
        all_metrics = []
        for pair in label_index_pairs[0:len(label_index_pairs)]:
            name = pair[0]
            idx = pair[1]
            metricData = {
                'Model': model_name,
                'Property':name,
                'R2': self.r_squared(test_data[name], sim_data[:,idx]),
                'agreement_d': self.agreement(test_data[name], sim_data[:,idx]),
                'efficiency_E': self.efficiency(test_data[name], sim_data[:,idx]),
                'percentPeakDeviation':self.percent_peak_deviation(test_data[name], sim_data[:,idx]),
                'RMSE':self.root_mean_square_error(test_data[name], sim_data[:,idx]),
                'MAE':self.mean_absolute_error(test_data[name], sim_data[:,idx])
            }
            all_metrics.append(metricData)
        return pd.DataFrame.from_dict(all_metrics)

