import os as os
from lib import ReadCsv
from lib import ReadConfig
from lib import ReadData
from lib import NetworkModel
from lib import ModelMetrics
import pandas as pd
import matplotlib.pyplot as plt
import pywt

import seaborn as sns

import keras

from pandas.plotting import scatter_matrix

config = ReadConfig.ReadConfig()
config_data = config.read_config(os.path.join("config", "config.json"))


reader = ReadData.ReadData()
all_data = reader.readClimateFiles(config_data)


plt.scatter(all_data["local_date"], all_data["Hs"])
plt.show()

##sns.pairplot(all_data)

subset = all_data[["local_date",
    "Hs",
    "Hmax",
    "Tz",
    "Tp",
    "DirTpTRUE",
    "SST",
    "Evapotranspiration_mm",
    "Rain_mm",
    "PanEvaporation_mm",
    "MaximumTemperature_C",
    "MaxRelativeHumidity_pc",
    "MinRelativeHumidity_pc",
    "Avg10mWindSpeed_m_sec",
    "SolarRadiation_MJ_sqm"]]
C = subset[subset.columns[1:14]].corr()

sns.heatmap(C)
plt.show()

modeller = NetworkModel.NetworkModel()


train, valid, test = modeller.partition_data(subset)

x_cols = ["Evapotranspiration_mm",
          "Rain_mm",
          "PanEvaporation_mm",
          "MaximumTemperature_C",
          "MaxRelativeHumidity_pc",
          "MinRelativeHumidity_pc",
          "Avg10mWindSpeed_m_sec",
          "SolarRadiation_MJ_sqm"]
y_cols = ["Hs",
          "Hmax",
          "Tz",
          "Tp",
          "DirTpTRUE",
          "SST"]

train_x = train[x_cols]
train_y = train[y_cols]
valid_x = valid[x_cols]
valid_y = valid[y_cols]
test_x = test[x_cols]
test_y = test[y_cols]

num_inputs = train_x.shape[1]
num_outputs = train_y.shape[1]

model = modeller.model_dense(num_inputs, num_outputs)

modeller.compile_model(model, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

modeller.fit_model(
    model,
    train_x.values,
    train_y.values,
    valid_x.values,
    valid_y.values,
    num_epochs=100,
    batch_size=32)

loss, accuracy, mae = model.evaluate(x=test_x.values,
                                     y=test_y.values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))


y_sim = model.predict(x=test_x.values)
y_sim.shape


metrics = ModelMetrics.ModelMetrics()




test_dates = test["local_date"]


pairs = list(zip(test.columns[0:7], range(0,7)))


all_metrics = []
for pair in pairs[1:len(pairs)]:
    name = pair[0]
    idx = pair[1] - 1
    metricData = {
        'Model': 'SimpleDense',
        'Property':name,
        'R2': metrics.r_squared(test[name], y_sim[:,idx]),
        'agreement_d': metrics.agreement(test[name], y_sim[:,idx]),
        'efficiency_E': metrics.efficiency(test[name], y_sim[:,idx]),
        'percentPeakDeviation':metrics.percent_peak_deviation(test[name],y_sim[:,idx]),
        'RMSE':metrics.root_mean_square_error(test[name],y_sim[:,idx]),
        'MAE':metrics.mean_absolute_error(test[name],y_sim[:,idx])
    }
    all_metrics.append(metricData)
all_metrics = pd.DataFrame.from_dict(all_metrics)
print(all_metrics)

for pair in pairs[1:len(pairs)]:
    name = pair[0]
    idx = pair[1] - 1
    plt.scatter(test["local_date"].values, test[name].values, label="Observed "+name)
    plt.scatter(test["local_date"].values, y_sim[:,idx], color="red", label="Simulated "+name)
    plt.legend()
    plt.show()

for pair in pairs[1:len(pairs)]:
    name = pair[0]
    idx = pair[1] - 1
    plt.hist(test[name].values, label="Observed "+name, lw=1, alpha=0.6, edgecolor='black')
    plt.hist(y_sim[:,idx], color="red", label="Simulated "+name, lw=1, alpha=0.6, edgecolor='black')
    plt.legend()
    plt.show()


all_data2 = reader.readClimateFiles(config_data, add_wavelets=True, wavelet_cols=y_cols)

all_data2.dtypes

## we construct a new model using predictors including the wavelet coefficients.

subset2 = all_data2[["local_date",
                   "Hs",
                   "Hmax",
                   "Tz",
                   "Tp",
                   "DirTpTRUE",
                   "SST",
                   "Evapotranspiration_mm",
                   "Rain_mm",
                   "PanEvaporation_mm",
                   "MaximumTemperature_C",
                   "MaxRelativeHumidity_pc",
                   "MinRelativeHumidity_pc",
                   "Avg10mWindSpeed_m_sec",
                   "SolarRadiation_MJ_sqm",
                    "Hs_C1",
                    "Hs_C2",
                    "Hs_C3",
                    "Hmax_C1",
                    "Hmax_C2",
                    "Hmax_C3",
                    "Tz_C1",
                    "Tz_C2",
                    "Tz_C3",
                    "Tp_C1",
                    "Tp_C2",
                    "Tp_C3",
                    "DirTpTRUE_C1",
                    "DirTpTRUE_C2",
                    "DirTpTRUE_C3",
                    "SST_C1",
                    "SST_C2",
                    "SST_C3"]]

train2, valid2, test2 = modeller.partition_data(subset2)

x_cols = ["Evapotranspiration_mm",
          "Rain_mm",
          "PanEvaporation_mm",
          "MaximumTemperature_C",
          "MaxRelativeHumidity_pc",
          "MinRelativeHumidity_pc",
          "Avg10mWindSpeed_m_sec",
          "SolarRadiation_MJ_sqm",
          "Hs_C1",
          "Hs_C2",
          "Hs_C3",
          "Hmax_C1",
          "Hmax_C2",
          "Hmax_C3",
          "Tz_C1",
          "Tz_C2",
          "Tz_C3",
          "Tp_C1",
          "Tp_C2",
          "Tp_C3",
          "DirTpTRUE_C1",
          "DirTpTRUE_C2",
          "DirTpTRUE_C3",
          "SST_C1",
          "SST_C2",
          "SST_C3"]
y_cols = ["Hs",
          "Hmax",
          "Tz",
          "Tp",
          "DirTpTRUE",
          "SST"]

train_x = train2[x_cols]
train_y = train2[y_cols]
valid_x = valid2[x_cols]
valid_y = valid2[y_cols]
test_x = test2[x_cols]
test_y = test2[y_cols]

num_inputs = train_x.shape[1]
num_outputs = train_y.shape[1]

model2 = modeller.model_dense(num_inputs, num_outputs)

modeller.compile_model(model2, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])


modeller.fit_model(
    model2,
    train_x.values,
    train_y.values,
    valid_x.values,
    valid_y.values,
    num_epochs=100,
    batch_size=32)

loss, accuracy, mae = model2.evaluate(x=test_x.values,
                                     y=test_y.values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))

y_sim2 = model2.predict(x=test_x.values)


test_dates = test2["local_date"]


pairs = list(zip(test2.columns[0:7], range(0,7)))


all_metrics2 = []
for pair in pairs[1:len(pairs)]:
    name = pair[0]
    idx = pair[1] - 1
    metricData = {
        'Model': 'WaveletDense',
        'Property':name,
        'R2': metrics.r_squared(test2[name], y_sim2[:,idx]),
        'agreement_d': metrics.agreement(test2[name], y_sim2[:,idx]),
        'efficiency_E': metrics.efficiency(test2[name], y_sim2[:,idx]),
        'percentPeakDeviation':metrics.percent_peak_deviation(test2[name],y_sim2[:,idx]),
        'RMSE':metrics.root_mean_square_error(test2[name],y_sim2[:,idx]),
        'MAE':metrics.mean_absolute_error(test2[name],y_sim2[:,idx])
    }
    all_metrics2.append(metricData)
all_metrics2 = pd.DataFrame.from_dict(all_metrics2)
print(all_metrics2)

for pair in pairs[1:len(pairs)]:
    name = pair[0]
    idx = pair[1] - 1
    plt.scatter(test2["local_date"].values, test2[name].values, label="Observed "+name)
    plt.scatter(test2["local_date"].values, y_sim2[:,idx], color="red", label="Simulated "+name)
    plt.legend()
    plt.show()

for pair in pairs[1:len(pairs)]:
    name = pair[0]
    idx = pair[1] - 1
    plt.hist(test2[name].values, label="Observed "+name, lw=1, alpha=0.6, edgecolor='black')
    plt.hist(y_sim2[:,idx], color="red", label="Simulated "+name, lw=1, alpha=0.6, edgecolor='black')
    plt.legend()
    plt.show()


pair = pairs[1]
name = pair[0]
idx = pair[1] - 1
plt.hist(test2[name].values, label="Observed "+name)
plt.hist(y_sim2[:,idx], color="red", label="Simulated "+name)
plt.legend()
plt.show()