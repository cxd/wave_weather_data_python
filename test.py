import os as os
from lib import ReadCsv
from lib import ReadConfig
from lib import ReadData
from lib import NetworkModel
from lib import ModelMetrics
from lib import SeriesPlot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras

from datetime import datetime



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

epochs = 1000

logdir="logs/scalars/model1" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model = modeller.model_dense(num_inputs, num_outputs)

modeller.compile_model(model, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

# model1 overfits at around 150 epochs
modeller.fit_model(
    model,
    train_x.values,
    train_y.values,
    valid_x.values,
    valid_y.values,
    num_epochs=145,
    batch_size=32,
    callback_list=[tensorboard_callback])

loss, accuracy, mae = model.evaluate(x=test_x.values,
                                     y=test_y.values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))


y_sim = model.predict(x=test_x.values)
y_sim.shape


metrics = ModelMetrics.ModelMetrics()




test_dates = test["local_date"]


pairs = list(zip(test.columns[1:7], range(0, 6)))
all_metrics = metrics.report_all_metrics('SimpleDense', pairs, test, y_sim)

series_plot = SeriesPlot.SeriesPlot()

series_plot.plot_series(pairs, test['local_date'].values, test, y_sim)

series_plot.plot_histograms(pairs, test, y_sim)



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

logdir="logs/scalars/model2" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model2 = modeller.model_dense(num_inputs, num_outputs)

modeller.compile_model(model2, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

## model2 overfits at > 90 epochs
modeller.fit_model(
    model2,
    train_x.values,
    train_y.values,
    valid_x.values,
    valid_y.values,
    num_epochs=90,
    batch_size=32,
    callback_list=[tensorboard_callback])

loss, accuracy, mae = model2.evaluate(x=test_x.values,
                                      y=test_y.values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))

y_sim2 = model2.predict(x=test_x.values)




pairs = list(zip(test2.columns[1:7], range(0,6)))

all_metrics2 = metrics.report_all_metrics('WaveletDense', pairs, test2, y_sim2)

series_plot = SeriesPlot.SeriesPlot()

series_plot.plot_series(pairs, test2['local_date'].values, test2, y_sim2)

series_plot.plot_histograms(pairs, test2, y_sim2)

print(all_metrics2)

## Investivate modelling using min/max standardisation.
features = ["Evapotranspiration_mm",
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
            "SST_C3",
            "Hs",
            "Hmax",
            "Tz",
            "Tp",
            "DirTpTRUE",
            "SST"]

## The min and max need to be kept for the ability to rescale after training.
data_min = all_data2[features].min()
data_max = all_data2[features].max()
data_delta = data_max[features] - data_min[features]

data_scaled = (all_data2[features] - data_min)/data_delta
data_scaled['local_date'] = all_data2['local_date']


## We now train on the scaled data.

train3, valid3, test3 = modeller.partition_data(data_scaled)


train_x = train3[x_cols]
train_y = train3[y_cols]
valid_x = valid3[x_cols]
valid_y = valid3[y_cols]
test_x = test3[x_cols]
test_y = test3[y_cols]

num_inputs = train_x.shape[1]
num_outputs = train_y.shape[1]

logdir="logs/scalars/model3" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model3 = modeller.model_dense(num_inputs, num_outputs)

modeller.compile_model(model3, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

## Model 3 overfits somewhere near 60 epochs
modeller.fit_model(
    model3,
    train_x.values,
    train_y.values,
    valid_x.values,
    valid_y.values,
    num_epochs=60,
    batch_size=32,
    callback_list=[tensorboard_callback])

loss, accuracy, mae = model3.evaluate(x=test_x.values,
                                      y=test_y.values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))



y_sim3 = model3.predict(x=test_x.values)


y_sim3_scaled = pd.DataFrame(y_sim3, columns=y_cols) * data_delta[y_cols] + data_min[y_cols]
obs3_scaled = test_y * data_delta[y_cols] + data_min[y_cols]

test_dates = test3["local_date"]


pairs = list(zip(['Hs', 'Hmax', 'Tz', 'Tp', 'DirTpTRUE', 'SST'], range(0,6)))


all_metrics3 = metrics.report_all_metrics('ScaledWaveletDense', pairs, obs3_scaled, y_sim3_scaled.values)

series_plot = SeriesPlot.SeriesPlot()

series_plot.plot_series(pairs, test3['local_date'].values, obs3_scaled, y_sim3_scaled.values)

series_plot.plot_histograms(pairs, obs3_scaled, y_sim3_scaled.values)


print(all_metrics3)



# a fourth model using sigmoid activation
logdir="logs/scalars/model4" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model4 = modeller.model_dense(num_inputs, num_outputs, hidden_activation='sigmoid')

modeller.compile_model(model4, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

## model4 overfits somewhere above 190 epochs
modeller.fit_model(
    model4,
    train_x.values,
    train_y.values,
    valid_x.values,
    valid_y.values,
    num_epochs=190,
    batch_size=32,
    callback_list=[tensorboard_callback])

loss, accuracy, mae = model4.evaluate(x=test_x.values,
                                      y=test_y.values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))


y_sim4 = model4.predict(x=test_x.values)

y_sim4_scaled = pd.DataFrame(y_sim4, columns=y_cols) * data_delta[y_cols] + data_min[y_cols]

all_metrics4 = metrics.report_all_metrics('ScaledWaveletDenseSigmoid', pairs, obs3_scaled, y_sim4_scaled.values)

series_plot = SeriesPlot.SeriesPlot()

series_plot.plot_series(pairs, test3['local_date'].values, obs3_scaled, y_sim4_scaled.values)

series_plot.plot_histograms(pairs, obs3_scaled, y_sim4_scaled.values)


print(all_metrics4)



## fifth model using tanh activation


logdir="logs/scalars/model5" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model5 = modeller.model_dense(num_inputs, num_outputs, hidden_activation='tanh')

modeller.compile_model(model5, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

## Model 5 seems to continue to decrease for 1000 epochs
## have not tested longer durations.
modeller.fit_model(
    model5,
    train_x.values,
    train_y.values,
    valid_x.values,
    valid_y.values,
    num_epochs=epochs,
    batch_size=32,
    callback_list=[tensorboard_callback])

loss, accuracy, mae = model5.evaluate(x=test_x.values,
                                      y=test_y.values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))


y_sim5 = model5.predict(x=test_x.values)

y_sim5_scaled = pd.DataFrame(y_sim5, columns=y_cols) * data_delta[y_cols] + data_min[y_cols]

all_metrics5 = metrics.report_all_metrics('ScaledWaveletDenseTanh', pairs, obs3_scaled, y_sim5_scaled.values)

series_plot = SeriesPlot.SeriesPlot()

series_plot.plot_series(pairs, test3['local_date'].values, obs3_scaled, y_sim5_scaled.values)

series_plot.plot_histograms(pairs, obs3_scaled, y_sim5_scaled.values)


print(all_metrics5)


metric_data = pd.concat([all_metrics, all_metrics2, all_metrics3, all_metrics4, all_metrics5])

metrics = ['R2','agreement_d', 'efficiency_E', 'RMSE','MAE','percentPeakDeviation']
for metric in metrics:
    bars = sns.barplot(x='Model', y=metric, data=metric_data, hue='Property')
    for item in bars.get_xticklabels():
        item.set_rotation(45)
    plt.show()


import numpy as np

a, b = np.corrcoef(y_sim3_scaled["SST"].values, obs3_scaled['SST'].values)
np.corrcoef(y_sim3_scaled["Hs"].values, obs3_scaled['Hs'].values)
np.corrcoef(y_sim3_scaled["Hmax"].values, obs3_scaled['Hmax'].values)
