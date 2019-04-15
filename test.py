import os as os
from lib import ReadCsv
from lib import ReadConfig
from lib import ReadData

from lib import NetworkModel

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import keras

from pandas.plotting import scatter_matrix

config = ReadConfig.ReadConfig()
config_data = config.read_config(os.path.join("config", "config.json"))


reader = ReadData.ReadData()
all_data = reader.readClimateFiles(config_data)


plt.scatter(all_data["local_date"], all_data["Hs"])
plt.show()

sns.pairplot(all_data)

subset = all_data[[
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
C = subset.corr()

sns.heatmap(C)

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

model.evaluate(x=test_x.values,
               y=test_y.values)



