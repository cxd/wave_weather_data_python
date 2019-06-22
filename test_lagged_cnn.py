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
from lib import modwt
import keras
from datetime import date,datetime,time
from datetime import datetime

config = ReadConfig.ReadConfig()
config_data = config.read_config(os.path.join("config", "config.json"))


reader = ReadData.ReadData()
all_data = reader.readClimateFiles(config_data)

subset = all_data[['local_date','site_x', 'Hs','Hmax','Tz','Tp','DirTpTRUE','SST']]
subset.describe()

subset = subset.sort_values(['local_date'])

target_cols = ['Hs', 'Hmax','Tz','Tp','DirTpTRUE','SST']

# generate lagged data
lagged_data = reader.make_lags_per_site(subset[['Hs', 'Hmax','Tz','Tp','DirTpTRUE','SST', 'site_x', 'local_date']], 'site_x')
lagged_data.index = range(0,lagged_data.shape[0])

numeric_cols = ['Hs', 'Hmax','Tz','Tp','DirTpTRUE','SST']
lagged_cols = reader.getlagged_group_columns(numeric_cols, 1, 8)
flat_lagged_cols = reader.getlagged_columns(numeric_cols, 1, 8)

modeller = NetworkModel.NetworkModel()

model = modeller.model_cnn_lagged(lagged_cols, len(target_cols), 3,
                                  hidden_units=len(lagged_cols),
                                  hidden_activation='relu',
                                  output_activation='linear',
                                  cnn_padding='valid',
                                  use_bias=1, pool_size=2, dropout=0.3)

model.summary()

# generate partitions.
sitenames = subset['site_x'].unique()

group_cols = ['dates','site_x']

temp = flat_lagged_cols
numeric_data = lagged_data[temp]
# min-max normalise the numeric input data.
data_min = numeric_data.min()
data_max = numeric_data.max()
data_delta = data_max - data_min

data_scaled = (numeric_data - data_min)/data_delta

data_scaled['dates'] = lagged_data['local_date']
data_scaled['site_x'] = lagged_data['site_x']

train = None
validate = None
test = None
for site in sitenames:
    data = reader.getsite(data_scaled, 'site_x', site)
    train1, validate1, test1 = modeller.partition_data(data)
    if train is None:
        train = train1
        validate = validate1
        test = test1
    else:
        train = pd.concat([train, train1], axis=0)
        validate = pd.concat([validate, validate1], axis=0)
        test = pd.concat([test, test1], axis=0)


epochs = 500

logdir="logs/scalars/model_lag_cnn" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


modeller.compile_model(model, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

def convert(data):
    n = len(data)
    group_size = data.shape[1]
    result = data.reshape(n, group_size, 1)
    return result

# model1 overfits at around 150 epochs
history = modeller.fit_model(
    model,
    [convert(train[lagged_cols[0]].values),
     convert(train[lagged_cols[1]].values),
     convert(train[lagged_cols[2]].values),
     convert(train[lagged_cols[3]].values),
     convert(train[lagged_cols[4]].values),
     convert(train[lagged_cols[5]].values)],
    train[target_cols[0:6]].reindex().values,
    [convert(validate[lagged_cols[0]].values),
     convert(validate[lagged_cols[1]].values),
     convert(validate[lagged_cols[2]].values),
     convert(validate[lagged_cols[3]].values),
     convert(validate[lagged_cols[4]].values),
     convert(validate[lagged_cols[5]].values)],
    validate[target_cols[0:6]].reindex().values,
    num_epochs=epochs,
    batch_size=32,
    callback_list=[tensorboard_callback])



loss, accuracy, mae = model.evaluate(x=[convert(test[lagged_cols[0]].values),
                                        convert(test[lagged_cols[1]].values),
                                        convert(test[lagged_cols[2]].values),
                                        convert(test[lagged_cols[3]].values),
                                        convert(test[lagged_cols[4]].values),
                                        convert(test[lagged_cols[5]].values)],
                                     y=test[target_cols[0:6]].values)


print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))


y_sim = model.predict([convert(test[lagged_cols[0]].values),
                       convert(test[lagged_cols[1]].values),
                       convert(test[lagged_cols[2]].values),
                       convert(test[lagged_cols[3]].values),
                       convert(test[lagged_cols[4]].values),
                       convert(test[lagged_cols[5]].values)])


metrics = ModelMetrics.ModelMetrics()
pairs = list(zip(target_cols, range(0,6)))


all_metrics = metrics.report_all_metrics('Lagged7Dense', pairs, test[target_cols], y_sim)

series_plot = SeriesPlot.SeriesPlot()
series_plot.plot_series(pairs, test['dates'].values, test[target_cols], y_sim)

series_plot.plot_histograms(pairs, test[target_cols], y_sim)