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

subset = all_data[['date','site_x', 'Hs','Hmax','Tz','Tp','DirTpTRUE','SST']]
subset.describe()

def make_date(series):
    for dt in series:
        yield datetime.strptime(dt, '%d/%m/%Y')

dates = list(make_date(subset['date']))

subset.index = range(0, subset.shape[0])

datesDf = pd.DataFrame({'dates': pd.Series(dates)}, index=range(0,len(dates)))

subset2 = pd.concat([subset, datesDf], axis=1)

subset2 = subset2.sort_values('dates')

idx1 = subset2.reindex(columns=['dates','size_x']).index
subset2.index = idx1

sitenames = subset2['site_x'].unique()

def getsite(data, col, site):
    return data[(data[col] == site)]

# 7 day lag.
def make_lags(data, fromN, maxN):
    for i in range(fromN,maxN):
        nextData = data.shift(i).dropna()
        colnames = list(map(lambda col: col+'_t-'+str(i), nextData.columns))
        nextData.columns = colnames
        yield nextData

target_set = None
for site in sitenames:
    data = getsite(subset2, 'site_x', site)
    data.index = range(0,data.shape[0])
    lags = list(make_lags(data, 1,8))
    minrows = lags[6].shape[0]
    target = data[6:minrows]
    for i in range(0,len(lags)):
        lags[i] = lags[i][i:minrows]
    lags.append(target)
    if target_set is None:
        target_set = pd.concat(lags, axis=1)
    else:
        temp = pd.concat(lags, axis=1)
        target_set = pd.concat([target_set, temp], axis=0)

target_set = target_set.dropna()

target_set[['Hs_t-7','Hs_t-6','Hs_t-5','Hs_t-4','Hs_t-3','Hs_t-2','Hs_t-1','Hs']].head(10)

# Now that we have timeseries data we now need to calculate wavelet decompositions for each
# window of time. Note that we are lagging only up to a period of 7 days.

norm_data = None
numeric_cols = ['Hs', 'Hmax','Tz','Tp','DirTpTRUE','SST']
temp = []
for col in numeric_cols:
    for i in range(1,8):
        temp.append(col+'_t-'+str(i))

numeric_cols.extend(temp)


wavelet='db3'
wavelet_cols = []
wavelet_data=None
for site in sitenames:
    data = getsite(target_set, 'site_x', site)
    data = data[numeric_cols]
    for col in numeric_cols:
        C1, C2, C3, A = modwt.modwt(data[col].values, wavelet, 3)
        nameA = col+"_A1"
        name1 = col+"_C1"
        name2 = col+"_C2"
        name3 = col+"_C3"
        wavelet_cols.append([nameA,name1,name2,name3])
        data[nameA] = pd.Series(A)
        data[name1] = pd.Series(C1)
        data[name2] = pd.Series(C2)
        data[name3] = pd.Series(C3)
    if wavelet_data is None:
        wavelet_data = data
    else:
        wavelet_data = pd.concat([wavelet_data, data], axis=0)

temp = []
for items in wavelet_cols:
    temp.extend(items)

wavelet_cols = list(set(temp))

# we now have our inputs for the wavelet coefficients C1,C2,C3 and the approximation A1

target_cols = ['Hs', 'Hmax','Tz','Tp','DirTpTRUE','SST']
group_cols = ['dates','site_x']

temp = target_cols
temp.extend(wavelet_cols)
numeric_data = wavelet_data[temp]

# min-max normalise the numeric input data.
data_min = numeric_data.min()
data_max = numeric_data.max()
data_delta = data_max - data_min

data_scaled = (numeric_data - data_min)/data_delta

data_scaled['dates'] = target_set['dates']
data_scaled['site_x'] = target_set['site_x']

# now we need to partition the data into train test and validate.
# however we need to balance this accross the sites.
modeller = NetworkModel.NetworkModel()

train = None
validate = None
test = None
for site in sitenames:
    data = getsite(data_scaled, 'site_x', site)
    train1, validate1, test1 = modeller.partition_data(data)
    if train is None:
        train = train1
        validate = validate1
        test = test1
    else:
        train = pd.concat([train, train1], axis=0)
        validate = pd.concat([validate, validate1], axis=0)
        test = pd.concat([test, test1], axis=0)

# Now we have train, validate and test data.
train = train.dropna()
validate = validate.dropna()
test = test.dropna()

# number of input columns
target_cols = ['Hs', 'Hmax','Tz','Tp','DirTpTRUE','SST']

input_size = len(wavelet_cols)
output_size = len(target_cols)


epochs = 500

logdir="logs/scalars/model_lag7" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model = modeller.model_dense(input_size, output_size)

modeller.compile_model(model, keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                       "mean_squared_error",
                       ["acc",
                        "mae"])

# model1 overfits at around 150 epochs
history = modeller.fit_model(
    model,
    train[wavelet_cols].reindex().values,
    train[target_cols].reindex().values,
    validate[wavelet_cols].reindex().values,
    validate[target_cols].reindex().values,
    num_epochs=epochs,
    batch_size=32,
    callback_list=[tensorboard_callback])

loss, accuracy, mae = model.evaluate(x=test[wavelet_cols].values,
                                     y=test[target_cols].values)

print("Loss: "+str(loss))
print("Accuracy: "+str(accuracy))
print("Mean Absolute Error: "+str(mae))

y_sim = model.predict(test[wavelet_cols].values)

metrics = ModelMetrics.ModelMetrics()
pairs = list(zip(target_cols, range(0,output_size)))


all_metrics = metrics.report_all_metrics('Lagged7Dense', pairs, test[target_cols], y_sim)

series_plot = SeriesPlot.SeriesPlot()
series_plot.plot_series(pairs, test['dates'].values, test[target_cols], y_sim)

series_plot.plot_histograms(pairs, test[target_cols], y_sim)