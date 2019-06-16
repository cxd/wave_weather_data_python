import pandas as pd
import numpy as py
import pywt
from lib import ReadCsv
from lib import ReadConfig
from lib import modwt

class ReadData:

    def __init__(self):
        self.all_data = []

    def convertToDate(self, data, column, target_column):
        dt = pd.to_datetime(data[column], infer_datetime_format=True)
        dt = dt.dt.floor('d')
        data[target_column] = dt
        data = data[data[target_column].notnull()]
        return data

    def readClimateFiles(self, config_data, add_wavelets=False, wavelet_cols=[], wavelet='db8'):
        # Read all files provided by the configuration and return a merged dataset.
        all_data = None
        climateReader = ReadCsv.ReadCsv(config_data['climate']['baseDir'])
        waveReader = ReadCsv.ReadCsv(config_data['wave']['baseDir'])

        for pair in config_data["siteMap"]:
            wave_site = pair["wave"]
            climate_site = pair["climate"]
            wave_data = waveReader.process_directory(wave_site,
                                                     config_data["wave"]["skipRows"],
                                                     config_data["wave"]["columns"])
            climate_data = climateReader.process_directory(climate_site,
                                                           config_data["climate"]["skipRows"],
                                                           config_data["climate"]["columns"])
            climate_data = climate_data.drop("Space", axis="columns")

            climate_data = self.convertToDate(climate_data, "date", "local_date")

            wave_data = self.convertToDate(wave_data, "DateTime", "local_date")


            wave_data = wave_data[(wave_data["Hs"] >= 0) &
                                  (wave_data["Hmax"] >= 0) &
                                  (wave_data["Tz"] >= 0) &
                                  (wave_data["Tp"] >= 0) &
                                  (wave_data["DirTpTRUE"] >= 0) &
                                  (wave_data["SST"] >= 0)]



            float_types = ["Evapotranspiration_mm",
                           "Rain_mm",
                           "PanEvaporation_mm",
                           "MaximumTemperature_C",
                           "MaxRelativeHumidity_pc",
                           "MinRelativeHumidity_pc",
                           "Avg10mWindSpeed_m_sec",
                           "SolarRadiation_MJ_sqm"]

            for col in float_types:
                climate_data[col] = pd.to_numeric(climate_data[col], errors='coerce')
                climate_data = climate_data[(climate_data[col] >= 0) & (climate_data[col].notnull())]


            summary_wave_data = wave_data[['local_date',
                                           'Hs',
                                           'Hmax',
                                           'Tz',
                                           'Tp',
                                           'DirTpTRUE',
                                           'SST']].groupby('local_date').mean()

            site = pd.Series([pair["wave"]])
            site = site.repeat(summary_wave_data.shape[0])
            summary_wave_data = summary_wave_data.assign(site = site.values)

            merged_data = summary_wave_data.merge(climate_data, on='local_date', how='inner')

            if add_wavelets is True and len(wavelet_cols) > 0:
                merged_data = self.add_wavelet_coefficients(merged_data, wavelet_cols, wavelet)
                merged_data = merged_data.dropna()


            if all_data is None:
                all_data = merged_data
            else:
                all_data = pd.concat([all_data, merged_data])

        all_data = all_data.sort_values(by=['local_date', 'site_x'])
        self.all_data = all_data

        return all_data

    def add_wavelet_coefficients(self, all_data, columns, wavelet="db3", prefix="", lag=1):
        # Extend the dataset by adding wavelet coefficients to the selected columns.
        # Compute the maximal overlap discrete wavelet transform
        # and obtain 3 wavelet coefficients that are to be added to the feature vectors
        # the vectors will be shifted by a specified lag amount.
        for col in columns:
            C1, C3, C3, A = modwt.modwt(all_data[col].values, wavelet, 3)
            nameA = prefix+col+"_A1"
            name1 = prefix+col+"_C1"
            name2 = prefix+col+"_C2"
            name3 = prefix+col+"_C3"
            # add the coefficients shifted by the specified lag amount
            all_data[nameA] = pd.Series(A).shift(lag)
            all_data[name1] = pd.Series(C1).shift(lag)
            all_data[name2] = pd.Series(C2).shift(lag)
            all_data[name3] = pd.Series(C3).shift(lag)
        return all_data


    def getsite(self, data, col, site):
        # Select the subset of the data where the site matches the value of the row.
        return data[(data[col] == site)]



    def make_lags(self, data, fromN, maxN):
        # Generate a lag from step fromN to step maxN
        for i in range(fromN,maxN):
            nextData = data.shift(i).dropna()
            colnames = list(map(lambda col: col+'_t-'+str(i), nextData.columns))
            nextData.columns = colnames
            yield nextData

    def make_lags_per_site(self, data, sitecol):
        # Lag each of the series per site.
        target_set = None
        sitenames = data[sitecol].unique()
        for site in sitenames:
            data = self.getsite(data, sitecol, site)
            data.index = range(0,data.shape[0])
            lags = list(self.make_lags(data, 1,8))
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
        return target_set

    def getlagged_columns(self, seed_cols, lag_start, lag_end):
        temp = []
        for col in seed_cols:
            for i in range(lag_start,lag_end):
                temp.append(col+'_t-'+str(i))
        seed_cols.extend(temp)
        return seed_cols

    def getlagged_group_columns(self, seed_cols, lag_start, lag_end):
        temp = []
        group_cols = []
        for col in seed_cols:
            temp = []
            for i in range(lag_start,lag_end):
                temp.append(col+'_t-'+str(i))
            group_cols.append(temp)
        return group_cols



    def make_windowed_wavelet_decomp(self, data, seed_cols, sitecol, lag_start, lag_end, wavelet='db3'):
        # Compute wavelet decomposition for wavelet data.
        norm_data = None
        sitenames = data[sitecol].unique()
        seed_cols = self.getlagged_columns(seed_cols, lag_start, lag_end)
        wavelet_cols = []
        wavelet_data=None
        for site in sitenames:
            data = self.getsite(data, sitecol, site)
            data = data[seed_cols]
            for col in seed_cols:
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
        return wavelet_cols, wavelet_data

