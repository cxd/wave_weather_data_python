import pandas as pd
import numpy as py
from lib import ReadCsv
from lib import ReadConfig

class ReadData:

    def __init__(self):
        self.all_data = []

    def convertToDate(self, data, column, target_column):
        dt = pd.to_datetime(data[column], infer_datetime_format=True)
        dt = dt.dt.floor('d')
        data[target_column] = dt
        data = data[data[target_column].notnull()]
        return data

    def readClimateFiles(self, config_data):
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
            if all_data is None:
                all_data = merged_data
            else:
                all_data = pd.concat([all_data, merged_data])
        self.all_data = all_data
        return all_data
