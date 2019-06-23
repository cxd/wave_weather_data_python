## Exploring Prediction of Wave Buoy Results from Historical Data.

This is an experiment on predicting wave buoy readings with an interest in the relationship between weather station readings and wave buoy readings.

The question is whether it is possible to learn a relationship between weather readings onshore that will have some skill in predicting wave buoy sensed data.

This project is an experiment in using a simple network architecture to explore the potential for modelling such data.



In general applications of modelling wave buoy data from time series 
include providing predictions for energy produced by waves, 
as well as estimates of near shore surges during extreme weather events
or perhaps simply to predict surf conditions for your local break ;) 


## Example Notebooks

An ipython notebooks running in google colabatory are available here:

- [Exploring Prediction of Wave Buoy Results from Historical Data.](https://colab.research.google.com/drive/1IR4FKSawpl7wgNghDwV2JsjHtF0nLzo-)

- [Feature Engineering Wave Buoy Data with Wavelets](https://colab.research.google.com/drive/1HE7ecPC5hw_AYyVvkElPqR_tvEsnHiX5)

- [Predicting Lagged Wave Buoy Data with a CNN.](https://colab.research.google.com/drive/1NTvR5Gc758aOIrd4e2DjEy9HlaEW0cPb)

## Data Sources

### Wave Data
Wave data is available from:

https://data.qld.gov.au/dataset?q=Coastal%20Data%20System%20–%20Waves
Field names are;
- Hs - Significant wave height, an average of the highest third of the waves in a record (26.6 minute recording period).
- Hmax - The maximum wave height in the record.
- Tz - The zero upcrossing wave period.
- Tp - The peak energy wave period.
- Dir_Tp TRUE - Direction (related to true north) from which the peak period waves are coming from.
- SST - Approximation of sea surface temperature.

Currently in early phases the 5 target variables are the variables of interest. 

## Wave Buoys selected

The following wave buoy were selected.

- caloundra
- mooloolaba
- brisbane
- moreton bay north
- gold coast

### Climate Station Observations

The climate station observation data is available from the BOM.
http://www.bom.gov.au/climate/change/datasets/datasets.shtml

The weather station data sets contain readings for:

- Evapo-transpiration (mm)
- Rain (mm)
- Pan Evaporation (mm)
- Maximum Temperature (degrees celcius)
- Minimum Temperature (degrees celcius)
- Maximum Relative Humidity (%)
- Minimum Relative Humidity (%)
- Average wind speed 10m above sea level (m/sec)
- Solar Radiation (MJ/sq m)


### Stations selected.

- "redcliffe"
- "brisbane",
- "sunshine_coast_airport",
- "cape_moreton_lighthouse",
- "gold_coast_seaway"

The mappings of buoy to weather station is very rough, the correspondances are loosely based on proximity to the buoy from the weather station. These were:

```
         waveSite             climateSite
1       caloundra               redcliffe
2        brisbane                brisbane
3      mooloolaba  sunshine_coast_airport
4 northmoretonbay cape_moreton_lighthouse
5       goldcoast       gold_coast_seaway
```

From an initial exploration of the data there is a mild correlation between the sea surface temperature and the Pan Evaporation and Maximum Temperature readings from the weather station.
There is a mild negative correlation between the average 10m windspeed readings at the site and the wave direction from true north (in completely different units) as well as the mean peak energy wave period and zero upcrossing wave period.

Several different methods of preprocessing and data arrangements will need to be attempted as well in as assessing the model skill in predicting the target variability in the wave buoy outputs.


5 models are constructed all of which are simple feedforward network models with 2 hidden layers and a single linear output layer.
​
The differences between these models are as follows.
​
- __Model 1__ simple dense network with inputs for climate observation data, hidden layer with linear activation, hidden layer with relu activation, output linear layer.
- __Model 2__ the same architecture as the first, but lagged wavelet coefficients for the previous 1 day averages of the wave data. A db3 wavelet function is used to generate the features.
- __Model 3__ the same architecture as model 2, all inputs are scaled using min/max normalisation.
- __Model 4__ the same as model 3, however the 2nd hidden layer uses sigmoid activation.
- __Model 5__ the same as model 3, the 2nd hidden layer uses tanh activation.
​
​


During training it was observed that overfitting started to occur at the following number of epochs.
​
- __Model 1__ > 145 epochs
- __Model 2__ > 90 epochs
- __Model 3__ > 60 epochs
- __Model 4__ > 190 epochs
- __Model 5__ 1000 epochs were observed with no overfitting on validation data.
​



Comparison of the models suggest __Model 4__ performs best on predicting the target variables than the other models.
​

## Modelling with 7-Day Lags - Example Notebook

The time series lags for t-7 to t-1 days was preprocessed with a Daubechies wavelet decomposition. 
This approach produced a model that out performed the __Model 4__ in predicting all six target variables for time t.
The notebook for this model is found at [Feature Engineering Wave Buoy Data with Wavelets](https://colab.research.google.com/drive/1HE7ecPC5hw_AYyVvkElPqR_tvEsnHiX5).

While the wavelet decomposition is very effective, it would be of interest to experiment with a network architecture
that is able to learn the decomposition directly from the input sequence. 
For this purpose the next steps are to experiment with a CNN architecture using the number of filters to
determine the number of decomposition coefficients. This may require several parallel layers with different filter sizes.

