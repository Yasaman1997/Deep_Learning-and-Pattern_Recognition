LSTM ON YAHOO FINANCE FOR PREDICTIONS

BASED ON [THIS](https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/)


**Introduction**

Time series analysis refers to the analysis of change in the trend of the data over a period of time. Time series analysis has a variety of applications. One such application is the prediction of the future value of an item based on its past values. Future stock price prediction is probably the best example of such an application. In this article, we will see how we can perform time series analysis with the help of a recurrent neural network. We will be predicting the future stock prices of the Apple Company (AAPL), based on its stock prices of the past 5 years.

**Dataset**

The data that we are going to use for this article can be downloaded from Yahoo Finance. For training our algorithm, we will be using the Apple stock prices from 1st January 2013 to 31 December 2017. For the sake of prediction, we will use the Apple stock prices for the month of January 2018. So in order to evaluate the performance of the algorithm, download the actual stock prices for the month of January 2018 as well.



![capture](https://user-images.githubusercontent.com/24721389/52470635-fd3fef00-2ba3-11e9-98d8-8527ed167748.PNG)
