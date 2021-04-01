# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:00:18 2021

@author: Luis Vicente
"""


#import yfinance as yf

#msft = yf.Ticker("AEROMEX.MX")
#msft.info
#hist = msft.history(start="2017-01-01", end="2017-04-30")
#hist[['Open','High']]

#data = yf.download("AEROMEX.MX", start="2017-01-01", end="2017-04-30")


from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

# download dataframe
data = pdr.get_data_yahoo("^MXX", start="2017-03-30", end="2021-03-30")
data.head()
data.to_csv("C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_IPC.csv")
