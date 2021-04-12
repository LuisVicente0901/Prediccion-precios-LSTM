# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:00:18 2021

@author: Luis Vicente
"""




from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

# download dataframe
data = pdr.get_data_yahoo("GMEXICOB.MX", start="2010-01-05", end="2016-12-30")
data.head()
data.to_csv("C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_G_Mexico.csv")

