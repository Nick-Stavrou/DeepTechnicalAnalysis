"""
Created on Wed Apr 14 19:12:31 2021
@author: nmsta
"""
import pandas as pd
import yfinance as yf
from random import shuffle
from pyts.image import GramianAngularField
import numpy as np
from skimage import io
from tqdm import tqdm
import os

path_to_file = "/Users/nmsta/OneDrive/Documents/S&P500.xlsx"
save_path = "/Users/nmsta/Pictures/Train_Images"
num_samples = 50
num_periods = 64
split = 0.2

if not os.path.exists(save_path):
    os.makedirs(save_path)
tickers = list(pd.read_excel(path_to_file)["Symbol"].values)
shuffle(tickers)
gasf = GramianAngularField(image_size=num_periods, method='difference')

train_summary = []
j = 0
dataset = "val"
for ticker in tqdm(tickers):
    try:
        ohlc = yf.download(ticker, period="10y", progress=False)
        assert len(ohlc) > num_samples + num_periods + 1
    except:
        j += 1
        continue
    returns = ohlc["Adj Close"].pct_change()[1:]
    ohlc.drop(columns=["Adj Close"], inplace=True)
    
    if j >= int(len(tickers) * split):
        dataset = "train"
         
    for i in range(1, num_samples+1):
        response = returns.iloc[-i]
        train = ohlc.iloc[len(ohlc)-i-num_periods:len(ohlc)-i]
        ohlc_gasf = np.moveaxis(gasf.fit_transform(train.values.T), 0, -1)
        im_path = save_path + "/GADF_" + ticker + "_" + str(i) + ".tif"
        io.imsave(im_path, ohlc_gasf, compress=9)
        train_summary.append([ticker, im_path, response, dataset])
    j += 1
    
train_summary = pd.DataFrame(train_summary, columns=["Ticker", "Train_Path", "Response", "Dataset"])
train_summary.to_csv("/Users/nmsta/OneDrive/Documents/Train_Summary.csv", index=False)