import numpy as np
from scipy.stats import norm
import scipy
import pandas as pd
import datetime
from dateutil import parser
import time
import matplotlib.pyplot as plt



SnP500 = pd.read_csv('DataCallS&P500.tsv', sep='\t')
SnP500.sample(frac=1)

for i in range(len(SnP500["Date"])):
    date_i = parser.parse(SnP500["Date"].iloc[i]).date()
    delta = date_i - DATE
    delta = delta.days
    SnP500["Date"].iloc[i] = delta / 365
"""
"""
SnP500['Strike'] = pd.to_numeric(SnP500['Strike'])
listT_K = SnP500[['Date', 'Strike']].values.tolist()
listObs = SnP500["Prix"].values.tolist()
