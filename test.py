# !/usr/bin/python3
# %%
import sys
import os

import numpy as np
from mkv import mkv
import matplotlib.pyplot as plt
# This is for tex plots according to acm format
from datetime import datetime



FILENAME = "test.mkv"
TITLE_TO_LOAD = "powermeter15"
# %%
myInfo = mkv.info(FILENAME)
print([s["title"] for s in myInfo["streams"]])
try:
    stream = next(s["streamIndex"] for s in myInfo["streams"] if s["title"] == TITLE_TO_LOAD)
except StopIteration:
    sys.exit("cannot find stream") 

dataDict = mkv.load(FILENAME, streamsToLoad=[stream])[0]

print(dataDict)

# %%
start = float(dataDict["metadata"]["TIMESTAMP"])
end = start+(len(dataDict["data"])/dataDict["samplingrate"])
timestamps = np.linspace(start, end, len(dataDict["data"]))
dates = [datetime.fromtimestamp(ts) for ts in timestamps]
# Plot data
fig, ax = plt.subplots()
ax.plot(dates, dataDict["data"]["p"], label="active power")
ax.plot(dates, dataDict["data"]["q"], label="reactive power")
# Format plot
ax.set(xlabel='Time of day', ylabel='Power [W/var]', title='Espresso Machine')
fig.autofmt_xdate()
plt.show()

# %%

# %%
