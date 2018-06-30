from binance.client import Client
from binance.websockets import BinanceSocketManager
import json
import pandas as pd
import matplotlib.pyplot as plt 

with open("./anhkhoabot_key.txt","r") as kf:
    line = kf.readlines()

api_key, api_secret = line[1], line[3]
api_secret = api_secret.replace("\n","")
api_key = api_key.replace("\n","")
client = Client(api_key, api_secret)

klines = pd.DataFrame(client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_1DAY, "30 May, 2017"))
klines.columns = ["Date", "Open", "High", "Low", "Close","Volume","Close time","Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume","Ignore" ]
klines.to_csv("A year.csv")
print(klines.shape)

open = klines['Open']
print(open)


#For realtime getting data
# def process_message(msg):
#     print("message type: {}".format(msg['e']))
#     print(json.dumps(msg, indent =4))

# bm = BinanceSocketManager(client)
# diff_key = bm.start_kline_socket('ETHBTC', process_message)
# bm.start()


