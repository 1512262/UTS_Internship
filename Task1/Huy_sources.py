import requests
import json
import time

class binanceAPI:
    base_endpoint = "https://api.binance.com"
    res = None
    def getRes(self):
        return self.res
    def get(self, query):
        self.res = requests.get(self.base_endpoint+query)
        return self
    def getJson(self):
        self.res = self.res.json()
        return self
    def jsonString(self):
        self.res = json.dumps(self.res, indent=4)
        return self
    def ping(self):
        return self.get("/api/v1/ping")
    def checkServerTime(self):
        return self.get("/api/v1/time")
    def exchangeInformation(self):
        return self.get("/api/v1/exchangeInfo")
    def orderBook(self, symbol, limit=None):
        query = "/api/v1/depth"+"?symbol="+symbol
        if limit != None:
            query = query+"&limit="+limit
        return self.get(query)
    def recentTradesList(self, symbol, limit=None):
        query = "/api/v1/trades"+"?symbol="+symbol
        if limit != None:
            query = query+"&limit="+limit
        return self.get(query)
    def aggregateTradesList(self, symbol, fromId=None, startTime=None, endTime=None, limit=None):
    	query = "/api/v1/aggTrades"+"?symbol="+symbol
    	if fromId != None:
    		query = query+"&fromId="+fromId
    	if startTime != None:
    		query = query+"&starTime="+startTime
    	if endTime != None:
    		query = query+"&endTime="+endTime
    	if limit != None:
    		query = query+"&limit="+limit
    	return self.get(query)
    def klineData(self, symbol, interval, limit=None, startTime=None, endTime=None):
    	query = "/api/v1/klines"+"?symbol="+symbol+"&interval="+interval
    	if limit != None:
    		query = query+"&limit="+limit
    	if startTime != None:
    		query = query+"&startTime="+startTime
    	if endTime != None:
    		query = query+"&endTime="+endTime
    	return self.get(query)
    def tickerPriceChangeStatistics(self, symbol=None):
    	query = "/api/v1/ticker/24hr"
    	if symbol != None:
    		query = query+"?symbol="+symbol
    	return self.get(query)
    def symbolPriceTicker(self, symbol=None):
    	query = "/api/v3/ticker/price"
    	if symbol != None:
    		query = query+"?symbol="+symbol
    	return self.get(query)
    def symbolOrderBookTicker(self, symbol=None):
    	query = "/api/v3/ticker/bookTicker"
    	if symbol != None:
    		query = query+"?symbol="+symbol
    	return self.get(query)
        
b = binanceAPI()
filename = "data.csv"
symbol = "ETHBTC"

f = open(filename, "w")
f.write("ethbtc,time\n")
f.close()

while True:
    time.sleep(1)
    ethbtc = b.symbolPriceTicker(symbol)
    tm = str(time.time())
    if ethbtc.getRes().status_code == 200:
        ethbtc = ethbtc.getJson().getRes()["price"]
        f = open(filename, "a")
        f.write(ethbtc+","+tm+"\n")
        f.close()
    else:
        break