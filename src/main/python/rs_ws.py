from config import Config
import requests
from lxml import etree
import time
import random
import pandas as pd
import numpy as np


def KMBT2Number(s):
    if s.find('K') > -1:
        return float(s.split('K')[0]) * 1000
    elif s.find('M') > -1:
        return float(s.split('M')[0]) * 1000000
    elif s.find('B') > -1:
        return float(s.split('B')[0]) * 1000000000000
    elif s.find('T') > -1:
        return float(s.split('T')[0]) * 1000000000000000000
    else:
        return s


# custom class

result = []

for offset in range(0, 2300, 100):
    complete_list = "https://finance.yahoo.com/screener/unsaved/4b6788ec-db90-477a-a44a-09e5cd5b5027?count=100&offset=" + \
        str(offset)
    complete_list_resp = requests.get(complete_list)
    print(complete_list)
    complete_list_resp.encoding = 'utf-8'
    complete_list_selector = etree.HTML(complete_list_resp.text)
    print(complete_list_selector.xpath(
        '//*[@id="fin-scr-res-table"]/div[1]/div[1]/span[2]/span//text()')[0])
    number_of_items = int(complete_list_selector.xpath(
        '//*[@id="fin-scr-res-table"]/div[1]/div[1]/span[2]/span//text()')[0].split(' ')[0].split('-')[1]) % 100
    number_of_items = 100 if number_of_items == 0 else number_of_items
    print(number_of_items)
    for index in range(1, number_of_items + 1):
        symbol = complete_list_selector.xpath(
            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(index) + ']/td[1]/a//text()')[0]
        print(symbol)
        name = complete_list_selector.xpath(
            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(index) + ']/td[2]//text()')[0]
        price = complete_list_selector.xpath(
            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(index) + ']/td[3]/span//text()')[0]
        change = complete_list_selector.xpath(
            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(index) + ']/td[4]/span//text()')[0]
        change_percentage = complete_list_selector.xpath(
            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(index) + ']/td[5]/span//text()')[0]
        volume = complete_list_selector.xpath(
            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(index) + ']/td[6]/span//text()')[0]
        market_cap = complete_list_selector.xpath(
            '///*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(index) + ']/td[8]/span//text()')[0]

        statistics = 'https://finance.yahoo.com/quote/' + \
            symbol + '/key-statistics?p=' + symbol

        time.sleep(random.randint(1, 2))
        statistics_resp = requests.get(statistics)
        statistics_resp.encoding = 'utf-8'
        statistics_selector = etree.HTML(statistics_resp.text)

        trailing_p_e = None
        revenue = None
        total_cash = None
        total_debt = None
        five_year_average_dividend_yield = None

        try:
            trailing_p_e = statistics_selector.xpath(
                '//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[1]/div[2]/div/div[1]/div[1]/table/tbody/tr[3]/td[2]//text()')[0]
        except:
            print(symbol, "Unable to find trailing p/e")

        try:
            revenue = statistics_selector.xpath(
                '//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[3]/div/div[4]/div/div/table/tbody/tr[1]/td[2]//text()')[0]
        except:
            print(symbol, "Unable to find revenue")

        try:
            total_cash = statistics_selector.xpath(
                '//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[3]/div/div[5]/div/div/table/tbody/tr[1]/td[2]//text()')[0]
        except:
            print(symbol, "Unable to find total_cash")

        try:
            total_debt = statistics_selector.xpath(
                '//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[3]/div/div[5]/div/div/table/tbody/tr[3]/td[2]//text()')[0]
        except:
            print(symbol, "Unable to find total_debt")

        try:
            five_year_average_dividend_yield = statistics_selector.xpath(
                '//*[@id="Col1-0-KeyStatistics-Proxy"]/section/div[3]/div[2]/div/div[3]/div/div/table/tbody/tr[5]/td[2]//text()')[0]
        except:
            print(symbol, "Unable to find five_year_average_dividend_yield")

        profile = 'https://finance.yahoo.com/quote/' + symbol + '/profile?p=' + symbol
        time.sleep(random.randint(1, 2))
        profile_resp = requests.get(profile)
        profile_resp.encoding = 'utf-8'
        profile_selector = etree.HTML(profile_resp.text)

        sector = None
        industry = None

        try:
            sector = profile_selector.xpath(
                '//*[@id="Col1-0-Profile-Proxy"]/section/div[1]/div/div/p[2]/span[2]//text()')[0]
        except:
            print(symbol, "Unable to find sector")

        try:
            industry = profile_selector.xpath(
                '//*[@id="Col1-0-Profile-Proxy"]/section/div[1]/div/div/p[2]/span[4]//text()')[0]
        except:
            print(symbol, "Unable to find industry")

        result_item = []
        result_item.append(symbol)
        result_item.append(name)
        result_item.append(price)
        result_item.append(change)
        result_item.append(change_percentage.split("%")[0])
        result_item.append(KMBT2Number(market_cap))
        result_item.append(trailing_p_e)
        result_item.append(revenue)
        result_item.append(KMBT2Number(volume))
        result_item.append(KMBT2Number(total_cash))
        result_item.append(KMBT2Number(total_debt))
        result_item.append(five_year_average_dividend_yield)
        result_item.append(sector)
        result_item.append(industry)
        result.append(result_item)

df = pd.DataFrame(data=np.array(result))
df.columns = ['symbol', 'name', 'price', 'change', 'change_percentage',
              'market_cap', 'trailing_p_e', 'revenue', 'volume', 'total_cash', 'total_debt', 'five_year_average_dividend_yield',
              'sector', 'industry']
print(df)
df.to_csv(Config.getNNProductFileInput(), index=False)
