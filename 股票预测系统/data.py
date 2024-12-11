import tushare as ts
import os
from datetime import datetime

# 设置 Tushare API Token
ts.set_token('733a5d49573d1b2329100b4798f69f6e7f3c9c0c7e5a1244a0ac3556')
pro = ts.pro_api()  # 初始化 Tushare Pro API 客户端


# 获取股票数据
def get_stock_data(ticker, start_date, end_date):
    # 使用 Tushare 获取股票历史数据
    # 这里使用的是 Tushare Pro 的接口，返回的数据会包含股票的开盘价、收盘价、最高价、最低价、成交量等
    df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
    return df


tickers = ['600000.SH', '000001.SZ', '601577.SH', '600967.SH']
start_date = '20180101'  # Tushare 日期格式是 YYYYMMDD
end_date = '20241206'
directory = 'StockData'

# 如果目录不存在则创建
if not os.path.exists(directory):
    os.makedirs(directory)

# 遍历每个股票代码，下载并保存为 CSV 文件
for ticker in tickers:
    data = get_stock_data(ticker, start_date, end_date)
    # 保存数据为 CSV 文件，注意 Tushare 返回的数据列较多，可以选择保存需要的列
    data.to_csv(f'{directory}/{ticker}.csv', index=False)  # 保存 CSV 文件，`index=False` 防止保存行索引
