# mo810-stock-predict
Stock Market Prediction

# Dataset

The ideal dataset should include, for every publicly-traded company in the world:
- Minute-by-minute values of stock
- Summary of the fundamentals (revenue, expenses, assets, liabilities)
- Analyst's recommendations (Buy, Hold, Sell, etc)
- Analysis of the news and associated sentiment.
- Indicators on how it is being influenced by everything else (competitors, politics, economy, etc)

While most of this informations is supposedly public, there is a number of obstacles:

- Detailed datasets on the historical data of stock prices, fundamentals, etc, are widely available, but only for those who pay a very significant fee. [¹](https://quantquote.com/historical-stock-data)
- Analyst's recommendations are often wrong, in conflict with each other, and not consistently available for every company, which makes them very hard to handle.
- News and sentiment analysis is a really though problem on it's own, and we are not taking the challenge at this time. (There are other groups planning to do related stuff -- Maybe partner with them?)
- Indicators on how the whole world influences one stock is a very, very complex, poorly defined and unstructured problem, and we will make no attempt to handle it.

In face of these obstacles, we will settle for a much-simplified, easily obtained and hopefully good-enough dataset: The daily summaries of US company's performance: Open, Close, Min and Max prices, and amount of transations.

This is large, well-known dataset, with several free sources. Initially, we will use [the one available on Kaggle](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs), which is reasonaly up-to-date (November/2017) and goes back several decades. We may switch for an updated dataset later on if it becomes available.

Unfortunately, the lack of real-time pricing will limit our prediction performance on day trading and short-term investments -- which would actually be the best possible scenario for an automatic trader, since a company's fundamentals, world economy, etc doesn't normally change within a day, even though people pay considerably different prices for a company within a day without anything special happening.


# Training considerations

There is a really high correlation between the prices of every stock in the planet -- A company goes bankrupt in Japan, and and every US company loses market value in the next day. Therefore, we must be careful not to use the same period for both validation and training. 

Also, day-of-week, day-of-month and holidays also influence the stock market. 

A possible approach would be to pick a random validation weekday for each week.

More difficult and much more importantly, Quarterly Earnings Report, Product launches and scandals all cause major shifts in stock prices, and might need special handling.
