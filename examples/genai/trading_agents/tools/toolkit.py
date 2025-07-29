from datetime import datetime

import pandas as pd
import tools.interface as interface
import yfinance as yf
from flyte_env import env

from flyte.io import File


@env.task
async def get_reddit_news(
    curr_date: str,  # Date you want to get news for in yyyy-mm-dd format
) -> str:
    """
    Retrieve global news from Reddit within a specified time frame.
    Args:
        curr_date (str): Date you want to get news for in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest global news
        from Reddit in the specified time frame.
    """

    global_news_result = interface.get_reddit_global_news(curr_date, 7, 5)

    return global_news_result


@env.task
async def get_finnhub_news(
    ticker: str,  # Search query of a company, e.g. 'AAPL, TSM, etc.
    start_date: str,  # Start date in yyyy-mm-dd format
    end_date: str,  # End date in yyyy-mm-dd format
) -> str:
    """
    Retrieve the latest news about a given stock from Finnhub within a date range
    Args:
        ticker (str): Ticker of a company. e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing news about the company
        within the date range from start_date to end_date
    """

    end_date_str = end_date

    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    look_back_days = (end_date - start_date).days

    finnhub_news_result = interface.get_finnhub_news(ticker, end_date_str, look_back_days)

    return finnhub_news_result


@env.task
async def get_reddit_stock_info(
    ticker: str,  # Ticker of a company. e.g. AAPL, TSM
    curr_date: str,  # Current date you want to get news for
) -> str:
    """
    Retrieve the latest news about a given stock from Reddit, given the current date.
    Args:
        ticker (str): Ticker of a company. e.g. AAPL, TSM
        curr_date (str): current date in yyyy-mm-dd format to get news for
    Returns:
        str: A formatted dataframe containing the latest news about the company on the given date
    """

    stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)

    return stock_news_results


@env.task
async def get_YFin_data(
    symbol: str,  # ticker symbol of the company
    start_date: str,  # Start date in yyyy-mm-dd format
    end_date: str,  # End date in yyyy-mm-dd format
) -> str:
    """
    Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data
        for the specified ticker symbol in the specified date range.
    """

    result_data = interface.get_YFin_data(symbol, start_date, end_date)

    return result_data


@env.task
async def get_YFin_data_online(
    symbol: str,  # ticker symbol of the company
    start_date: str,  # Start date in yyyy-mm-dd format
    end_date: str,  # End date in yyyy-mm-dd format
) -> str:
    """
    Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data
        for the specified ticker symbol in the specified date range.
    """

    result_data = interface.get_YFin_data_online(symbol, start_date, end_date)

    return result_data


@env.task  # TODO: This can be a flyte.trace
async def cache_market_data(symbol: str, start_date: str, end_date: str) -> File:
    data_file = f"{symbol}-YFin-data-{start_date}-{end_date}.csv"

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        multi_level_index=False,
        progress=False,
        auto_adjust=True,
    )
    data = data.reset_index()
    data.to_csv(data_file, index=False)

    return await File.from_local(data_file)


@env.task
async def get_stockstats_indicators_report(
    symbol: str,  # ticker symbol of the company
    indicator: str,  # technical indicator to get the analysis and report of
    curr_date: str,  # The current trading date you are trading on, YYYY-mm-dd
    look_back_days: int = 30,  # how many days to look back
) -> str:
    """
    Retrieve stock stats indicators for a given ticker symbol and indicator.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the stock stats indicators
        for the specified ticker symbol and indicator.
    """

    today_date = pd.Timestamp.today()

    end_date = today_date
    start_date = today_date - pd.DateOffset(years=15)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    data_file = await cache_market_data(symbol, start_date, end_date)
    local_data_file = await data_file.download()

    result_stockstats = interface.get_stock_stats_indicators_window(
        symbol, indicator, curr_date, look_back_days, False, local_data_file
    )

    return result_stockstats


@env.task
async def get_stockstats_indicators_report_online(
    symbol: str,  # ticker symbol of the company
    indicator: str,  # technical indicator to get the analysis and report of
    curr_date: str,  # The current trading date you are trading on, YYYY-mm-dd"
    look_back_days: int = 30,  # "how many days to look back"
) -> str:
    """
    Retrieve stock stats indicators for a given ticker symbol and indicator.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the stock stats indicators
        for the specified ticker symbol and indicator.
    """

    today_date = pd.Timestamp.today()

    end_date = today_date
    start_date = today_date - pd.DateOffset(years=15)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    data_file = await cache_market_data(symbol, start_date, end_date)
    local_data_file = await data_file.download()

    result_stockstats = interface.get_stock_stats_indicators_window(
        symbol, indicator, curr_date, look_back_days, True, local_data_file
    )

    return result_stockstats


@env.task
async def get_finnhub_company_insider_sentiment(
    ticker: str,  # ticker symbol for the company
    curr_date: str,  # current date of you are trading at, yyyy-mm-dd
) -> str:
    """
    Retrieve insider sentiment information about a company (retrieved
    from public SEC information) for the past 30 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 30 days starting at curr_date
    """

    data_sentiment = interface.get_finnhub_company_insider_sentiment(ticker, curr_date, 30)

    return data_sentiment


@env.task
async def get_finnhub_company_insider_transactions(
    ticker: str,  # ticker symbol
    curr_date: str,  # current date you are trading at, yyyy-mm-dd
) -> str:
    """
    Retrieve insider transaction information about a company
    (retrieved from public SEC information) for the past 30 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transactions/trading information in the past 30 days
    """

    data_trans = interface.get_finnhub_company_insider_transactions(ticker, curr_date, 30)

    return data_trans


@env.task
async def get_simfin_balance_sheet(
    ticker: str,  # ticker symbol
    freq: str,  # reporting frequency of the company's financial history: annual/quarterly
    curr_date: str,  # current date you are trading at, yyyy-mm-dd
):
    """
    Retrieve the most recent balance sheet of a company
    Args:
        ticker (str): ticker symbol of the company
        freq (str): reporting frequency of the company's financial history: annual / quarterly
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's most recent balance sheet
    """

    data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)

    return data_balance_sheet


@env.task
async def get_simfin_cashflow(
    ticker: str,  # ticker symbol
    freq: str,  # reporting frequency of the company's financial history: annual/quarterly
    curr_date: str,  # current date you are trading at, yyyy-mm-dd
) -> str:
    """
    Retrieve the most recent cash flow statement of a company
    Args:
        ticker (str): ticker symbol of the company
        freq (str): reporting frequency of the company's financial history: annual / quarterly
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
            str: a report of the company's most recent cash flow statement
    """

    data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)

    return data_cashflow


@env.task
async def get_simfin_income_stmt(
    ticker: str,  # ticker symbol
    freq: str,  # reporting frequency of the company's financial history: annual/quarterly
    curr_date: str,  # current date you are trading at, yyyy-mm-dd
) -> str:
    """
    Retrieve the most recent income statement of a company
    Args:
        ticker (str): ticker symbol of the company
        freq (str): reporting frequency of the company's financial history: annual / quarterly
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
            str: a report of the company's most recent income statement
    """

    data_income_stmt = interface.get_simfin_income_statements(ticker, freq, curr_date)

    return data_income_stmt


@env.task
async def get_google_news(
    query: str,  # Query to search with
    curr_date: str,  # Curr date in yyyy-mm-dd format
) -> str:
    """
    Retrieve the latest news from Google News based on a query and date range.
    Args:
        query (str): Query to search with
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): How many days to look back
    Returns:
        str: A formatted string containing the latest news from Google News
        based on the query and date range.
    """

    google_news_results = interface.get_google_news(query, curr_date, 7)

    return google_news_results


@env.task
async def get_stock_news_openai(
    ticker: str,  # the company's ticker
    curr_date: str,  # Current date in yyyy-mm-dd format
) -> str:
    """
    Retrieve the latest news about a given stock by using OpenAI's news API.
    Args:
        ticker (str): Ticker of a company. e.g. AAPL, TSM
        curr_date (str): Current date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing the latest news about the company on the given date.
    """

    openai_news_results = interface.get_stock_news_openai(ticker, curr_date)

    return openai_news_results


@env.task
async def get_global_news_openai(
    curr_date: str,  # Current date in yyyy-mm-dd format
) -> str:
    """
    Retrieve the latest macroeconomics news on a given date using OpenAI's macroeconomics news API.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing the latest macroeconomic news on the given date.
    """

    openai_news_results = interface.get_global_news_openai(curr_date)

    return openai_news_results


@env.task
async def get_fundamentals_openai(
    ticker: str,  # the company's ticker
    curr_date: str,  # Current date in yyyy-mm-dd format
) -> str:
    """
    Retrieve the latest fundamental information about a given stock
    on a given date by using OpenAI's news API.
    Args:
        ticker (str): Ticker of a company. e.g. AAPL, TSM
        curr_date (str): Current date in yyyy-mm-dd format

    Returns:
        str: A formatted string containing the latest fundamental information
        about the company on the given date.
    """

    openai_fundamentals_results = interface.get_fundamentals_openai(ticker, curr_date)

    return openai_fundamentals_results
