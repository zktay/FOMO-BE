import yfinance as yf
import pandas as pd
from typing import Dict, List, Union


class StockService:
    @staticmethod
    def get_stock_data(symbol: str, period: str = "1y") -> Dict[str, Union[str, List]]:
        """
        Fetch stock data from Yahoo Finance
        """
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)

            return {
                "symbol": symbol,
                "prices": hist['Close'].tolist(),
                "dates": hist.index.strftime('%Y-%m-%d').tolist(),
                "volumes": hist['Volume'].tolist(),
                "high": hist['High'].tolist(),
                "low": hist['Low'].tolist(),
            }
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic technical indicators
        """
        try:
            # Calculate Simple Moving Averages
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]

            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            return {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": rsi
            }
        except Exception as e:
            raise Exception(f"Error calculating indicators: {str(e)}")