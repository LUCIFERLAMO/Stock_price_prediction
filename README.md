# Stock Price Prediction Using Machine Learning

This is a Python project that predicts the next trading day's **High** and **Low** stock prices for NSE-listed companies. The application collects recent historical stock data, trains machine learning models, and displays predictions and accuracy graphs in a Tkinter desktop interface.

## Team Members

- 261CAI33
- 261CAI34
- 261CAI35
- 261CAI49
- 261CAI50
- 261CAI18
- 261CAI03

## How the Project Works

1. The application downloads the current list of NSE stock symbols.
2. The user selects or enters a stock symbol, such as `RELIANCE`, `TCS`, or `INFY`.
3. The application collects historical price data from Stock Analysis using web scraping.
4. Pandas cleans the data and converts the prices, dates, percentage changes, and volume into usable values.
5. The project uses the current day's and previous day's market values as input features:
	- Open
	- High
	- Low
	- Close
	- Previous Close
	- Previous High
	- Previous Low
6. Two Linear Regression models are trained: one predicts the next day's High price and the other predicts the next day's Low price.
7. The predicted values, mean absolute error, and historical prediction graphs are shown in the desktop application.

## Technologies and Frameworks

- **Python**: Main programming language.
- **Tkinter**: Built-in Python framework for the graphical user interface.
- **Requests**: Downloads stock and NSE data from web pages.
- **Beautiful Soup**: Extracts historical stock data from HTML tables.
- **Pandas**: Cleans, organizes, and prepares the market data.
- **Scikit-learn**: Provides the Linear Regression models and mean absolute error calculation.
- **Matplotlib**: Displays historical prices and model predictions as charts.

## Requirements

The external Python packages are listed in [requirements.txt](requirements.txt). Python 3.9 or newer is recommended. Tkinter and `datetime` are included with most standard Python installations and are not installed through `requirements.txt`.

## Installation

1. Install Python from [python.org](https://www.python.org/downloads/). During installation on Windows, enable **Add Python to PATH**.
2. Open PowerShell or Command Prompt in this project folder.
3. Create a virtual environment:

	```powershell
	python -m venv .venv
	```

4. Activate the virtual environment on Windows:

	```powershell
	.\.venv\Scripts\Activate.ps1
	```

	If PowerShell blocks activation, run this once in PowerShell as your user:

	```powershell
	Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
	```

5. Install the project dependencies:

	```powershell
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	```

## Running the Project

With the virtual environment activated, run:

```powershell
python stock_price_predictor.py
```

Select a stock from the search box and click the prediction button. An internet connection is required because the application retrieves the latest NSE symbols and historical market data online.
