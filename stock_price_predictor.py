import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime

def fetch_and_predict(stock_symbol):
    url = f"https://stockanalysis.com/quote/nse/{stock_symbol}/history/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return {"error": f"Error {response.status_code}. Could not connect or invalid stock symbol."}
    
    # --- giving the response to beautiful soup
    soup = BeautifulSoup(response.text,"html.parser")
    
    # ---- Finding all the tables 
    tables = soup.find_all("table")
    if not tables:
        return {"error": "Could not find data tables for this stock."}
        
    table = tables[0]
    
    # ------- finding the table from the website and storing it in a vairable called rows ----
    rows = table.find_all("tr")
    if len(rows) < 10:
        return {"error": "Not enough data available for this stock."}
    
    # ---- Extracting all the 50 rows
    stock_data = []
    
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
            
        date = cells[0].get_text(strip=True)
        open_price = cells[1].get_text(strip=True)
        high = cells[2].get_text(strip=True)
        low = cells[3].get_text(strip=True)
        close = cells[4].get_text(strip=True)
        adj_close = cells[5].get_text(strip=True)
        change = cells[6].get_text(strip=True)
        volume = cells[7].get_text(strip=True)
    
        stock_data.append([
            date, open_price, high, low, close, adj_close, change, volume
        ])
    
    # ---- Feading the recirds recived to pandas 
    df = pd.DataFrame(
        stock_data,
        columns=["Date", "Open", "High", "Low", "Close", "Adj_Close", "Change", "Volume"]
    )
    
    # --- converting the ojbject types into numeric data 
    df["Date"] = pd.to_datetime(df["Date"])
    df["Change"] = pd.to_numeric(df["Change"].str.replace("%",""), errors='coerce')
    df["Volume"] = pd.to_numeric(df["Volume"].str.replace(",",""), errors='coerce')
    
    numeric_column = ["Open","High","Low","Close","Adj_Close"]
    for col in numeric_column:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",",""), errors='coerce')
        
    df = df.dropna()
    
    if df.empty:
        return {"error": "Data could not be parsed."}
    
    # ---- sorting the data to assecnding order of date
    df = df.sort_values("Date").reset_index(drop=True)
    
    # --- creating tmr high and tmr low
    df["Next_high"] = df["High"].shift(-1)
    df["Next_low"] = df["Low"].shift(-1)
    
    df["Previous_Close"] = df["Close"].shift(1)
    df["Previous_High"] = df["High"].shift(1)
    df["Previous_Low"] = df["Low"].shift(1)
    
    # --- Making a sperate copy to train the model
    model_data = df.dropna(
        subset=["Previous_Close", "Previous_High", "Previous_Low", "Next_high", "Next_low"]
    ).copy()
    
    if len(model_data) < 10:
        return {"error": "Not enough valid data points for training."}
    
    latest_day = df.iloc[-1]
    latest_date = latest_day["Date"]
    
    #  ------ we will now train the model using only some features 
    x = model_data[["Open", "High", "Low", "Close", "Previous_Close", "Previous_High", "Previous_Low"]]
    
    # --- target (what we want the model to learn)
    y_high = model_data["Next_high"]
    y_low = model_data["Next_low"]
    
    # ---- train the model ----
    split_data = int(len(x) * 0.8)
    
    x_train = x.iloc[:split_data]
    x_test = x.iloc[split_data:]
    
    y_high_train = y_high.iloc[:split_data]
    y_high_test = y_high.iloc[split_data:]
    
    y_low_train = y_low.iloc[:split_data]
    y_low_test = y_low.iloc[split_data:]
    
    # ----- training the model
    high_model_lr = LinearRegression()
    high_model_lr.fit(x_train, y_high_train)
    
    low_model_lr = LinearRegression()
    low_model_lr.fit(x_train, y_low_train)
    
    # ---- make the predictions
    lr_high_prediction = high_model_lr.predict(x_test)
    lr_low_prediction = low_model_lr.predict(x_test)
    
    # --- check the MAE
    high_mae_lr = mean_absolute_error(y_high_test, lr_high_prediction)
    low_mae_lr = mean_absolute_error(y_low_test, lr_low_prediction)
    
    # -------------- giving the latest data to the model
    latest_data = pd.DataFrame([{
        "Open": latest_day["Open"],
        "High": latest_day["High"],
        "Low": latest_day["Low"],
        "Close": latest_day["Close"],
        "Previous_Close": latest_day["Previous_Close"],
        "Previous_High": latest_day["Previous_High"],
        "Previous_Low": latest_day["Previous_Low"]
    }])
    
    # --- predict the next day's High and Low
    predicted_high = high_model_lr.predict(latest_data)[0]
    predicted_low = low_model_lr.predict(latest_data)[0]
    
    # --- Predict on the last 30 days for graph
    last_30 = model_data.tail(30)
    last_30_features = last_30[["Open", "High", "Low", "Close", "Previous_Close", "Previous_High", "Previous_Low"]]
    
    return {
        "success": True,
        "date": latest_date,
        "predicted_high": predicted_high,
        "predicted_low": predicted_low,
        "high_mae": high_mae_lr,
        "low_mae": low_mae_lr,
        "historical_dates": last_30["Date"].tolist(),
        "actual_highs": last_30["Next_high"].tolist(),
        "actual_lows": last_30["Next_low"].tolist(),
        "predicted_highs_hist": high_model_lr.predict(last_30_features).tolist(),
        "predicted_lows_hist": low_model_lr.predict(last_30_features).tolist()
    }

# ---------------- TKINTER GUI ----------------

def start_gui():
    # --- Fetch NSE symbols ---
    try:
        url_nse = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers_nse = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
        res_nse = requests.get(url_nse, headers=headers_nse, timeout=5)
        import io
        df_nse = pd.read_csv(io.StringIO(res_nse.text))
        all_stocks_display = (df_nse['SYMBOL'] + " - " + df_nse['NAME OF COMPANY']).tolist()
    except Exception:
        all_stocks_display = [
            "GMRAIRPORT - GMR Airports Infrastructure Limited",
            "RELIANCE - Reliance Industries Limited",
            "TCS - Tata Consultancy Services Limited",
            "INFY - Infosys Limited",
            "HDFCBANK - HDFC Bank Limited",
            "SBIN - State Bank of India",
            "ICICIBANK - ICICI Bank Limited"
        ]

    root = tk.Tk()
    root.title("Stock Price Predictor")
    root.geometry("1000x800")
    
    # ---------------- TITLE ----------------
    title_label = tk.Label(
        root,
        text="STOCK PRICE PREDICTOR",
        font=("Arial", 22, "bold")
    )
    title_label.pack(pady=15)
    
    # ---------------- STOCK SELECTION ----------------
    selection_frame = tk.Frame(root)
    selection_frame.pack(pady=5)
    
    dropdown_label = tk.Label(
        selection_frame,
        text="Enter or Select Stock Symbol (e.g. RELIANCE, TCS):",
        font=("Arial", 12)
    )
    dropdown_label.pack()
    
    stock_var = tk.StringVar()
    stock_combo = ttk.Combobox(
        selection_frame,
        textvariable=stock_var,
        values=all_stocks_display,
        font=("Arial", 14),
        width=40
    )
    stock_combo.pack(pady=5)
    if all_stocks_display:
        stock_combo.set(all_stocks_display[0])
        
    def check_input(event):
        if event.keysym in ('Up', 'Down', 'Left', 'Right', 'Return', 'Tab'):
            return
        
        value = event.widget.get()
        if value == '':
            stock_combo['values'] = all_stocks_display
        else:
            data = [item for item in all_stocks_display if value.lower() in item.lower()]
            stock_combo['values'] = data
            
    stock_combo.bind('<KeyRelease>', check_input)
    
    # ---------------- PREDICT BUTTON ----------------
    predict_button = tk.Button(
        root,
        text="PREDICT",
        font=("Arial", 14, "bold"),
        command=lambda: predict()
    )
    predict_button.pack(pady=5)

    # ---------------- LATEST DATE ----------------
    day_label = tk.Label(
        root,
        text="Latest Trading Day: N/A",
        font=("Arial", 12)
    )
    day_label.pack(pady=5)
    
    # ---------------- RESULT & ACCURACY LABELS ----------------
    result_label = tk.Label(
        root,
        text="",
        font=("Arial", 14)
    )
    result_label.pack(pady=5)
    
    accuracy_label = tk.Label(
        root,
        text="",
        font=("Arial", 11)
    )
    accuracy_label.pack(pady=5)
    
    # ---------------- GRAPH FRAME ----------------
    graph_frame = tk.Frame(root)
    graph_frame.pack(pady=5, fill=tk.BOTH, expand=True)
    canvas_widget = None
    
    # ---------------- BUTTON FUNCTION ----------------
    def predict():
        nonlocal canvas_widget
        selection = stock_var.get().strip()
        if not selection:
            messagebox.showerror("Error", "Please enter a stock symbol")
            return
            
        if " - " in selection:
            symbol = selection.split(" - ")[0].upper().replace(" ", "")
        else:
            symbol = selection.upper().replace(" ", "")
            
        result_label.config(text="Fetching data and training model...\nPlease wait.", fg="blue")
        accuracy_label.config(text="")
        day_label.config(text="Latest Trading Day: N/A")
        
        if canvas_widget:
            canvas_widget.get_tk_widget().destroy()
            canvas_widget = None
            
        # Force full update to show loading message before blocking and clear old artifacts
        root.update()
        
        res = fetch_and_predict(symbol)
        
        if "error" in res:
            result_label.config(text="", fg="black")
            messagebox.showerror("Error", res["error"])
        else:
            date_str = res["date"].strftime('%d-%b-%Y')
            day_label.config(text=f"Latest Trading Day: {date_str}")
            
            result_label.config(
                text=f"Predicted High: ₹{res['predicted_high']:.2f}\n"
                     f"Predicted Low: ₹{res['predicted_low']:.2f}",
                fg="green"
            )
            
            accuracy_label.config(
                text=f"Model Accuracy (MAE):\n"
                     f"High MAE: ₹{res['high_mae']:.2f}\n"
                     f"Low MAE: ₹{res['low_mae']:.2f}"
            )
            
            # --- Draw Graph ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Formatted dates for x-axis
            dates = [d.strftime('%m-%d') for d in res["historical_dates"]]
            next_date = (res["date"] + datetime.timedelta(days=1)).strftime('%m-%d')
            plot_dates = dates + [next_date]
            
            actual_highs = res["actual_highs"]
            pred_highs_hist = res["predicted_highs_hist"]
            
            actual_lows = res["actual_lows"]
            pred_lows_hist = res["predicted_lows_hist"]
            
            # 1. Plot Highs
            ax1.plot(dates, actual_highs, label="Actual High", color="blue", marker="o", markersize=4)
            # Combine historical predictions + tomorrow's prediction
            ax1.plot(plot_dates, pred_highs_hist + [res["predicted_high"]], label="Predicted High", color="cyan", linestyle="--", marker="X", markersize=5)
            ax1.set_title(f"{symbol} - High Price Predictions")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price (₹)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            # show fewer ticks to avoid clutter
            ax1.set_xticks(plot_dates[::5] + [plot_dates[-1]])
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Plot Lows
            ax2.plot(dates, actual_lows, label="Actual Low", color="red", marker="o", markersize=4)
            # Combine historical predictions + tomorrow's prediction
            ax2.plot(plot_dates, pred_lows_hist + [res["predicted_low"]], label="Predicted Low", color="orange", linestyle="--", marker="X", markersize=5)
            ax2.set_title(f"{symbol} - Low Price Predictions")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price (₹)")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_xticks(plot_dates[::5] + [plot_dates[-1]])
            ax2.tick_params(axis='x', rotation=45)
            
            fig.tight_layout()
            
            canvas_widget = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
    # ---------------- RUN GUI ----------------
    root.mainloop()

if __name__ == "__main__":
    start_gui()