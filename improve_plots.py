import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile, BadZipFile
import os
from scipy.optimize import curve_fit

# --- إعدادات التصميم الاحترافي للرسوم البيانية ---
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'lines.linewidth': 2.5
})

# --- تحميل البيانات ---
def load_data(ticker_paths):
    ticker_dfs = {}
    for ticker, zip_path in ticker_paths.items():
        extracted_path = f'data/{ticker}/extracted'
        
        # Try to load from extracted CSV first (if already extracted)
        if os.path.exists(extracted_path):
            csv_files = [f for f in os.listdir(extracted_path) if f.endswith(".csv")]
            if csv_files:
                try:
                    df = pd.read_csv(os.path.join(extracted_path, csv_files[0]))
                    if {'price', 'volume'}.issubset(df.columns):
                        ticker_dfs[ticker] = df
                        print(f"Loaded {ticker} from extracted CSV: {os.path.join(extracted_path, csv_files[0])}")
                        continue # Move to next ticker if data found
                except Exception as e:
                    print(f"Error reading extracted CSV for {ticker}: {e}")
        
        # If not loaded, try to extract from zip and then load
        if os.path.exists(zip_path):
            try:
                with ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_path)
                    csv_files = [f for f in os.listdir(extracted_path) if f.endswith(".csv")]
                    if csv_files:
                        df = pd.read_csv(os.path.join(extracted_path, csv_files[0]))
                        if {'price', 'volume'}.issubset(df.columns):
                            ticker_dfs[ticker] = df
                            print(f"Loaded {ticker} from zip extraction: {os.path.join(extracted_path, csv_files[0])}")
                            continue # Move to next ticker if data found
            except BadZipFile:
                print(f"Warning: {zip_path} is not a valid zip file. Trying direct CSV.")
            except Exception as e:
                print(f"Error processing {ticker} zip: {e}")
        
        # Fallback: Check for direct CSV in the ticker's main data directory (e.g., for CRWV, SOUN)
        # This handles cases where the data is a CSV directly in the ticker's folder, not zipped.
        direct_csv_path = os.path.join(os.path.dirname(zip_path), f'{ticker}_2025-04-03 00:00:00+00:00.csv') # Specific to CRWV/SOUN naming
        if os.path.exists(direct_csv_path):
            try:
                df = pd.read_csv(direct_csv_path)
                if {'price', 'volume'}.issubset(df.columns):
                    ticker_dfs[ticker] = df
                    print(f"Loaded {ticker} from direct CSV: {direct_csv_path}")
            except Exception as e:
                print(f"Error reading direct CSV for {ticker}: {e}")
        else:
            print(f"No data found for {ticker} at {zip_path} or {extracted_path} or {direct_csv_path}")

    return ticker_dfs

# --- نمذجة التأثير ---
def linear_model(x, beta):
    return beta * x

def nonlinear_model(x, alpha, p):
    return alpha * np.power(x, p)

# --- إنشاء الرسوم البيانية المحسنة ---
def create_improved_plots(ticker_dfs, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ticker, df in ticker_dfs.items():
        df = df.copy()
        df['impact'] = df['price'].diff().fillna(0)
        x_data = df['volume'].values
        y_data = df['impact'].values

        try:
            # Filter out zero volume data points for curve fitting if necessary
            non_zero_volume_indices = x_data != 0
            x_data_filtered = x_data[non_zero_volume_indices]
            y_data_filtered = y_data[non_zero_volume_indices]

            # Only attempt fit if there's enough data after filtering
            if len(x_data_filtered) > 0:
                popt_nonlin, _ = curve_fit(nonlinear_model, x_data_filtered, y_data_filtered, bounds=([0, 0], [np.inf, 2]))
                alpha, p = popt_nonlin
            else:
                print(f"Not enough non-zero volume data for {ticker} to fit model.")
                continue

            plt.figure()
            plt.scatter(x_data, y_data, alpha=0.4, label='Observed Data', color='gray', s=50)
            
            # Ensure x_fit covers the range of observed data for plotting the model
            x_fit = np.linspace(x_data_filtered.min(), x_data_filtered.max(), 400)
            y_fit = nonlinear_model(x_fit, alpha, p)
            
            plt.plot(x_fit, y_fit, label=f'Power-Law Fit (p={p:.2f})', color='#ff5722')
            
            plt.title(f'Temporary Market Impact for {ticker}')
            plt.xlabel('Order Volume')
            plt.ylabel('Price Impact (Slippage)')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'{ticker}_impact_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {save_path}")

        except Exception as e:
            print(f"Could not generate plot for {ticker}: {e}")

# --- التنفيذ ---
if __name__ == "__main__":
    ticker_paths = {
        'CRWV': 'data/CRWV/CRWV.zip',
        'FROG': 'data/FROG/FROG.zip',
        'SOUN': 'data/SOUN/SOUN.zip'
    }
    ticker_data = load_data(ticker_paths)
    create_improved_plots(ticker_data)


