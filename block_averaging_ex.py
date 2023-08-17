import pandas as pd
import numpy as np
from scipy import stats, optimize
import plotly.express as px

# Function to load and preprocess data
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path, header=None)
    if data.iloc[0, 0] == 'atom_type':
        data.columns = ['Atom_Type'] + [str(i) for i in range(data.shape[1] - 1)]
        data = data.set_index('Atom_Type').drop('atom_type')
    else:
        data.columns = ['Atom_Type'] + [str(i) for i in range(data.shape[1] - 1)]
        data = data.set_index('Atom_Type')
    return data

# Block averaging function
def block_averaging(corr, max_block_size=None):
    sems = list()
    corr_array = corr.to_numpy()  # Convert to NumPy array
    max_block_size = max_block_size or len(corr_array) // 4
    for blocksize in range(1, max_block_size + 1):
        y_ = corr_array[:len(corr_array) - (len(corr_array) % blocksize)].reshape(-1, blocksize).mean(1)
        sems.append(stats.sem(y_))
    return sems

# Arctan function
def arctan_function(x, a, b, c):
    return a * np.arctan(b * (x - c))

# Fitting arctan curve
def fit_arctan(blocked_SEMs):
    popt, _ = optimize.curve_fit(arctan_function, np.arange(len(blocked_SEMs)), np.array(blocked_SEMs), maxfev=20000)
    return popt

# Function to create a plot and handle clicks
def block_averaging_plot(file_path):
    data = load_and_preprocess(file_path)
    atom_types_list = ['CA', 'CB', 'H', 'HA', 'HA2', 'HA3', 'N']
    for atom_type_name in atom_types_list:
        timeseries = data.loc[atom_type_name].dropna().astype(float)
        if len(timeseries) == 0:
            continue
        blocked_sems = block_averaging(timeseries)
        popt = fit_arctan(blocked_sems)
        x_values = np.arange(len(blocked_sems))
        y_values_arctan = arctan_function(x_values, *popt)

        # Create a DataFrame for Plotly Express
        df_plot = pd.DataFrame({
            'Block Size': x_values,
            'SEM': blocked_sems,
            'Type': ['Blocked SEM'] * len(blocked_sems)
        })
        df_arctan = pd.DataFrame({
            'Block Size': x_values,
            'SEM': y_values_arctan,
            'Type': ['Fitted arctan curve'] * len(blocked_sems)
        })
        df = pd.concat([df_plot, df_arctan])

        # Create the initial plot
        fig = px.scatter(df, x='Block Size', y='SEM', color='Type', title=f'Block Size vs SEM (Atom: {atom_type_name})')
        fig.update_traces(mode='markers+lines', line=dict(dash='dash'))
        fig.show()

    # Prompt user for the selected block size
    # selected_block_size = int(input("Please enter the selected block size: "))
    # print(f"Selected block size for all atom types: {selected_block_size}")

# File path to the RMSE data
file_path = '/content/file/it2-da2_vs_bmr27931_RMSE.csv'  # Replace with the path to your RMSE file

# Plot with clickable points
block_averaging_plot(file_path)
