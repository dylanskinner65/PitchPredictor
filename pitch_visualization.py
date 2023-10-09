import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your pitch data
pitch_data = pd.read_parquet('pitches_data.parquet')

# Define a function to generate pitch visualizations
def generate_pitch_visualizations(pitch_data, selected_pitcher, selected_batter):
    # Create a scatter plot with colors based on pitch type
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=100)
    
    # Get subset data based on desired pitcher.
    pitch_data_p = pitch_data[(pitch_data['pitcher_name'] == selected_pitcher) & (pitch_data['batter_name'] == selected_batter)]

    # Create a colormap mapping pitch types to colors
    pitch_types = pitch_data_p['pitch_type'].unique()
    colors = plt.cm.get_cmap('tab20', len(pitch_types))  # You can choose a different colormap if desired
    color_map = {pitch_type: colors(i) for i, pitch_type in enumerate(pitch_types)}

    # Iterate through pitch types and plot each type with its corresponding color
    for pitch_type, color in color_map.items():
        subset = pitch_data_p[pitch_data_p['pitch_type'] == pitch_type]
        ax[0].scatter(subset['px'], subset['pz'], label=pitch_type, color=color, alpha=0.5)

    ax[0].set_xlabel('Horizontal Location (px)')
    ax[0].set_ylabel('Vertical Location (pz)')
    ax[0].set_title(f'Pitch Location by Type')
    ax[0].legend(title='Pitch Type', loc='best')
    ax[0].grid(True)

    # Create a 2D histogram and heatmap
    hist, xedges, yedges = np.histogram2d(pitch_data_p['px'], pitch_data_p['pz'], bins=(50, 50), range=[[-3, 3], [0, 5]])

    img = ax[1].imshow(hist.T, extent=[-3, 3, 0, 5], origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')
    fig.colorbar(img, ax=ax[1], label='Pitch Frequency')
    ax[1].set_xlabel('Horizontal Location ($px$)')
    ax[1].set_ylabel('Vertical Location ($pz$)')
    ax[1].set_title(f'Pitch Frequency Heatmap')
    ax[1].grid(False)

    plt.suptitle(f"Pitch Visualization for {selected_pitcher}")
    plt.tight_layout()

    return fig

# Create Streamlit app
st.title("Pitch Visualization")

# Add a sidebar for filtering options
st.sidebar.header("Filter Data")

# Example: Filter data based on pitcher's name
selected_pitcher = st.sidebar.selectbox("Select Pitcher", pitch_data['pitcher_name'].unique())

# Filter data based on batter's name
selected_batter = st.sidebar.selectbox("Select Batter", pitch_data['batter_name'].unique())

# Generate and display the pitch visualizations
st.subheader(f"Pitch Visualizations for {selected_pitcher} Against {selected_batter}")
pitch_fig = generate_pitch_visualizations(pitch_data, selected_pitcher, selected_batter)
st.pyplot(pitch_fig)

# Display the selected pitcher's data
st.subheader(f"Selected Pitch Data for {selected_pitcher}")
st.write(pitch_data[(pitch_data['pitcher_name'] == selected_pitcher) & (pitch_data['batter_name'] == selected_batter)].head(10))