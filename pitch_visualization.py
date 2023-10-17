import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Load your pitch data
pitch_data = pd.read_parquet('pitches_data.parquet')
player_ids = pd.read_parquet('dashboard_notebooks/player_ids_scaled.parquet')
inning_scaled = pd.read_parquet('dashboard_notebooks/inning_type_scaled.parquet')
pitch_encoded = pd.read_parquet('dashboard_notebooks/pitch_type_encoding.parquet')

# Define my model architecture for the pitch prediction model
class PenalizedTanH(nn.Module):
    def __init__(self):
        super(PenalizedTanH, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, torch.tanh(x), 0.25*torch.tanh(x))

# First, define the hyperparameters
input_size = 31  # Input size (e.g., number of features)
hidden_size = 64  # Size of the hidden layer(s)
output_size = 19  # Output size (e.g., number of classes)
learning_rate = 0.001

# Define the neural network architecture
class PitchDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PitchDNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                PenalizedTanH(),
                                nn.Linear(hidden_size, hidden_size*2),
                                PenalizedTanH(),
                                nn.Linear(hidden_size*2, hidden_size*4),
                                PenalizedTanH(),
                                nn.Linear(hidden_size*4, hidden_size*4),
                                PenalizedTanH(),
                                nn.Linear(hidden_size*4, hidden_size*2),
                                PenalizedTanH(),
                                nn.Linear(hidden_size*2, hidden_size),
                                PenalizedTanH(),
                                nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)

# Create an instance of the model
model = PitchDNN(input_size, hidden_size, output_size)
checkpoint = torch.load('dashboard_notebooks/pitch_dnn_L4.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])


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

    # plt.suptitle(f"Pitch Visualization for {selected_pitcher} against ")
    plt.tight_layout()

    return fig

def get_data(data_array):
    # Get the pitcher's id
    pitcher_name = data_array[0]
    pitcher_id = player_ids[player_ids['full_name'] == pitcher_name]['scaled_id'].values
    # Get the batter's id
    batter_id = data_array[1]
    batter_id = player_ids[player_ids['id'] == batter_id]['scaled_id'].values
    # Get the pitcher's score
    pitcher_score = int(data_array[10])/25
    # Get the batter's score
    batter_score = int(data_array[2])/25
    # Get the inning and scale it
    inning = data_array[9]
    inning = inning_scaled[inning_scaled['inning'] == inning]['inning_scale'].values
    # Get the top or bottom of the inning
    top = 1.0 if data_array[13] == 'Top' else 0.0
    # Get the number of outs
    outs = int(data_array[5])/2
    # Get the number of balls
    balls = int(data_array[3])/3
    # Get the number of strikes
    strikes = int(data_array[4])/2
    # Get the pitcher's hand
    pitcher_hand = 1.0 if data_array[11] == 'R' else 0.0
    # Get the batter's hand
    batter_hand = 1.0 if data_array[12] == 'R' else 0.0
    # Get all of the pitch counts together in one array and normalize them
    pitch_counts = data_array[14:31].astype(np.float16)
    pitch_counts = pitch_counts/np.sum(pitch_counts)
    # Bases
    first_base = 1.0 if data_array[6] == 'Occupied' else 0.0
    second_base = 1.0 if data_array[7] == 'Occupied' else 0.0
    third_base = 1.0 if data_array[8] == 'Occupied' else 0.0

    # Return all of the information as one numpy array in order of how it was passed in
    return np.array([pitcher_id, batter_id, batter_score, balls, strikes, outs, first_base, second_base,
                     third_base, inning, pitcher_score, pitcher_hand, batter_hand, top, *pitch_counts])

# Define a function to predict the next pitch
def predict_next_pitch(data_array):
    # Get the data
    data_array = get_data(data_array)
    # Create a tensor from the data
    print(data_array)
    data_tensor = torch.from_numpy(data_array).float()
    # Get the model's prediction
    with torch.no_grad():
        prediction = model(data_tensor)
        # Get the index of the maximum value
        prediction = torch.argmax(prediction)
        percentage = torch.max(prediction)
    # Get the pitch type
    pitch_type = pitch_encoded[pitch_encoded['encoded_pitch_type'] == prediction]['pitch_type'].values[0]

    return pitch_type, percentage


# Create Streamlit app
st.title("Pitch Visualization")
# Add a sidebar for filtering options

st.subheader("Pitcher/Batter Data to Visualize")
# Example: Filter data based on pitcher's name
selected_pitcher = st.selectbox("Select Pitcher", pitch_data['pitcher_name'].unique())

# Filter data based on batter's name
selected_batter = st.selectbox("Select Batter", pitch_data['batter_name'].unique())

# Generate and display the pitch visualizations
st.subheader(f"Pitch Visualizations for {selected_pitcher} Against {selected_batter}")
pitch_fig = generate_pitch_visualizations(pitch_data, selected_pitcher, selected_batter)
st.pyplot(pitch_fig)

# Display the selected pitcher's data
# st.subheader(f"Selected Pitch Data for {selected_pitcher}")
# st.write(pitch_data[(pitch_data['pitcher_name'] == selected_pitcher) & (pitch_data['batter_name'] == selected_batter)].head(10))

st.sidebar.header(f'Game Information to Predict Next Pitch Between {selected_pitcher} and {selected_batter}')

# Information to get:
# Pitcher Score, Batter Score, Inning, Outs, Balls, Strikes, Pitcher Hand, Batter Hand
potential_pitches = ['CH', 'CU', 'FC', 'FF', 'FS',
                    'FT', 'IN', 'KC', 'KN', 'PO',
                    'SC', 'SI', 'SL', 'UN', 'Unknown',
                    'EP', 'FA']
pitcher_score = st.sidebar.number_input('Pitcher Score', min_value=0, max_value=40, value=0)
batter_score = st.sidebar.number_input('Batter Score', min_value=0, max_value=40, value=0)
inning = st.sidebar.number_input('Inning', min_value=1, max_value=30, value=1)
top = st.sidebar.selectbox('Top or Bottom of Inning', ['Top', 'Bottom'])
outs = st.sidebar.number_input('Outs', min_value=0, max_value=2, value=0)
balls = st.sidebar.number_input('Balls', min_value=0, max_value=3, value=0)
strikes = st.sidebar.number_input('Strikes', min_value=0, max_value=2, value=0)
pitcher_hand = st.sidebar.selectbox('Pitcher Hand', ['L', 'R'])
batter_hand = st.sidebar.selectbox('Batter Hand', ['L', 'R'])
first_base = st.sidebar.selectbox('First Base', ['None', 'Occupied'])
second_base = st.sidebar.selectbox('Second Base', ['None', 'Occupied'])
third_base = st.sidebar.selectbox('Third Base', ['None', 'Occupied'])
num_ch = st.sidebar.number_input('Number of Changeups', min_value=0, max_value=100, value=0)
num_cu = st.sidebar.number_input('Number of Curveballs', min_value=0, max_value=100, value=0)
num_fc = st.sidebar.number_input('Number of Cutters', min_value=0, max_value=100, value=0)
num_ff = st.sidebar.number_input('Number of Four-Seam Fastballs', min_value=0, max_value=100, value=0)
num_fs = st.sidebar.number_input('Number of Splitter Fastballs', min_value=0, max_value=100, value=0)
num_ft = st.sidebar.number_input('Number of Two-Seam Fastballs', min_value=0, max_value=100, value=0)
num_in = st.sidebar.number_input('Number of Intentional Balls', min_value=0, max_value=100, value=0)
num_kc = st.sidebar.number_input('Number of Knuckle Curves', min_value=0, max_value=100, value=0)
num_kn = st.sidebar.number_input('Number of Knuckleballs', min_value=0, max_value=100, value=0)
num_po = st.sidebar.number_input('Number of Pitchouts', min_value=0, max_value=100, value=0)
num_sc = st.sidebar.number_input('Number of Screwballs', min_value=0, max_value=100, value=0)
num_si = st.sidebar.number_input('Number of Sinker Fastballs', min_value=0, max_value=100, value=0)
num_sl = st.sidebar.number_input('Number of Sliders', min_value=0, max_value=100, value=0)
num_un = st.sidebar.number_input('Number of Unknown Pitches', min_value=0, max_value=100, value=0)
num_ep = st.sidebar.number_input('Number of Eephus Pitches', min_value=0, max_value=100, value=0)

# Create a numpy array to hold the data
pitch_data = np.array([selected_pitcher, selected_batter, batter_score, balls, strikes, outs, first_base, second_base, 
                       third_base, inning, pitcher_score, pitcher_hand, batter_hand, top, num_ch, num_cu, num_fc,
                       num_ff, num_fs, num_ft, num_in, num_kc, num_kn, num_po, num_sc, num_si, num_sl, num_un, 0, num_ep,
                       0])

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(pitch_data)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# Send it through the model
pitch_type, percentage = predict_next_pitch(pitch_data)

# Display the prediction
st.subheader(f"Next Pitch Prediction for {selected_pitcher} Against {selected_batter}")
st.write(f"The model predicts that {pitch_type} will be the next pitch with {percentage*100}% certainty.")


