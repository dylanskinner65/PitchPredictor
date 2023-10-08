import streamlit as st
import pandas as pd

# Load your pitch data
pitch_data = pd.read_parquet('pitches_data.parquet')

# Create Streamlit app
st.title("Pitch Visualization")

# Add a sidebar for filtering options, if needed
st.sidebar.header("Filter Data")
# Add filter widgets here, e.g., for pitcher's name, hitter's name, pitch type, etc.

# Display the selected data
st.subheader("Selected Pitch Data")

# Use st.dataframe to display the filtered data
st.dataframe(pitch_data)

# You can also use st.map for location-based visualization if you have coordinates

# You can create a scatter plot or other charts to visualize pitch location
# You may need to install additional libraries like Matplotlib or Plotly for this

# You can also add interactive components like dropdowns or sliders to filter the data
# based on user input.

# Example: Filter data based on pitcher's name
selected_pitcher = st.selectbox("Select Pitcher", pitch_data['full_name'].unique())
filtered_data = pitch_data[pitch_data['full_name'] == selected_pitcher]

# Display the filtered data
st.dataframe(filtered_data)

