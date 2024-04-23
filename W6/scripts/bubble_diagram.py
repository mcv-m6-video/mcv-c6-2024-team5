import pandas as pd
import plotly.express as px

# Load your data
data = pd.read_csv('convertcsv_patch.csv')
data['Test accuracy (%)'] = data['Test accuracy (%)'].str.replace('%', '').astype(float)
data['Test accuracy (%)'] /= 100  # Convert to a fraction for plotting
data['Parameters (M)'] = data['Parameters (M)'].astype(float)  # Ensure this is correctly scaled as needed

# Aggregate data to find the best test accuracy for each model
grouped_data = data.groupby('Name').agg({
    'Test accuracy (%)': 'max',
    'Parameters (M)': 'first',
    'FLOPs (G)': 'first',
    'Color': 'first'  # Ensure colors are in a format Plotly can understand (e.g., HEX, RGB)
}).reset_index()

# Create the Bubble plot
fig = px.scatter(
    grouped_data,
    x='FLOPs (G)',
    y='Test accuracy (%)',
    size='Parameters (M)',
    hover_name='Name',
    text='Name',  # Displays the model name on the bubbles
    title='Test Accuracy vs GFLOPs with Bubble Size as Parameters (M)',
    labels={'FLOPs (G)': 'GFLOPs', 'Test accuracy (%)': 'Test Accuracy', 'Parameters (M)': 'Parameters (M)'},
    size_max=60
)

# Set colors manually from 'Color' column if not applying automatically
fig.update_traces(marker=dict(color=grouped_data['Color']))

# Add 'Parameters' text annotation at a specified position
fig.add_annotation(
    x=4.85, y=0.59,
    text='Parameters',
    showarrow=False,
    font=dict(size=12)
)

# Adjust layout
fig.update_layout(
    showlegend=False  # Disable the legend if it is not needed
)

# Display the figure
# fig.show()

# Optionally save the plot as a PNG image
fig.write_image('bubble_plot.png')
