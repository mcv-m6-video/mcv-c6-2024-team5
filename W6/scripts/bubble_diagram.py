import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Import for more control over the figures

# Load your data
data = pd.read_csv('convertcsv.csv')
data['Test accuracy (%)'] = data['Test accuracy (%)'].str.replace('%', '').astype(float)
data['Test accuracy (%)'] /= 100  # Convert to a fraction for plotting

# Aggregate data to find the best test accuracy for each architecture
grouped_data = data.groupby('Architecture').agg({
    'Test accuracy (%)': 'max',
    'Parameters (M)': 'first',
    'FLOPs (G)': 'first'
}).reset_index()

# Create the Bubble plot with color differentiation and labels
fig = px.scatter(
    grouped_data,
    x='FLOPs (G)',
    y='Test accuracy (%)',
    size='Parameters (M)',
    color='Architecture',  # This assigns a unique color to each architecture
    hover_name='Architecture',
    text='Architecture',  # Displays the architecture name on the bubbles
    title='Bubble Plot: Test Accuracy vs GFLOPs with Bubble Size as Parameters',
    labels={'FLOPs (G)': 'GFLOPs', 'Test accuracy (%)': 'Test Accuracy', 'Parameters (M)': 'Parameters (M)'},
    size_max=60
)

# Manually add size annotations
sizes = [1, 10, 20]  # Example sizes: adjust these based on your actual parameter sizes
for size in sizes:
    fig.add_trace(go.Scatter(
        x=[max(grouped_data['FLOPs (G)']) * 1.1],  # Position the legend to the right of the data
        y=[max(grouped_data['Test accuracy (%)']) * (0.8 - 0.1 * sizes.index(size))],  # Position vertically
        text=[f'{size}M parameters'],
        mode='markers+text',
        marker=dict(size=size, color='gray'),
        showlegend=False,
        textposition="middle right"
    ))

# Adjust layout
fig.update_layout(
    xaxis_range=[0, max(grouped_data['FLOPs (G)']) * 1.2],  # Extend x-axis to fit annotations
    showlegend=True,
    legend_title_text='Architecture'
)

# Show the figure
fig.show()

# Save the plot as a PNG image
fig.write_image('bubble_plot.png')
