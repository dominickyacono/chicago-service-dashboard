import geopandas
import pandas as pd
from datetime import date
import numpy as np

from bokeh.models import (
    GeoJSONDataSource, HoverTool, LinearColorMapper, ColorBar, Div,
    CustomJS, Patches, ColumnDataSource, NumeralTickFormatter
)
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.events import Tap
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from bokeh.tile_providers import CARTODBPOSITRON_RETINA


# --- Data Loading ---
# It's good practice to wrap file loading in try-except blocks
try:
    # This GeoJSON should contain the boundaries for Chicago Community Areas
    chicago_areas = geopandas.read_file("Boundaries - Community Areas_20250802.geojson")
except Exception as e:
    print(f"Error loading GeoJSON file: {e}")
    # Create a dummy dataframe to allow the script to run without the file
    chicago_areas = geopandas.GeoDataFrame({
        'community': ['DUMMY'],
        'geometry': [None],
        'area_num_1': ['0']
    })

try:
    # This file should contain predictions AND the historical data for the trend graph
    # Example columns: community_area, count_prediction, count_lag_1, count_lag_2, count_lag_3
    predictions_df = pd.read_csv("test_latest_counts_graphed_with_predictions.csv")
    if 'historical_average' not in predictions_df.columns:
        # If no average exists, use yesterday's count (lag_1) as a proxy.
        predictions_df['historical_average'] = predictions_df['count_lag_1']

except Exception as e:
    print(f"Error loading predictions CSV file: {e}")
    # Create a dummy dataframe if the file is not found
    predictions_df = pd.DataFrame({
        'community_area': [],
        'count_prediction': [],
        'count_lag_1': [],
        'count_lag_2': [],
        'count_lag_3': [],
        'historical_average': []
    })


try:
    # This file should contain top service types for each community area
    # Example columns: community_area_number, top_3_services
    top_services_df = pd.read_csv("top_services_list.csv")
except Exception as e:
    print(f"Error loading top services CSV file: {e}")
    # Create a dummy dataframe if the file is not found
    top_services_df = pd.DataFrame({
        'community_area_number': [],
        'top_3_services': []
    })

# --- Data Preparation and Merging (replace the original section with this) ---
chicago_areas['area_num_1'] = chicago_areas['area_num_1'].astype(int)
predictions_df['community_area'] = predictions_df['community_area'].astype(int)
top_services_df['community_area_number'] = top_services_df['community_area_number'].astype(int)

# Use an 'inner' merge to ensure we only keep geometries with corresponding prediction data.
chicago_data = chicago_areas.merge(predictions_df, left_on='area_num_1', right_on='community_area', how='inner')

# Now, perform a 'left' merge to add the top services data.
# A left merge ensures we keep all community areas, even if they don't have a service list.
chicago_data = chicago_data.merge(top_services_df, left_on='area_num_1', right_on='community_area_number', how='left')

# Fill any missing service lists with a placeholder text.
chicago_data['top_3_services'] = chicago_data['top_3_services'].fillna('No data available')

# Ensure correct data types for all relevant columns.
for col in ['count_prediction', 'count_lag_1', 'count_lag_2', 'count_lag_3', 'historical_average']:
    chicago_data[col] = chicago_data[col].astype(int)

# Calculate the increase over the historical average.
chicago_data['increase'] = chicago_data['count_prediction'] - chicago_data['historical_average']

# Reproject to Web Mercator (EPSG:3857) to match the basemap tiles
chicago_data = chicago_data.to_crs("EPSG:3857")

# This line MUST come AFTER all data is merged and prepared.
geosource = GeoJSONDataSource(geojson=chicago_data.to_json())

# --- THEME & STYLING ---
curdoc().theme = 'light_minimal'

# --- UI ELEMENTS ---
formatted_date = date.today().strftime("%B %d, %Y")

# Use a more modern font stack and even larger font sizes for a "flashy" look.
header = Div(text=f"""
    <h1 style="font-family: system-ui, -apple-system, sans-serif; color: #2C3E50; margin-bottom: 0; font-size: 2.8em; font-weight: 600;">Chicago Service Request Predictions</h1>
    <p style="font-family: system-ui, -apple-system, sans-serif; color: #566573; font-size: 1.4em; margin-top: 5px;">Predicted 311 service requests for {formatted_date}</p>
""", width=1200)

# Style the description as a modern UI card with larger text and a more prominent shadow.
description_div = Div(text="""
    <div style='
        font-family: system-ui, -apple-system, sans-serif; 
        color: #34495e; 
        font-size: 1.2em; 
        margin-top: 20px; 
        max-width: 1200px;
        padding: 22px;
        background-color: #fdfdfd;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    '>
    This dashboard highlights predicted 311 service request hotspots. The map is color-coded by predicted volume. 
    The <b>Top 5 Hotspots</b> list shows areas with the largest predicted increase compared to yesterday. 
    Click any area on the map to see its specific trend graph.
    </div>
""", width=1200)

# Create the Top 5 Hotspots list.
hotspots_df = chicago_data.nlargest(5, 'increase')

# FIX: Create a new GeoJSONDataSource specifically for the hotspot outlines.
hotspot_geosource = GeoJSONDataSource(geojson=hotspots_df.to_json())

hotspot_list_html = "<h3 style='margin-top:0; color: #2c3e50;'>Top 5 Hotspots</h3><ol style='padding-left: 20px; margin:0;'>"
# Renamed the 'row' variable to 'hotspot_row' to avoid overwriting the Bokeh 'row' function.
for index, hotspot_row in hotspots_df.iterrows():
    increase_text = f"+{hotspot_row['increase']}"
    hotspot_list_html += f"<li style='margin-bottom: 8px; font-size: 1.1em;'>{hotspot_row['community']} <span style='color: #c0392b; font-weight: 600;'>({increase_text})</span></li>"
hotspot_list_html += "</ol>"

hotspot_div = Div(text=f"""
    <div style='
        font-family: system-ui, -apple-system, sans-serif;
        padding: 20px;
        background-color: #fdfdfd;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        height: 250px; 
    '>
    {hotspot_list_html}
    </div>
""", width=190)


# Create a Div to display the top service types for a selected area.
service_list_div = Div(text="""
    <div style='
        font-family: system-ui, -apple-system, sans-serif;
        padding: 20px;
        background-color: #fdfdfd;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        height: 250px; /* CHANGED from 180px */
    '>
    <h3 style='margin-top:0; color: #2c3e50;'>Top 3 Service Types</h3>
    <p style='color: #566573;'>Click a community area on the map to see details.</p>
    </div>
""", width=190)

# --- MAP PLOTTING ---
# Improve color mapping by clipping the max value at the 95th percentile.
p95 = chicago_data.count_prediction.quantile(0.95)
color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=p95)

map_plot = figure(
    height=600, width=800,
    tools="pan,wheel_zoom,box_zoom,reset,save,tap",
    x_axis_type="mercator", y_axis_type="mercator",
    active_scroll='wheel_zoom',
    x_range=(chicago_data.total_bounds[0], chicago_data.total_bounds[2]),
    y_range=(chicago_data.total_bounds[1], chicago_data.total_bounds[3]),
    outline_line_color=None
)
map_plot.axis.visible = False
map_plot.grid.grid_line_color = None
# OLD CODE
map_plot.add_tile(CARTODBPOSITRON_RETINA)

# Add patches for all community areas (the base layer)
patches = map_plot.patches(
    'xs', 'ys', source=geosource,
    fill_color={'field': 'count_prediction', 'transform': color_mapper},
    line_color='#FFFFFF', line_width=1.0, fill_alpha=0.75
)
# Make selection more prominent
patches.selection_glyph = Patches(fill_alpha=0.9, line_color='#007BFF', line_width=2.5, fill_color={'field': 'count_prediction', 'transform': color_mapper})
patches.nonselection_glyph = Patches(fill_alpha=0.6, line_color='#FFFFFF', line_width=1.0)

# FIX: Add a new renderer specifically for the hotspot outlines.
map_plot.patches(
    'xs', 'ys', source=hotspot_geosource,
    fill_alpha=0,
    line_color='#c0392b',
    line_width=3
)

hover = HoverTool(
    tooltips=[("Area", "@community"), ("Predicted Requests", "@count_prediction{0,0}"), ("Change vs yesterday", "@increase{+0}")],
    renderers=[patches]
)
map_plot.add_tools(hover)

# Move color bar below the map for a cleaner layout
color_bar = ColorBar(
    color_mapper=color_mapper,
    title="Predicted Request Count",
    orientation='horizontal',
    location='bottom_center',
    padding=20,
    major_label_text_font_size='11pt',
    title_text_font_size='13pt',
    title_standoff=10,
    border_line_color=None,
    background_fill_alpha=0
)
map_plot.add_layout(color_bar, 'below')

# --- TREND GRAPH PLOTTING ---
# Calculate citywide aggregate data for the default view.
total_prediction = chicago_data['count_prediction'].sum()
total_lag1 = chicago_data['count_lag_1'].sum()
total_lag2 = chicago_data['count_lag_2'].sum()

aggregate_data = dict(
    x_obs=[0, 1],
    y_obs=[total_lag2, total_lag1],
    x_pred=[1, 2],
    y_pred=[total_lag1, total_prediction]
)

# Initialize the trend source with the aggregate data.
trend_source = ColumnDataSource(data=aggregate_data)

trend_plot = figure(
    height=300, width=400,
    # Set the initial title to reflect the citywide view.
    title="Citywide Trend",
    x_axis_label='Time', y_axis_label='Total Request Count',
    tools="pan,reset,save",
    outline_line_color=None
)

# Add renderers for the trend lines
# Increase line width and circle size for a "flashier" look.
trend_plot.line(x='x_obs', y='y_obs', source=trend_source, legend_label="Observed", line_width=4, color="#34495E")
trend_plot.circle(x='x_obs', y='y_obs', source=trend_source, legend_label="Observed", size=10, color="#34495E")
trend_plot.line(x='x_pred', y='y_pred', source=trend_source, legend_label="Predicted", line_width=4, color="#E74C3C", line_dash="dashed")
trend_plot.circle(x='x_pred', y='y_pred', source=trend_source, legend_label="Predicted", size=10, color="#E74C3C")

# Increase font sizes for better readability.
trend_plot.title.text_font_size = '16pt'
trend_plot.xaxis.axis_label_text_font_size = '13pt'
trend_plot.yaxis.axis_label_text_font_size = '13pt'
trend_plot.xaxis.major_label_text_font_size = '11pt'
trend_plot.yaxis.major_label_text_font_size = '11pt'

trend_plot.legend.location = "top_left"
trend_plot.legend.click_policy = "hide"
trend_plot.legend.background_fill_alpha = 0
trend_plot.legend.border_line_alpha = 0
trend_plot.legend.label_text_font_size = '11pt'
trend_plot.grid.grid_line_alpha = 0.3
trend_plot.yaxis.formatter = NumeralTickFormatter(format="0,0")


# Reverse the ticks and labels for the x-axis to go left-to-right.
trend_plot.xaxis.ticker = [0, 1, 2]
trend_plot.xaxis.major_label_overrides = {
    0: '2 Days Ago',
    1: 'Yesterday',
    2: 'Today (Pred)'
}


# --- INTERACTIVE JAVASCRIPT CALLBACK (MODIFIED) ---
callback = CustomJS(args=dict(
    map_source=geosource,
    trend_source=trend_source,
    trend_plot=trend_plot,
    agg_data=aggregate_data,
    service_div=service_list_div, # Argument for the new Div
), code="""
    const selected_index = map_source.selected.indices[0];
    const data = map_source.data;
    
    const initial_service_html = `
        <div style='
            font-family: system-ui, -apple-system, sans-serif;
            padding: 20px;
            background-color: #fdfdfd;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            height: 250px; /* CHANGED */
        '>
        <h3 style='margin-top:0; color: #2c3e50;'>Top Service Types</h3>
        <p style='color: #566573;'>Click a community area on the map to see details.</p>
        </div>
    `;

    // If nothing is selected, revert to the citywide aggregate view.
    if (selected_index === undefined) {
        trend_source.data = agg_data;
        trend_plot.title.text = "Citywide Trend";
        service_div.text = initial_service_html; // Reset the service div
        map_source.selected.indices = []; // Ensure selection is cleared visually
        trend_source.change.emit();
        return;
    }

    // --- Update Trend Graph for a specific community area ---
    const community_name = data.community[selected_index];
    const prediction = data.count_prediction[selected_index];
    const lag1 = data.count_lag_1[selected_index];
    const lag2 = data.count_lag_2[selected_index];

    trend_source.data = {
        x_obs: [0, 1], y_obs: [lag2, lag1],
        x_pred: [1, 2], y_pred: [lag1, prediction]
    };
    trend_plot.title.text = `Trend for ${community_name}`;
    trend_source.change.emit();

    // --- Update Top Services Div ---
    const services_string = data.top_3_services[selected_index];
    let services_html;

    if (services_string === 'No data available') {
        services_html = "<p style='color: #566573;'>Top service data is not available for this area.</p>";
    } else {
        const services_array = services_string.split(',').map(service => service.trim());
        services_html = "<ul style='padding-left: 20px; margin:0; font-size: 1.1em;'>";
        for (const service of services_array) {
            services_html += `<li style='margin-bottom: 8px;'>${service}</li>`;
        }
        services_html += "</ul>";
    }

    service_div.text = `
        <div style='
            font-family: system-ui, -apple-system, sans-serif;
            padding: 20px;
            background-color: #fdfdfd;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            height: 250px; /* CHANGED */
        '>
        <h3 style='margin-top:0; color: #2c3e50;'>Top Service Types</h3>
        ${services_html}
        </div>
    `;
""")

map_plot.js_on_event(Tap, callback)


# --- FINAL LAYOUT (SIMPLIFIED) ---
# Create a row for the two top panels
top_info_row = row(hotspot_div, service_list_div, spacing=20)

# Group the new row and the trend plot into a single column.
info_panel = column(top_info_row, trend_plot, spacing=20)

final_layout = column(
    header,
    description_div,
    row(map_plot, info_panel, spacing=20),
    sizing_mode='stretch_width'
)

curdoc().add_root(final_layout)
curdoc().title = "Chicago Service Request Predictions"