"""
Global Relocation Dashboard
Developed by: Luke Tam, Manuela Fernandes Maldonado, Karen Alderete Romo
Live version: https://relocation.streamlit.app/
This interactive dashboard helps users compare and evaluate countries in order to make informed relocation decisions.
Data is sourced from the World Bank and from United Nations. Charts and visualizations are built using Streamlit and Plotly.
The dashboard offers three analysis levels: Country, Metric, and Topic.
Users can explore indicators across seven major topics: Education, Employment, Environment, Health, Income, Infrastructure, and Safety.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
import streamlit as st

###################################
########## DATA CLEANING ##########
###################################

# Load data
df = pd.read_csv('Data/WDI Data.csv')
country_list = pd.read_csv('Data/Immigrant Population Data.csv')
topic_map = pd.read_csv('Data/Topic Map.csv')
continent_lookup = pd.read_csv('Data/Continent Lookup.csv')
iso_lookup = pd.read_csv('Data/ISO Lookup.csv')

# Filter to top 50 countries by immigrant population
df = df[df['Country Name'].isin(country_list['Country'])]

# Merge continent data
df = df.merge(continent_lookup, on="Country Name", how="left")

# Replace ".." with NA
df.replace("..", pd.NA, inplace=True)

# Remove extra characters from year columns
df.columns = [col.split()[0] if col[:4].isdigit() else col for col in df.columns]

# Create "Latest Data" column with the most recent non-null value across the years
year_columns = [col for col in df.columns if col.isdigit()]
df['Latest Data'] = df[year_columns].ffill(axis=1).iloc[:, -1]

# Add "Topic" column using the topic map
df = df.merge(topic_map, on='Series Name', how='left')

# Place "Topic" column after "Series Code" column
cols = df.columns.tolist()
series_code_index = cols.index('Series Code')
cols.insert(series_code_index + 1, cols.pop(cols.index('Topic')))
df = df[cols]

# Create grouped metrics dictionary
grouped_metrics = {}
for _, row in topic_map.iterrows():
    topic = row['Topic']
    metric = row['Series Name']
    if topic not in grouped_metrics:
        grouped_metrics[topic] = []
    grouped_metrics[topic].append(metric)

# Specify metrics where lower values are better
lower_is_better_metrics = [
    "Unemployment with advanced education (% of total labor force with advanced education)",
    "Unemployment, total (% of total labor force) (national estimate)",
    "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)",
    "Mortality rate, adult, female (per 1,000 female adults)",
    "Poverty headcount ratio at societal poverty line (% of population)",
    "Intentional homicides (per 100,000 people)"
]
metric_direction = {metric: -1 for metric in lower_is_better_metrics}

# Calculate normalized metric scores
df['Latest Data'] = pd.to_numeric(df['Latest Data'], errors='coerce')
normalized_scores = []
for metric in df['Series Name'].unique():
    subset = df[df['Series Name'] == metric]
    min_val = subset['Latest Data'].min()
    max_val = subset['Latest Data'].max()
    direction = metric_direction.get(metric, 1)
    if direction == 1:
        norm_values = (subset['Latest Data'] - min_val) / (max_val - min_val) * 100
    else:
        norm_values = (max_val - subset['Latest Data']) / (max_val - min_val) * 100
    temp_df = subset.copy()
    temp_df['Normalized Score'] = norm_values
    normalized_scores.append(temp_df)
normalized_df = pd.concat(normalized_scores)

# Calculate topic averages per country
topic_scores = normalized_df.groupby(['Country Name', 'Topic'])['Normalized Score'].mean().reset_index()

# Identify countries with missing topic scores
pivoted = topic_scores.pivot(index='Country Name', columns='Topic', values='Normalized Score')
valid_countries = pivoted.dropna().index.tolist()

# Create ISO code dictionary
country_flags = dict(zip(iso_lookup['Country'], iso_lookup['ISO Code']))

###################################
###### OVERALL DASHBOARD UI #######
###################################

# Set page width
st.set_page_config(layout="wide")

# Add title
st.title("Global Relocation Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Country-Level Analysis", "Metric-Level Analysis", "Topic-Level Analysis"])

# Global styling
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Bree+Serif&family=Poppins&family=Lobster&display=swap" rel="stylesheet">
    <style>
    * {
       overflow-anchor: none !important;
       }
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif !important;
    }
    h1 {
        font-family: 'Lobster', cursive !important;
        text-align: center !important;
        margin-bottom: 20px !important;
        text-decoration: none;
    }
    h2, h3, h4, h5, h6 {
        font-family: 'Bree Serif', serif !important;
        text-align: center !important;
        margin-bottom: 20px !important;
        text-decoration: none;
    }
    .stMarkdown a {
        display: none;
    }
    
    .stMarkdown a.allow-link {
        display: inline;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 50px;
    }
    .stTabs [data-baseweb="tab"] {
		height: 80px;
		white-space: pre-wrap;
	    background-color: #F0F2F6;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
		padding: 30px;
		font-family: 'Poppins', sans-serif;
		font-size: 20px;
        color: black;
    }
	.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
  		background-color: #f29f05;
  		font-weight: bold;
  		color: white;
	}
	.stTabs [data-baseweb="tab-highlight"] {
	    background-color: #f29f05;
	}
	.stTabs [data-baseweb="tab-list"] button:hover[aria-selected="false"] [data-testid="stMarkdownContainer"] p {
	    color: #f29f05;
	}
	.stTabs [data-baseweb="tab-list"] button:hover[aria-selected="false"] [data-baseweb="tab-highlight"] {
	    background-color: #f29f05;
	}
	.stTabs [data-baseweb="tab-panel"] [data-testid="stMarkdownContainer"] p {
	    font-family: 'Poppins', sans-serif;
	    font-size: 16px;
	}
	.stTabs [data-baseweb="tab-panel"] [data-testid="stWidgetLabel"] p {
	    font-weight: bold;
	}
	.stTabs [data-baseweb="tab-panel"] [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
	    font-weight: normal;
	}
	.stTabs [data-baseweb="tab-panel"] [data-baseweb="select"] {
	    font-family: 'Poppins', sans-serif;
	    font-size: 16px;
	}
	[data-testid="popover"] {
	    font-family: 'Poppins', sans-serif !important;
	    font-size: 16px !important;
	}
	[data-testid="stSliderThumbValue"] {
        font-family: 'Poppins', sans-serif;
        font-size: 14px;
    }
    [data-testid="stSliderTickBarMin"] {
        font-family: 'Poppins', sans-serif;
        font-size: 14px;
    }
    [data-testid="stSliderTickBarMax"] {
        font-family: 'Poppins', sans-serif;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Section divider styling
def section_divider():
    st.markdown("""
        <hr style="border: none; border-top: 2px dashed #aaa; margin: 50px 0;">
    """, unsafe_allow_html=True)

# Information card styling
def info_card(title, value, color, height=200):
    return f"""
    <div style="
        background-color:{color}; padding:20px; border: 3px solid;
        border-image: linear-gradient(to right, #1a1a1a, #444444) 1;
        text-align:center; margin:5px; height:{height}px; display:flex;
        flex-direction:column; justify-content:center;
    ">
        <p style="color:white; margin:0; font-size:20px; font-weight:700; font-family:'Poppins', sans-serif;">{title}</p>
        <div style="border-top: 1px solid rgba(255,255,255,0.3); margin:10px 0;"></div>
        <p style="color:white; margin:0; font-size:20px; font-weight:400; font-family:'Poppins', sans-serif;">{value}</p>
    </div>
    """

# Define continent-specific colors
continent_colors = {
    "Africa": "#b9375e",
    "Asia": "#f78c6b",
    "Europe": "#ffd166",
    "North America": "#06d6a0",
    "Oceania": "#118ab2",
    "South America": "#073b4c"
}

# Define topic-specific colors
topic_colors = {
    'Education': '#274001',
    'Employment': '#828a00',
    'Environment': '#f29f05',
    'Health': '#f25c05',
    'Income': '#d6568c',
    'Infrastructure': '#4d8584',
    'Safety': '#a62f03'
}

def footer():
    st.subheader("Footnotes")

    # Country flag title
    st.markdown(f"""
        <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-bottom: 50px;">
            Countries analyzed in this dashboard<br>(top 50 countries by 2024 immigrant population):
        </div>
        """, unsafe_allow_html=True)

    # Show country flags
    cols = st.columns(5)
    for idx, (country, code) in enumerate(country_flags.items()):
        with cols[idx % 5]:
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 50px;">
                <div style="display: inline-block; border: 1px solid black; overflow: hidden;">
                    <img src="https://flagcdn.com/w160/{code}.png" style="
                        width: 120px;
                        height: 80px;
                        object-fit: cover;
                        display: block;
                    ">
                </div>
                <p style="font-family: Poppins; font-size:14px; margin-top:10px;">{country}</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer text
    st.markdown("""
    <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 14px; margin-top: 50px;">
        <p><b>Data sources</b>: 
            <a class="allow-link" href="https://databank.worldbank.org/source/world-development-indicators" target="_blank" style="color: #118ab2;">
                World Bank - World Development Indicators (WDI)
            </a>, 
            <a class="allow-link" href="https://www.un.org/development/desa/pd/content/international-migrant-stock" target="_blank" style="color: #118ab2;">
                United Nations Migration Data
            </a>
        </p>
        <p><b>Developed by</b>: Luke Tam, Manuela Fernandes Maldonado, Karen Alderete Romo</p>
        <p>Babson College | OIM 7502 | Advanced Programming for Business Analytics | Spring 2025</p>
    </div>
    """, unsafe_allow_html=True)

###################################
## TAB 1: COUNTRY-LEVEL ANALYSIS ##
###################################

with tab1:
    st.header("Country-Level Analysis")

    # Tab description
    st.markdown(f"""
        <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-bottom: 40px;">
            Explore key indicators for a selected country across seven major topics.<br>
            View individual metric values, country rankings, and overall topic performance.
        </div>
        """, unsafe_allow_html=True)

    # Create country dropdown
    country_list = sorted(df['Country Name'].unique())
    selected_country = st.selectbox("Select a country:", country_list)

    # Group metrics by topic
    topics = sorted(topic_map['Topic'].unique())
    topic_metrics = {topic: topic_map[topic_map['Topic'] == topic]['Series Name'].tolist() for topic in topics}

    section_divider()

    ##### Info Cards #####

    for topic in topics:
        st.subheader(topic)

        # Show topic score
        topic_score_row = topic_scores[
            (topic_scores['Country Name'] == selected_country) &
            (topic_scores['Topic'] == topic)
            ]
        if not topic_score_row.empty and pd.notna(topic_score_row['Normalized Score'].values[0]):
            avg_score = topic_score_row['Normalized Score'].values[0]
            st.markdown(f"""
                        <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-bottom: 40px;">
                            Topic score: {avg_score:.2f} / 100
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                        <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-bottom: 40px;">
                            Topic score: No data
                        </div>
                    """, unsafe_allow_html=True)

        cols = st.columns(3)
        color = topic_colors.get(topic, "#274001")
        for i, metric in enumerate(topic_metrics[topic]):

            # Filter data for the selected metric
            metric_subset = df[df['Series Name'] == metric][['Country Name', 'Latest Data']]
            metric_subset['Latest Data'] = pd.to_numeric(metric_subset['Latest Data'], errors='coerce')
            metric_subset = metric_subset.dropna()

            # Calculate ranks
            metric_subset['Rank'] = metric_subset['Latest Data'].rank(ascending=False, method='min')

            # Get selected country value and rank
            if selected_country in metric_subset['Country Name'].values:
                value = metric_subset.loc[metric_subset['Country Name'] == selected_country, 'Latest Data'].values[0]
                rank = int(metric_subset.loc[metric_subset['Country Name'] == selected_country, 'Rank'].values[0])
                value_display = f"Value: {value:,.2f}"  # Format with 2 decimals
                rank_display = f"Rank: {rank} / 50"
            else:
                value_display = "No data"
                rank_display = ""

            # Display info card
            with cols[i % 3]:
                st.markdown(info_card(title=metric, value=f"{value_display}<br>{rank_display}", color=color, height=300),
                            unsafe_allow_html=True)

        section_divider()

    footer()

###################################
## TAB 2: METRIC-LEVEL ANALYSIS ###
###################################

with tab2:
    st.header("Metric-Level Analysis")

    # Tab description
    st.markdown(f"""
            <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-bottom: 40px;">
                Analyze and compare countries based on a selected metric.<br>
                View descriptive statistics, top/bottom country rankings, and geographic distribution patterns.
            </div>
            """, unsafe_allow_html=True)

    ##### Dropdown Lists #####

    # Create grouped dropdown
    sorted_topics = sorted(grouped_metrics.keys())
    selected_topic = st.selectbox("Choose a topic:", sorted_topics)
    sorted_metrics = sorted(grouped_metrics[selected_topic])
    selected_metric = st.selectbox("Choose a metric:", sorted_metrics)

    # Convert and drop NaNs
    metric_data = df[df['Series Name'] == selected_metric].copy()
    metric_data['Latest Data'] = pd.to_numeric(metric_data['Latest Data'], errors='coerce')
    values = metric_data['Latest Data'].dropna()

    # Compute stats
    stat_dict = {
        "Countries with Data": f"{len(values):,}",
        "Mean": f"{values.mean():,.2f}",
        "Median": f"{values.median():,.2f}",
        "Standard Deviation": f"{values.std():,.2f}",
        "Minimum": f"{values.min():,.2f}",
        "25th Percentile": f"{np.percentile(values, 25):,.2f}",
        "75th Percentile": f"{np.percentile(values, 75):,.2f}",
        "Maximum": f"{values.max():,.2f}"
    }

    section_divider()

    ##### Descriptive Statistics #####

    st.subheader("Key Statistics")

    # First row of info cards
    keys = list(stat_dict.keys())
    cols_row1 = st.columns(4)
    colors_row1 = ["#274001", "#828a00", "#f29f05", "#f25c05"]
    for i in range(4):
        html = info_card(keys[i], stat_dict[keys[i]], colors_row1[i])
        cols_row1[i].markdown(html, unsafe_allow_html=True)

    # Spacer between rows
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Second row of info cards
    cols_row2 = st.columns(4)
    colors_row2 = ["#d6568c", "#4d8584", "#a62f03", "#400d01"]
    for i in range(4, 8):
        html = info_card(keys[i], stat_dict[keys[i]], colors_row2[i - 4])
        cols_row2[i - 4].markdown(html, unsafe_allow_html=True)

    section_divider()

    ##### Bar Chart #####

    st.subheader("Country Rankings")

    # Prepare data for bar chart
    bar_data = metric_data[['Country Name', 'Latest Data', 'Continent']].dropna()
    bar_data['Latest Data'] = pd.to_numeric(bar_data['Latest Data'], errors='coerce')

    # Configure radio buttons for bar chart
    order = st.radio("Show top or bottom countries:", ["Top", "Bottom"], horizontal=True)
    top_n = st.radio("Select number of countries to display:", [5, 10, 15, 20, 25], index=2, horizontal=True)
    sorted_data = bar_data.sort_values(by='Latest Data', ascending=False)
    if order == "Top":
        selected_data = sorted_data.head(top_n)
        selected_data = selected_data.sort_values(by='Latest Data', ascending=False)
    else:
        selected_data = sorted_data.tail(top_n)
        selected_data = selected_data.sort_values(by='Latest Data', ascending=True)

    # Create bar chart
    selected_data['Color'] = selected_data['Continent'].map(continent_colors)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=selected_data['Country Name'],
        y=selected_data['Latest Data'],
        marker_color=selected_data['Color'],
        showlegend=False,
        customdata=selected_data[['Country Name', 'Latest Data']]
    ))

    # Add legend for continents
    for continent, color in continent_colors.items():
        fig_bar.add_trace(go.Bar(
            x=[None],
            y=[None],
            name=continent,
            marker_color=color,
            showlegend=True
        ))

    # Bar chart styling
    fig_bar.update_layout(
        height=700,
        xaxis={'categoryorder':'array', 'categoryarray': selected_data['Country Name'].tolist()},
        xaxis_title=dict(text="Country", standoff=20),
        xaxis_title_font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"}),
        xaxis_tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
        yaxis_title="Value of Selected Metric",
        yaxis_title_font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"}),
        yaxis_tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
        font=dict(family="Poppins, sans-serif"),
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.8,
            title=dict(
                text="Continent<br>",
                side="top",
                font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"})
            ),
            font=dict(
                family="Poppins, sans-serif",
                size=14,
                color="black"
            )
        )
    )

    # Bar chart tooltip content
    fig_bar.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Value of Selected Metric: %{customdata[1]:,.2f}<extra></extra>"
    )

    # Bar chart title
    st.markdown(f"""
    <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-top: 20px;">
        {order} {top_n} countries:<br>{selected_metric}
    </div>
    """, unsafe_allow_html=True)

    # Show bar chart
    st.plotly_chart(fig_bar, use_container_width=True)

    section_divider()

    ##### Choropleth Map #####

    st.subheader("Geographic Heat Map")

    # Prepare data for map
    map_data = metric_data[['Country Name', 'Latest Data']].dropna()
    map_data['Latest Data'] = pd.to_numeric(map_data['Latest Data'], errors='coerce')

    # Create map
    fig_map = px.choropleth(
        map_data,
        locations="Country Name",
        locationmode="country names",
        color="Latest Data",
        color_continuous_scale=px.colors.sequential.Oranges,
        custom_data=["Country Name", "Latest Data"]
    )

    # Map styling
    fig_map.update_layout(
        height=700,
        margin=dict(t=10),
        coloraxis_colorbar=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            title=None
        ),
        geo=dict(
            projection_type='equirectangular',
            showframe=True,
            showcoastlines=True,
            showcountries=True,
            showland=True,
            landcolor="lightgray",
        ),
        dragmode=False
    )

    # Map tooltip content
    fig_map.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Value of Selected Metric: %{customdata[1]:,.2f}<extra></extra>"
    )

    # Map title
    st.markdown(f"""
    <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-bottom: 20px;">
        {selected_metric}
    </div>
    """, unsafe_allow_html=True)

    # Show map
    st.plotly_chart(fig_map, use_container_width=True)

    # Add note to explain countries in gray
    st.markdown("""
    <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 14px; color: gray; margin-top: 10px;">
        Only the top 50 countries by immigrant population are included in this analysis. All other countries are shown in gray.
    </div>
    """, unsafe_allow_html=True)

    section_divider()

    ##### Box Plot #####

    st.subheader("Metric Distribution")

    # Allow users to group by continent for box plot
    group_by_continent = st.radio("Group by continent?", ["Yes", "No"], horizontal=True)

    # Create box plot
    x_col = "Continent" if group_by_continent == "Yes" else "Global"
    metric_data["Global"] = "All Countries"
    fig_box = px.box(
        metric_data,
        x=x_col,
        y="Latest Data",
        points="all",
        color="Continent" if group_by_continent == "Yes" else None,
        color_discrete_map=continent_colors if group_by_continent == "Yes" else None,
        custom_data=["Country Name", "Latest Data"],
        labels={"Latest Data": "Value of Selected Metric"},
    )

    # Box plot styling
    fig_box.update_layout(
        height=700,
        font=dict(family="Poppins, sans-serif"),
        xaxis=dict(
            tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
            categoryorder="array",
            categoryarray=sorted(metric_data["Continent"].dropna().unique().tolist())
        ),
        xaxis_title=dict(
            text="Continent" if group_by_continent == "Yes" else "",
            font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"}),
            standoff=40
        ),
        yaxis_tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
        yaxis_title=dict(
            text="Value of Selected Metric",
            font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"})
        ),
        showlegend=False
    )

    # Box plot tooltip content
    fig_box.update_traces(
        hoveron="boxes+points", boxmean=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Value of Selected Metric: %{customdata[1]:,.2f}<extra></extra>"
    )

    # Box plot title
    st.markdown(f"""
    <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-top: 20px;">
        {selected_metric}
    </div>
    """, unsafe_allow_html=True)

    # Show box plot
    st.plotly_chart(fig_box, use_container_width=True)

    section_divider()

    footer()

    ###################################
    ### TAB 3: TOPIC-LEVEL ANALYSIS ###
    ###################################

    with tab3:
        st.header("Topic-Level Analysis")

        # Tab description
        st.markdown(f"""
                    <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-bottom: 20px;">
                        Compare countries across broader topics using aggregated scores.<br>
                        Customize weights based on your preferences to create personalized country rankings.
                    </div>
                    """, unsafe_allow_html=True)

        section_divider()

        ##### Radar Chart #####

        st.subheader("Topic Score Comparison")

        # Create country dropdowns
        col1, col2 = st.columns(2)
        with col1:
            country1 = st.selectbox('Select first country:', sorted(valid_countries), key="country1")
        with col2:
            country2 = st.selectbox('Select second country:', sorted(valid_countries), index=1, key="country2")

        # Prepare data based on user-selected countries
        metric_details = (
            normalized_df[['Country Name', 'Topic', 'Series Name', 'Normalized Score']]
            .groupby(['Country Name', 'Topic'])
            .apply(lambda d: list(zip(d['Series Name'], d['Normalized Score'])))
            .reset_index()
            .rename(columns={0: 'Metric Details'})
        )
        topic_scores = topic_scores.merge(metric_details, on=['Country Name', 'Topic'], how='left')
        def clean_metric_details(detail_list):
            cleaned = []
            for name, score in detail_list:
                if pd.isna(score):
                    cleaned.append((name, "No data"))
                else:
                    cleaned.append((name, f"{score:.2f}"))
            return cleaned
        topic_scores['Metric Details'] = topic_scores['Metric Details'].apply(clean_metric_details)
        country1_scores = topic_scores[topic_scores['Country Name'] == country1]
        country2_scores = topic_scores[topic_scores['Country Name'] == country2]

        # Crearte radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=country1_scores['Normalized Score'],
            theta=country1_scores['Topic'],
            fill='toself',
            name=country1,
            customdata=country1_scores['Metric Details'],
            text=[country1] * len(country1_scores),
            hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "%{theta} score: %{r:.2f}<br><br>" +
                    "%{customdata[0][0]}: %{customdata[0][1]}<br>" +
                    "%{customdata[1][0]}: %{customdata[1][1]}<br>" +
                    "%{customdata[2][0]}: %{customdata[2][1]}"
            )
        ))
        fig.add_trace(go.Scatterpolar(
            r=country2_scores['Normalized Score'],
            theta=country2_scores['Topic'],
            fill='toself',
            name=country2,
            customdata = country2_scores['Metric Details'],
            text = [country2] * len(country2_scores),
            hovertemplate = (
                    "<b>%{text}</b><br>" +
                    "%{theta} score: %{r:.2f}<br><br>" +
                    "%{customdata[0][0]}: %{customdata[0][1]}<br>" +
                    "%{customdata[1][0]}: %{customdata[1][1]}<br>" +
                    "%{customdata[2][0]}: %{customdata[2][1]}"
            )
        ))

        # Radar chart styling
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                ),
            ),
            showlegend=True,
            height=700,
            margin=dict(t=120),
            font=dict(family="Poppins, sans-serif", size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5,
                font=dict(
                    family="Poppins, sans-serif",
                    size=14,
                    color="black"
                )
            )
        )

        # Show radar chart
        st.plotly_chart(fig, use_container_width=True)

        section_divider()

        ##### Scatter Plot #####

        st.subheader("Topic Scores by Country")

        # Create topic dropdowns
        topics = sorted(topic_scores['Topic'].unique())
        col1, col2 = st.columns(2)
        with col1:
            x_topic = st.selectbox('Select X-axis topic:', topics)
        with col2:
            y_topic = st.selectbox('Select Y-axis topic:', topics, index=1)

        # Prepare data for scatter plot
        scatter_data = topic_scores.pivot(index='Country Name', columns='Topic', values='Normalized Score').reset_index()
        scatter_data = scatter_data.merge(continent_lookup, on='Country Name', how='left')
        scatter_data = scatter_data.dropna(subset=[x_topic, y_topic])

        # Create scatter plot with trend line
        fig_scatter = go.Figure()
        x_vals = scatter_data[x_topic]
        y_vals = scatter_data[y_topic]
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = slope * x_line + intercept
        r_value, _ = pearsonr(x_vals, y_vals)
        fig_scatter.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Trend Line',
            hovertemplate=f"Correlation (r): {r_value:.2f}<extra></extra>",
            showlegend=False
        ))
        fig_scatter.add_trace(go.Scatter(
            x=scatter_data[x_topic],
            y=scatter_data[y_topic],
            mode='markers',
            marker=dict(
                size=12,
                color=scatter_data['Continent'].map(continent_colors),
                line=dict(width=1, color='black')
            ),
            text=scatter_data['Country Name'],
            hovertemplate=(
                    "<b>%{text}</b><br>" +
                    x_topic + " score: %{x:.2f}<br>" +
                    y_topic + " score: %{y:.2f}<extra></extra>"
            ),
            showlegend=False
        ))

        # Add legend for continents
        for continent, color in continent_colors.items():
            fig_scatter.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=12, color=color),
                name=continent,
                showlegend=True
            ))

        # Scatter plot styling
        fig_scatter.update_layout(
            height=700,
            xaxis=dict(
                gridcolor='lightgray',
                showgrid=True
            ),
            xaxis_title=dict(text=x_topic + " Score", standoff=20),
            yaxis=dict(
                title=y_topic + " Score",
                gridcolor='lightgray',
                showgrid=True
            ),
            xaxis_title_font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"}),
            yaxis_title_font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"}),
            xaxis_tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
            yaxis_tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.4,
                title=dict(
                    text="Continent<br>",
                    side="top",
                    font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"})
                ),
                font=dict(
                    family="Poppins, sans-serif",
                    size=14,
                    color="black"
                )
            )
        )

        # Scatter plot title
        st.markdown(f"""
            <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold;">
                <br>{x_topic} score vs. {y_topic.lower()} score by country
            </div>
            """, unsafe_allow_html=True)

        # Show scatter plot
        st.plotly_chart(fig_scatter, use_container_width=True)

        section_divider()

        ##### Bar Chart #####

        st.subheader("Weighted Country Rankings")

        # Create topic importance sliders
        st.markdown("<p style='font-family: Poppins, sans-serif; font-weight: bold;'>"
                    "Set the importance of each topic (on a scale from 1 to 10) for your relocation decision:</p>",
                    unsafe_allow_html=True)
        importance = {}
        cols = st.columns(3)
        for idx, topic in enumerate(topics):
            with cols[idx % 3]:
                importance[topic] = st.slider(topic, min_value=1, max_value=10, value=5)

        # Calculate weighted scores
        topic_wide = topic_scores.pivot(index='Country Name', columns='Topic', values='Normalized Score')
        for topic in topics:
            topic_wide[topic] = topic_wide[topic] * importance[topic]
        topic_wide['Weighted Total'] = topic_wide.sum(axis=1)
        min_score = topic_wide['Weighted Total'].min()
        max_score = topic_wide['Weighted Total'].max()
        topic_wide['Weighted Total'] = (topic_wide['Weighted Total'] - min_score) / (max_score - min_score) * 100
        weighted_ranking = topic_wide[['Weighted Total']].reset_index()
        weighted_ranking = weighted_ranking.dropna()

        # Prepare data for bar chart
        weighted_ranking = weighted_ranking.merge(continent_lookup, on='Country Name', how='left')
        weighted_ranking['Color'] = weighted_ranking['Continent'].map(continent_colors)

        # Configure radio buttons for bar chart
        order = st.radio("Show Top or Bottom Countries:", ["Top", "Bottom"], horizontal=True)
        top_n = st.radio("Number of Countries to Display:", [5, 10, 15, 20, 25], index=2, horizontal=True)
        if order == "Top":
            ranked_data = weighted_ranking.sort_values(by='Weighted Total', ascending=False).head(top_n)
        else:
            ranked_data = weighted_ranking.sort_values(by='Weighted Total', ascending=True).head(top_n)

        # Create bar chart
        fig_weighted = go.Figure()
        fig_weighted.add_trace(go.Bar(
            x=ranked_data['Country Name'],
            y=ranked_data['Weighted Total'],
            marker_color=ranked_data['Color'],
            hovertemplate="<b>%{x}</b><br>Overall weighted score: %{y:.2f}<extra></extra>",
            showlegend=False
        ))

        # Add legend for continents
        for continent, color in continent_colors.items():
            fig_weighted.add_trace(go.Bar(
                x=[None],
                y=[None],
                name=continent,
                marker_color=color,
                showlegend=True
            ))

        # Bar chart styling
        fig_weighted.update_layout(
            height=700,
            xaxis={'categoryorder': 'array', 'categoryarray': ranked_data['Country Name'].tolist()},
            xaxis_title=dict(text="Country", standoff=20),
            xaxis_title_font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"}),
            xaxis_tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
            yaxis_title="Weighted Score",
            yaxis_tickfont=dict(family="Poppins, sans-serif", size=14, color="gray"),
            yaxis_title_font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"}),
            font=dict(family="Poppins, sans-serif", size=14),
            xaxis_tickangle=-45,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.8,
                title=dict(
                    text="Continent<br>",
                    side="top",
                    font=dict(family="Poppins, sans-serif", size=14, color="black", **{"weight": "bold"})
                ),
                font=dict(
                    family="Poppins, sans-serif",
                    size=14,
                    color="black"
                )
            )
        )

        # Create bar chart title
        st.markdown(f"""
            <div style="text-align: center; font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: bold; margin-top: 20px;">
                {order} {top_n} countries: Overall weighted score
            </div>
        """, unsafe_allow_html=True)

        # Show bar chart
        st.plotly_chart(fig_weighted, use_container_width=True)

        section_divider()

        footer()