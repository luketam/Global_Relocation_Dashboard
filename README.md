
# Global Relocation Dashboard

This interactive Streamlit dashboard helps users compare and evaluate countries 
based on key economic, social, and environmental indicators. 
Users can explore data through three levels of analysis: 

- **Country-Level Analysis**: View a selected country's metrics and ranks.
- **Metric-Level Analysis**: Compare countries by specific metrics with charts and statistics.
- **Topic-Level Analysis**: Compare countries using aggregated topic scores and personalized weighting.

## Features

- Explore data for 50 countries selected based on UN migration statistics.
- View country-level metric values and rankings across seven topics.
- Analyze descriptive statistics and visualize top/bottom countries for any metric.
- Compare topic-level performance and generate personalized weighted rankings.
- Interactive visualizations built with Plotly and Streamlit.

## Data Sources

- [World Bank - World Development Indicators (WDI)](https://databank.worldbank.org/source/world-development-indicators)
- [United Nations Migration Data](https://www.un.org/development/desa/pd/content/international-migrant-stock)

## Project Structure

```
app.py
requirements.txt
Data/
├── WDI Data.csv
├── Immigrant Population Data.csv
├── Topic Map.csv
├── Continent Lookup.csv
├── ISO Lookup.csv
```

## How to Run Locally

1. Clone this repository.
2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the Streamlit app using Command Prompt or Terminal:

```
streamlit run app.py
```

## Live Version

A live version of the dashboard can be found [here](https://relocation.streamlit.app/).

## Credits

This dashboard was created in April 2025 by Luke Tam, Manuela Fernandes Maldonado, and Karen Alderete Romo as part of Babson College's OIM 7502 "Advanced Programming for Business Analytics" course.  


---

