import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from dotenv import load_dotenv
import os
import plotly.express as px
import sqlalchemy
from snowflake.sqlalchemy import URL

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="COVID-19 Europe Explorer", page_icon="🌍", layout="wide")
st.title("COVID-19 in 2020, Europe")
st.write("Interactive map and charts showing COVID-19 data across European countries using a selectable date range for the year 2020.")

# ---------- Data Loading and Processing Section ----------
@st.cache_data
def load_europe_map():
    local_map_path = "countries.geojson" 
    try:
        if not os.path.exists(local_map_path):
            st.error(f"Map file not found at {local_map_path}.")
            return gpd.GeoDataFrame()
        
        world = gpd.read_file(local_map_path)
        if "ADMIN" not in world.columns or "CONTINENT" not in world.columns or "SUBREGION" not in world.columns:
            st.error("GeoJSON file is missing required properties like ADMIN, CONTINENT, or SUBREGION.")
            return gpd.GeoDataFrame()

        europe_gdf = world[(world.CONTINENT == "Europe") |
                       ((world.ADMIN == "Russia") & (world.SUBREGION == "Eastern Europe")) |
                       (world.ADMIN == "Turkey")].copy()
        
        for idx, row in europe_gdf[europe_gdf.ADMIN == "Russia"].iterrows():
            european_part_russia = box(-180, -90, 50, 90) 
            clipped_geometry = row.geometry.intersection(european_part_russia)
            europe_gdf.loc[idx, "geometry"] = clipped_geometry
        
        if europe_gdf.empty:
            st.warning("No European countries found after filtering GeoJSON.")
        return europe_gdf
    except Exception as e:
        st.error(f"Error loading or processing map GeoJSON data: {e}")
        return gpd.GeoDataFrame()


#Snowflake query, and pandas dataframe processing
@st.cache_data
def load_and_process_covid_data():
    local_cache_path = "covid_data_cache.csv"
    
    try:
        # try Snowflake first
        try:
            print("Connecting to Snowflake...")
            engine = sqlalchemy.create_engine(URL(
                user=os.getenv("SNOWFLAKE_USER"),
                password=os.getenv("SNOWFLAKE_PWD"),
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                role=os.getenv("SNOWFLAKE_ROLE"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA")
            ))
            
            query_daily = """
            SELECT COUNTRY_REGION, ISO3166_1, DATE, CASES AS DAILY_CASES, DEATHS AS DAILY_DEATHS, POPULATION
            FROM ECDC_GLOBAL
            WHERE CONTINENTEXP = 'Europe' AND POPULATION IS NOT NULL AND POPULATION > 0 AND DATE IS NOT NULL
            ORDER BY COUNTRY_REGION, DATE;
            """
            df_daily = pd.read_sql(query_daily, engine)

            # Save to cache for future fallback
            df_daily.to_csv(local_cache_path, index=False)
            print("Connected to Snowflake")
            
        except Exception as e:
            # If no internet or whatever, automatically fall back to cache
            print(f"Cannot connect to Snowflake. Falling back to cached data.")
            
            if os.path.exists(local_cache_path):
                df_daily = pd.read_csv(local_cache_path)

                # convert all columns to uppercase to match Snowflake format
                df_daily.columns = [col.upper() for col in df_daily.columns]
                df_daily["DATE"] = pd.to_datetime(df_daily["DATE"])
            else:
                st.error("No internet connection and no cached data available.")
                return pd.DataFrame()

        df_daily.columns = [col.upper() for col in df_daily.columns]
        df_daily.rename(columns={"COUNTRY_REGION": "COUNTRY_NAME"}, inplace=True)

        #Great Britain has UK code but needs GB, and Greece has EL but needs GR for the map
        df_daily.loc[df_daily["COUNTRY_NAME"] == "United_Kingdom", "ISO3166_1"] = "GB"
        df_daily.loc[df_daily["COUNTRY_NAME"] == "Greece", "ISO3166_1"] = "GR"

        #Make consistent names for some countries
        df_daily.loc[df_daily["COUNTRY_NAME"] == "Czechia", "COUNTRY_NAME"] = "Czech Republic"
        df_daily.loc[df_daily["COUNTRY_NAME"] == "Moldova, Republic of", "COUNTRY_NAME"] = "Moldova"

        #cleaning up country names by removing underscores
        df_daily["COUNTRY_NAME"] = df_daily["COUNTRY_NAME"].str.replace("_", " ")

        # These countries would make the map a bit ugly, and are relatively small
        countries_to_exclude = ["Armenia", "Azerbaijan", "Georgia", "Cyprus", "Malta"]
        df_daily = df_daily[df_daily["COUNTRY_NAME"].isin(countries_to_exclude) == False]
        
        # also remove countries with very small populations
        POPULATION_THRESHOLD = 150000
        country_populations = df_daily.groupby("COUNTRY_NAME")["POPULATION"].max()
        countries_to_keep = country_populations[country_populations >= POPULATION_THRESHOLD].index
        df_daily = df_daily[df_daily["COUNTRY_NAME"].isin(countries_to_keep)]
        if df_daily.empty:
            st.warning(f"No countries after population filter of {POPULATION_THRESHOLD}.")
            return pd.DataFrame()
        
        # Handle missing data points in time series
        min_date_overall = df_daily["DATE"].min()
        max_date_overall = df_daily["DATE"].max()

        all_dates_range = pd.date_range(start=min_date_overall, end=max_date_overall, name="DATE")

        processed_country_dfs = []
        
        for country_name_iter_val in df_daily["COUNTRY_NAME"].unique():
            # Get the subset of data for the current country
            country_subset_df = df_daily[df_daily["COUNTRY_NAME"] == country_name_iter_val].copy()

            country_subset_df = country_subset_df.set_index("DATE")
            
            iso_code_for_country = country_subset_df["ISO3166_1"].dropna().iloc[0] if not country_subset_df["ISO3166_1"].dropna().empty else None
            population_for_country = country_subset_df["POPULATION"].dropna().iloc[0] if not country_subset_df["POPULATION"].dropna().empty else None
                
            # Reindex this country's data against the full date range
            country_reindexed_data = country_subset_df.reindex(all_dates_range)
            
            # Fill in the identifying columns for the new date rows
            country_reindexed_data["COUNTRY_NAME"] = country_reindexed_data["COUNTRY_NAME"].fillna(country_name_iter_val)
            country_reindexed_data["ISO3166_1"] = country_reindexed_data["ISO3166_1"].fillna(iso_code_for_country)
            country_reindexed_data["POPULATION"] = country_reindexed_data["POPULATION"].fillna(population_for_country)
            
            # fill missing data points with zeros (typically early pandemic days)
            country_reindexed_data["DAILY_CASES"] = country_reindexed_data["DAILY_CASES"].fillna(0)
            country_reindexed_data["DAILY_DEATHS"] = country_reindexed_data["DAILY_DEATHS"].fillna(0)
            
            # Add it back to our collection
            processed_country_dfs.append(country_reindexed_data.reset_index())
            

        df_daily = pd.concat(processed_country_dfs, ignore_index=True)
        
        # Some data cleaning and aggregation
        df_daily["DAILY_CASES"] = df_daily["DAILY_CASES"].clip(lower=0)
        df_daily["DAILY_DEATHS"] = df_daily["DAILY_DEATHS"].clip(lower=0)
        df_daily["DATE"] = pd.to_datetime(df_daily["DATE"])
        df_daily["DAILY_CASES_PER_100K"] = (df_daily["DAILY_CASES"] / df_daily["POPULATION"]) * 100000
        df_daily["DAILY_DEATHS_PER_100K"] = (df_daily["DAILY_DEATHS"] / df_daily["POPULATION"]) * 100000
        cols_to_fill = ["DAILY_CASES_PER_100K", "DAILY_DEATHS_PER_100K"]
        for col in cols_to_fill:
            df_daily[col] = df_daily[col].replace([float("inf"), -float("inf")], 0).fillna(0)
        return df_daily
    except Exception as e:
        st.error(f"Error loading/processing COVID data: {e}")
        return pd.DataFrame()

# Streamlit config for the plotly charts   
config = {
        "scrollZoom": False,
        "displayModeBar": True, 
        "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", 
                                    "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
                                    "zoomInGeo", "zoomOutGeo", "resetGeo", "hoverClosestGeo", 
                                    "hoverCompareCartesian", "toggleSpikelines"],
        "displaylogo": False
    }

# Some custom css that i found handy to make the app a bit nicer
st.markdown("""
    <style>
        h1 {margin-top: 0rem !important; margin-bottom: 0.25rem !important; font-size: 2rem !important;}
        h2 {margin-top: 0.5rem !important; margin-bottom: 0.25rem !important; font-size: 1.5rem !important;}
        h3 {margin-top: 0.5rem !important; margin-bottom: 0.25rem !important; font-size: 1.2rem !important;}

        /* Vertical separator padding for the map and barchart*/
        .stApp [data-testid="stHorizontalBlock"] > div:nth-child(2) {
            border-left: 1px solid #cccccc !important;
            padding-left: 20px !important;
        }

        /* But it looks ugly on the config panel, so if child divs have a radio button, remove it */
        .stApp [data-testid="stHorizontalBlock"] > div:has(.stRadio) {
            border-left: none !important;
            padding-left: 0px !important;
        }
            
        /* also remove it for the metrics */
        .stApp [data-testid="stHorizontalBlock"] > div:has(.stMetric) {
            border-left: none !important;
            padding-left: 0px !important;
        }

        hr {
            border: none !important;
            border-top: 1px solid #cccccc !important;
            margin-top: 0.75rem !important;
            margin-bottom: 1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ---------- Load datasets ----------
europe_gdf = load_europe_map()
daily_covid_df = load_and_process_covid_data()

# ---------- Initialize session control variables ----------
metric_type_str = "Cases"
normalization_type_str = "Per 100k"
selected_date_range_val = None 
top_n_val = 10
selected_country_for_trend_val = None

# ---------- Global config controls ----------
st.subheader("Chart Configuration")
cfg_col1, cfg_col2, cfg_col3 = st.columns([1, 1, 3])
with cfg_col1:
    metric_type_str = st.radio("Metric Type:", ("Cases", "Deaths"), index=0, key="global_metric_type")
with cfg_col2:
    normalization_type_str = st.radio("Normalization:", ("Per 100k", "Total Number"), index=0, key="global_normalization")

with cfg_col3:
    if not daily_covid_df.empty:
        available_dates = sorted(daily_covid_df["DATE"].dt.date.unique())
        if available_dates:
            default_start_date = available_dates[0]
            default_end_date = available_dates[-1]
            if selected_date_range_val is None:
                #initially default to the whole date range
                selected_date_range_val = (default_start_date, default_end_date)
            
            current_start, current_end = selected_date_range_val
            if not (current_start in available_dates and current_end in available_dates and current_start <= current_end):
                 selected_date_range_val = (default_start_date, default_end_date)
            
            # Init of the slider for selection
            selected_date_range_val = st.select_slider(
                "Date Range:", options=available_dates, value=selected_date_range_val,
                format_func=lambda d: d.strftime("%Y-%m-%d"), key="map_date_range_slider"
            )
            if selected_date_range_val[0] > selected_date_range_val[1]:
                selected_date_range_val = (selected_date_range_val[1], selected_date_range_val[0])
        else:
            st.warning("No dates available for range selection.")
            selected_date_range_val = None
    else:
        st.warning("Daily COVID data not loaded. Date range slider unavailable.")
        selected_date_range_val = None
st.markdown("---")

# ---------- Main layout ----------
left_col, right_col = st.columns([3, 2]) 
data_for_bar_chart_df = pd.DataFrame()

# The left column has the map, 60% of the width
with left_col:
    st.subheader("Map Display & Date Range Selection")
    

    # Plotly Map with fixed European view
    if selected_date_range_val and not europe_gdf.empty and not daily_covid_df.empty:

        #make a subest of the data based on the range selection
        start_date, end_date = pd.to_datetime(selected_date_range_val[0]), pd.to_datetime(selected_date_range_val[1])
        range_data_df = daily_covid_df[
            (daily_covid_df["DATE"] >= start_date) & (daily_covid_df["DATE"] <= end_date)
        ].copy()

        # wrangle the data a bit, by grouping by name and iso code
        map_agg_data_df = range_data_df.groupby(["COUNTRY_NAME", "ISO3166_1"]).agg(
            RANGE_CASES=("DAILY_CASES", "sum"), RANGE_DEATHS=("DAILY_DEATHS", "sum"),
            POPULATION=("POPULATION", "first")
        ).reset_index()

        #calculate the aggregations for the subset
        map_agg_data_df["RANGE_CASES_PER_100K"] = (map_agg_data_df["RANGE_CASES"] / map_agg_data_df["POPULATION"]) * 100000
        map_agg_data_df["RANGE_DEATHS_PER_100K"] = (map_agg_data_df["RANGE_DEATHS"] / map_agg_data_df["POPULATION"]) * 100000

        map_metric_base = metric_type_str.upper()
        map_norm_suffix = "_PER_100K" if normalization_type_str == "Per 100k" else ""
        map_selected_metric_col = f"RANGE_{map_metric_base}{map_norm_suffix}"

        # Unify the covid data with the geo data to make the map interactive
        if map_selected_metric_col in map_agg_data_df.columns:
            merged_gdf_for_plotly = europe_gdf.merge(
                map_agg_data_df, left_on="ISO_A2_EH", right_on="ISO3166_1", how="inner"
            )
            merged_gdf_for_plotly[map_selected_metric_col] = merged_gdf_for_plotly[map_selected_metric_col].fillna(0)
            
            current_display_date_label = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            map_title_text = f"{metric_type_str} ({normalization_type_str})<br>{current_display_date_label}"


            fig_plotly = px.choropleth(
                merged_gdf_for_plotly,
                geojson=merged_gdf_for_plotly.geometry,
                locations=merged_gdf_for_plotly.index,
                color=map_selected_metric_col,
                hover_name="COUNTRY_NAME",
                color_continuous_scale="Reds",
                hover_data={map_selected_metric_col: True},
                height=700
            )

            fig_plotly.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" + 
                                f"{metric_type_str} ({normalization_type_str}): %{{customdata[0]:.2f}}" +
                                "<extra></extra>"
            )
            
            fig_plotly.update_geos(
                visible=False, 
                lonaxis_range=[-25, 50], 
                lataxis_range=[41, 75],  
                projection_type="mercator"
            )
            
            fig_plotly.update_layout(
                title_text=map_title_text, 
                title_x=0, 
                title_font_size=16,
                coloraxis_colorbar=dict(
                    title=f"{metric_type_str} ({normalization_type_str})", 
                    title_side="top" 
                ),
                margin={"r":0,"t":40,"l":0,"b":0}, 
                geo=dict(bgcolor="rgba(0,0,0,0)"),
                dragmode=False,
                hovermode="closest" 
            )
            
            st.plotly_chart(fig_plotly, use_container_width=True, config=config)
            
            #update the data which will be used for the bar chart
            data_for_bar_chart_df = map_agg_data_df[["COUNTRY_NAME", map_selected_metric_col, "POPULATION"]].copy()
            data_for_bar_chart_df.dropna(subset=["COUNTRY_NAME", map_selected_metric_col], inplace=True)
            data_for_bar_chart_df["COUNTRY_NAME"] = data_for_bar_chart_df["COUNTRY_NAME"].astype(str)
        else:
            st.warning(f"Map metric column '{map_selected_metric_col}' not found in aggregated data.")
            st.map()
    elif europe_gdf.empty:
         st.warning("Map cannot be displayed: GeoDataFrame not loaded.")
         st.map()
    else:
        st.info("Select a date range to display the map, or check data loading.")
        st.map()


#right col has the bar chart, 40% of the width
with right_col:
    # Top Countries Bar Chart
    st.subheader("Top Countries for Selected Range")

    top_n_options = list(range(1, 21))
    top_n_val = st.selectbox(
        label="Show Top N Countries:",
        options=top_n_options,
        index=top_n_options.index(top_n_val) if top_n_val in top_n_options else 9,
        key="bar_top_n"
    )

    bar_metric_base = metric_type_str.upper()
    bar_norm_suffix = "_PER_100K" if normalization_type_str == "Per 100k" else ""
    bar_selected_metric_col = f"RANGE_{bar_metric_base}{bar_norm_suffix}"

    if not data_for_bar_chart_df.empty and bar_selected_metric_col in data_for_bar_chart_df.columns:
        #sort data for proper top-N display with largest values at the top
        top_n_data_df = data_for_bar_chart_df.nlargest(top_n_val, bar_selected_metric_col).sort_values(by=bar_selected_metric_col, ascending=True)
        
        if not top_n_data_df.empty:
            fig_bar_plotly = px.bar(
                top_n_data_df,
                x=bar_selected_metric_col,
                y="COUNTRY_NAME",
                orientation="h",
                labels={"COUNTRY_NAME": "Country", bar_selected_metric_col: f"{metric_type_str} ({normalization_type_str})"},
                height=max(200, top_n_val * 50),
                hover_data={bar_selected_metric_col: ":.1f"}
            )
            
            fig_bar_plotly.update_layout(
                title_text=f"Top {top_n_val} Countries",
                title_x=0.5, 
                yaxis_title="Country",
                xaxis_title=f"{metric_type_str} in Range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
                font=dict(size=10), 
                margin=dict(l=10, r=10, t=30, b=10), 
                yaxis={"categoryorder":"total ascending"},
            )
            
            st.plotly_chart(fig_bar_plotly, use_container_width=True, config=config)
        else: 
            st.write(f"No data for Top {top_n_val} countries in selected range.")
    else: 
        st.write("Bar chart data unavailable for the selected range.")

st.markdown("---")
# Comparative Trend Analysis Section
st.subheader("Comparative Country Trends For Selected Range")

if not daily_covid_df.empty:
    all_countries_for_trend = sorted(daily_covid_df["COUNTRY_NAME"].unique())
    if all_countries_for_trend:
        #add European average option for baseline comparison
        all_countries_for_trend_with_avg = ["All Countries (Average)"] + all_countries_for_trend

        #smart default selection handling for multiselect
        if not isinstance(selected_country_for_trend_val, list):
            if selected_country_for_trend_val is None or selected_country_for_trend_val not in all_countries_for_trend_with_avg:
                default_selection = ["All Countries (Average)"] if "All Countries (Average)" in all_countries_for_trend_with_avg else ([all_countries_for_trend_with_avg[1]] if len(all_countries_for_trend_with_avg) > 1 else [])
            else:
                default_selection = [selected_country_for_trend_val]
        else:
            default_selection = [country for country in selected_country_for_trend_val if country in all_countries_for_trend_with_avg]
            if not default_selection and all_countries_for_trend_with_avg:
                default_selection = ["All Countries (Average)"] if "All Countries (Average)" in all_countries_for_trend_with_avg else ([all_countries_for_trend_with_avg[1]] if len(all_countries_for_trend_with_avg) > 1 else [])

        selected_countries_for_trend = st.multiselect(
            "Select Countries:",
            options=all_countries_for_trend_with_avg,
            default=default_selection,
            key="trend_countries_multiselect"
        )
        #update session state to remember selections
        selected_country_for_trend_val = selected_countries_for_trend

    else:
        st.write("No countries available for trend selection.")
else:
    st.write("Daily data not loaded for trend chart.")

if selected_countries_for_trend and selected_date_range_val and not daily_covid_df.empty:
    start_date_trend, end_date_trend = pd.to_datetime(selected_date_range_val[0]), pd.to_datetime(selected_date_range_val[1])

    #filter regular countries from "All Countries (Average)" special case
    selected_countries_for_trend_filtered = [
        country for country in selected_countries_for_trend if country != "All Countries (Average)"
    ]

    comparative_trend_data_df = daily_covid_df[
        (daily_covid_df["COUNTRY_NAME"].isin(selected_countries_for_trend_filtered)) &
        (daily_covid_df["DATE"] >= start_date_trend) &
        (daily_covid_df["DATE"] <= end_date_trend)
    ].copy()

    trend_metric_base_name = metric_type_str.upper()
    trend_norm_suffix_name = "_PER_100K" if normalization_type_str == "Per 100k" else ""
    final_trend_metric_col = f"DAILY_{trend_metric_base_name}{trend_norm_suffix_name}"

    #special handling for European average calculation
    if "All Countries (Average)" in selected_countries_for_trend:
        all_countries_avg_df = daily_covid_df[
            (daily_covid_df["DATE"] >= start_date_trend) &
            (daily_covid_df["DATE"] <= end_date_trend)
        ].groupby("DATE")[final_trend_metric_col].mean().reset_index()
        all_countries_avg_df["COUNTRY_NAME"] = "All Countries (Average)"
        comparative_trend_data_df = pd.concat([comparative_trend_data_df, all_countries_avg_df], ignore_index=True)

    if not comparative_trend_data_df.empty and final_trend_metric_col in comparative_trend_data_df.columns:
        #prettier hover labels with formatted values
        comparative_trend_data_df["hover_text"] = comparative_trend_data_df["COUNTRY_NAME"] + ": " + comparative_trend_data_df[final_trend_metric_col].round(1).astype(str)
        
        fig_trend_plotly = px.line(
            comparative_trend_data_df,
            x="DATE",
            y=final_trend_metric_col,
            color="COUNTRY_NAME",
            markers=True,
            labels={"DATE": "Date",
                    final_trend_metric_col: f"Daily {metric_type_str} ({normalization_type_str})",
                    "COUNTRY_NAME": "Country"},
            hover_data={final_trend_metric_col: ":.1f"}
        )

        title_countries_str = ", ".join(selected_countries_for_trend)
        fig_trend_plotly.update_layout(
            title_text=f"Daily Trend for: {title_countries_str}<br>({metric_type_str} - {normalization_type_str})",
            title_x=0,
            xaxis_title=f"Date ({start_date_trend.strftime('%Y-%m-%d')} to {end_date_trend.strftime('%Y-%m-%d')})",
            yaxis_title=f"Daily {metric_type_str} ({normalization_type_str})",
            font=dict(size=10),
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis_tickangle=-30,
            hovermode="x unified",
            legend_title_text="Country"
        )
        
        #nice visuals with markers and proper line thickness
        fig_trend_plotly.update_traces(marker=dict(size=4), line=dict(width=1.5))

        st.plotly_chart(fig_trend_plotly, use_container_width=True, config=config)

    elif not comparative_trend_data_df.empty:
        st.write(f"Trend metric '{final_trend_metric_col}' not found for the selected countries.")
    else:
        st.write(f"No daily data for the selected countries in the specified range.")
elif selected_countries_for_trend:
    st.write("Please select a date range to view the trend.")
elif not daily_covid_df.empty and ("all_countries_for_trend" in locals() and all_countries_for_trend):
    st.write("Select countries and a date range to see their trends.")

# ---------- Key Insights Summary Section ----------
st.markdown("---")
st.subheader("Key Insights Summary")

st.markdown("""
    <style>
        /* Change color to blue */
        div[data-testid="stMetricDelta"] {
            color: #0068c9 !important;
        }
        
        /* Remove the up arrow icno */
        div[data-testid="stMetricDelta"] svg {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Find some meaningful numbers to show
if not daily_covid_df.empty and selected_date_range_val:
    insights_col1, insights_col2, insights_col3, insights_col4, insights_col5 = st.columns(5)
    
    start_date_insights, end_date_insights = pd.to_datetime(selected_date_range_val[0]), pd.to_datetime(selected_date_range_val[1])
    filtered_df = daily_covid_df[(daily_covid_df["DATE"] >= start_date_insights) & (daily_covid_df["DATE"] <= end_date_insights)]
    
    # aggregate
    agg_by_country = filtered_df.groupby("COUNTRY_NAME").agg(
        Total_Cases=("DAILY_CASES", "sum"),
        Total_Deaths=("DAILY_DEATHS", "sum"),
        Population=("POPULATION", "first")
    )
    agg_by_country["Cases_Per_100K"] = (agg_by_country["Total_Cases"] / agg_by_country["Population"]) * 100000
    agg_by_country["Deaths_Per_100K"] = (agg_by_country["Total_Deaths"] / agg_by_country["Population"]) * 100000
    
    # Daily totals
    daily_totals = filtered_df.groupby("DATE").agg(
        Cases=("DAILY_CASES", "sum"),
        Deaths=("DAILY_DEATHS", "sum")
    )
    
    # Calculate total cases and deaths
    total_cases = filtered_df["DAILY_CASES"].sum()
    total_deaths = filtered_df["DAILY_DEATHS"].sum()
    days_in_range = (end_date_insights - start_date_insights).days + 1
    daily_avg_cases = total_cases / days_in_range
    daily_avg_deaths = total_deaths / days_in_range

    case_fatality_rate = (total_deaths / total_cases * 100) if total_cases > 0 else 0
    
    # Column 1: Total statistics
    with insights_col1:

        st.metric("Total Cases", f"{int(total_cases):,}", f"Daily avg: {int(daily_avg_cases):,}")
        st.metric("Total Deaths", f"{int(total_deaths):,}", f"Daily avg: {int(daily_avg_deaths):,}")
    
    # Column 2: Most severe day and case fatality rate
    with insights_col2:

        worst_day = daily_totals.nlargest(1, "Cases").index[0]
        worst_day_count = daily_totals.nlargest(1, "Cases")["Cases"].values[0]
        
        st.metric("Most Severe Day", worst_day.strftime("%Y-%m-%d"), f"{int(worst_day_count):,} cases")
        
        st.metric("Case Fatality Rate", f"{case_fatality_rate:.2f}%", f"{int(total_deaths):,} deaths")
    
    # Column 3: Highest/Lowest Cases per 100K
    with insights_col3:

        countries_with_cases = agg_by_country[agg_by_country["Total_Cases"] >= 0]
        
        # Highest per 100K
        highest_country = agg_by_country.nlargest(1, "Cases_Per_100K").index[0]
        highest_value = agg_by_country.nlargest(1, "Cases_Per_100K")["Cases_Per_100K"].values[0]
        highest_format = f"{highest_value:.1f} per 100K"
        
        # Lowest per 100K
        lowest_country = countries_with_cases.nsmallest(1, "Cases_Per_100K").index[0]
        lowest_value = countries_with_cases.nsmallest(1, "Cases_Per_100K")["Cases_Per_100K"].values[0]
        lowest_format = f"{lowest_value:.1f} per 100K"
        
        st.metric("Highest Cases / 100K", f"{highest_country}", f"{highest_format}")
        st.metric("Lowest Cases / 100K", f"{lowest_country}", f"{lowest_format}")
    
    # Column 4: Highest/Lowest Deaths per 100K
    with insights_col4:
        
        # Avioding division with 0
        countries_with_deaths = agg_by_country[agg_by_country["Total_Deaths"] > 0]
        
        if not countries_with_deaths.empty:
            # Highest deaths per 100K
            highest_deaths_country = agg_by_country.nlargest(1, "Deaths_Per_100K").index[0]
            highest_deaths_value = agg_by_country.nlargest(1, "Deaths_Per_100K")["Deaths_Per_100K"].values[0]
            highest_deaths_format = f"{highest_deaths_value:.1f} per 100K"
            
            # Lowest deaths per 100K (among countries with deaths)
            lowest_deaths_country = countries_with_deaths.nsmallest(1, "Deaths_Per_100K").index[0]
            lowest_deaths_value = countries_with_deaths.nsmallest(1, "Deaths_Per_100K")["Deaths_Per_100K"].values[0]
            lowest_deaths_format = f"{lowest_deaths_value:.1f} per 100K"
            
            st.metric("Highest Deaths / 100K", f"{highest_deaths_country}", f"{highest_deaths_format}")
            st.metric("Lowest Deaths / 100K", f"{lowest_deaths_country}", f"{lowest_deaths_format}")

# ---------- Insights in Markdown ----------
st.markdown("""
### Key Observations for the whole Dataset

- **Waves**: There were two significant waves, the first in March-May and the second in October-December 2020.

- **First events in Europe**: The first case was confirmed in France on January 24, 2020, and the first death occured on February 15, 2020, also in France.
            
- **First wave**: The first wave affected whole Europe, but the western states were hit harder than the eastern ones.

- **Summer 2020**: The summer months saw a significant decrease in cases and deaths, with many countries reporting very low numbers.

- **Second wave**: The second wave started in October 2020, with a significant increase in cases and deaths across Europe, also dwarfing the numbers from the first wave. Now, the countries in middle Europe were hit the hardest.
            
- **Overall**: Central and western Europe stand out as hotspots, Scandinavia and parts of the east stay comparatively low.
                     
""")