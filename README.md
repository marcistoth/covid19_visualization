# COVID-19 Dashboard

A Streamlit dashboard for visualizing COVID-19 statistics across European countries.

# Local Setup Instructions

1. **Create and activate a virtual environment**
   ```bash
   conda create -n covid_venv python=3.10
   ```

   ```bash
   conda activate covid_venv
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas geopandas plotly snowflake-sqlalchemy python-dotenv shapely pyarrow<19.0.0
   ```

## Configuration

1. **Create a `.env` file** with Snowflake credentials:
   ```
   SNOWFLAKE_USER=*username*   
   SNOWFLAKE_PWD=*password*
   SNOWFLAKE_ACCOUNT=*account*
   SNOWFLAKE_ROLE=*role*
   SNOWFLAKE_WAREHOUSE=*warehouse*
   SNOWFLAKE_DATABASE=COVID19_EPIDEMIOLOGICAL_DATA
   SNOWFLAKE_SCHEMA=PUBLIC
   ```

2. **Get the dataset** on the Snowflake marketplace:
   - The data is available [here](https://app.snowflake.com/marketplace/listing/GZSNZ7F5UH/starschema-covid-19-epidemiological-data)

3. **Download the GeoJSON file** containing European country boundaries:
   - Download the .geojson file from [this GitHub repository](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_50m_admin_0_countries.geojson)
   - Rename it to "countries.geojson", and place it in the project's root directory

## Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**
   - Open a browser and go to `http://localhost:8501`


# Docker Setup Instructions

1. **Create a `.env` file** with Snowflake credentials:
   ```
   SNOWFLAKE_USER=*username*   
   SNOWFLAKE_PWD=*password*
   SNOWFLAKE_ACCOUNT=*account*
   SNOWFLAKE_ROLE=*role*
   SNOWFLAKE_WAREHOUSE=*warehouse*
   SNOWFLAKE_DATABASE=COVID19_EPIDEMIOLOGICAL_DATA
   SNOWFLAKE_SCHEMA=PUBLIC
   ```

2. **Quick start with docker compose**
   ```bash
   docker-compose up -d
   ```

3. **Or use individual docker commands**
   ```bash
   docker build -t covid-dashboard .
   ```

   ```bash
   docker run -p 8501:8501 -v $(pwd)/.env:/app/.env covid-dashboard
   ```

2. **Access the dashboard**
   - Open a browser and go to `http://0.0.0.0:8501`





