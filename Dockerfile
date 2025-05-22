FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for GeoPandas with minimal bloat
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the GeoJSON file during build
RUN curl -o countries.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson

# Copy the rest of the app
COPY . .

# Expose port 8501
EXPOSE 8501

# Set environment variables for Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]