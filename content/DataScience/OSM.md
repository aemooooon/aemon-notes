---
title: 
draft: false
tags: 
date: 2024-08-16
---
# OpenStreetMap (OSM) API and OSMnx in Python

**Author**: Dr. Vanessa Bastos | Dr. Gorden Jiang  
**School**: School of Earth and Environment | Te Kura Aronukurangi  
**Course**: Web GIS and Geoinformatics - GEOG 324 / Spatial Data Science - GISC412  
**Date**: 8/9/24

---

## Table of Contents

1. [Introduction to OSM and OSMnx](#introduction-to-osm-and-osmnx)
2. [Main Functions and Methods](#main-functions-and-methods)
   - [Extracting Road Networks](#extracting-road-networks)
   - [Geocoding with OSMnx](#geocoding-with-osmnx)
   - [Routing and Shortest Path](#routing-and-shortest-path)
   - [Extracting Boundaries](#extracting-boundaries)
   - [Extracting Geographic Features by Tags](#extracting-geographic-features-by-tags)
   - [Exporting Data from OSMnx](#exporting-data-from-osmnx)
3. [Examples of API Usage](#examples-of-api-usage)
4. [Useful Resources](#useful-resources)

---

## 1. Introduction to OSM and OSMnx

**OpenStreetMap (OSM)** is a global, editable map database that allows users to contribute and access spatial data. OSM is often used in GIS applications, urban planning, and transportation analysis. **OSMnx** is a Python library that simplifies the interaction with OSM data and provides tools for retrieving, analyzing, and visualizing geospatial networks, particularly for street networks.

- **OSMnx** can be used for:
  - Downloading street networks and geographical data.
  - Geocoding (converting addresses to coordinates).
  - Performing graph/network analysis.
  - Routing and calculating shortest paths.
  - Visualizing geographic data, such as streets, buildings, and natural features.

---

## 2. Main Functions and Methods

This section summarizes the key OSM APIs and methods used in Python via the OSMnx library.

### 2.1 Extracting Road Networks

To retrieve a street network graph for a specific area, you can use `ox.graph_from_place()` or `ox.graph_from_bbox()`.

```python
import osmnx as ox

# Extract street network by place name
place = "University of Canterbury"
network = ox.graph_from_place(place)

# Extract street network using bounding box coordinates
north, south, east, west = -43.5199, -43.5282, 172.5883, 172.5729
network = ox.graph_from_bbox(north, south, east, west, network_type='all')

# Visualize the network
fig, ax = ox.plot_graph(network)
```

### 2.2 Geocoding with OSMnx

Geocoding converts an address to geographical coordinates. The `ox.geocode()` method is used to perform geocoding.

```python
import geopandas as gpd
from shapely.geometry import Point

# Convert address to coordinates
address_list = ["11A Montana Avenue, Christchurch, New Zealand", ...]
geocode_result = ox.geocode(address_list[0])

# Store results as GeoDataFrame
gdf = gpd.GeoDataFrame({'Address': address_list}, geometry=geom, crs=4326)
```

### 2.3 Routing and Shortest Path

To compute the shortest path between two points on a network, you can use `ox.distance.nearest_nodes()` to find the nearest nodes and `nx.shortest_path()` to compute the route.

```python
import networkx as nx

# Find nearest nodes to start and end points
start_point, end_point = (-43.5228, 172.5833), (-43.5215, 172.5798)
start_node = ox.distance.nearest_nodes(network, start_point[1], start_point[0])
end_node = ox.distance.nearest_nodes(network, end_point[1], end_point[0])

# Find the shortest path
route = nx.shortest_path(network, source=start_node, target=end_node, weight='length')

# Plot the route on the street network
ox.plot_graph_route(network, route)
```

### 2.4 Extracting Boundaries

To extract boundaries (such as the polygon of a city or neighborhood), `ox.geocode_to_gdf()` can be used.

```python
# Extract a boundary polygon of a specified place
area = ox.geocode_to_gdf("University of Canterbury")

# Plot the boundary
area.plot()
```

### 2.5 Extracting Geographic Features by Tags

Tags in OSM describe the attributes of geographical features, such as roads, buildings, or natural features. You can use `ox.features_from_place()` to extract features by tags.

```python
# Extract building features using the "building" tag
buildings = ox.features_from_place("University of Canterbury", tags={"building": True})

# Extract trees using the "natural" tag
trees = ox.features_from_place("University of Canterbury", tags={"natural": "tree"})

# Plot the extracted features
buildings.plot()
trees.plot()
```

### 2.6 Exporting Data from OSMnx

Once you have processed the data into a GeoDataFrame, you can export it to various formats such as Shapefiles or GeoPackages.

```python
# Export data to a shapefile
buildings.to_file("Lab4_data/buildings.shp")

# Export data to a GeoPackage
buildings.to_file("Lab4_data/buildings.gpkg", driver="GPKG")
```

---

## 3. Examples of API Usage

Here are a few examples of using OSM APIs in a Pythonic way:

- **Extracting the road network for a specific area**:  
  Retrieve the road network of a city or neighborhood using `graph_from_place()` and perform graph analysis or routing.

- **Geocoding multiple addresses**:  
  Use `ox.geocode()` to geocode a list of addresses and store the results in a GeoDataFrame for further spatial analysis.

- **Visualizing shortest paths**:  
  Combine NetworkX and OSMnx to compute and visualize the shortest path between two locations in a transportation network.

- **Exporting to standard GIS formats**:  
  After data wrangling, export your data to formats like Shapefiles or GeoPackages for use in other GIS applications like QGIS or ArcGIS.

---

## 4. Useful Resources

- **OSMnx Documentation**: [OSMnx GitHub](https://github.com/gboeing/osmnx)
- **OpenStreetMap Documentation**: [OSM Wiki](https://wiki.openstreetmap.org/wiki/Main_Page)
- **NetworkX Documentation**: [NetworkX Documentation](https://networkx.github.io/documentation/stable/)
- **Christchurch City Council Spatial Open Data Portal**: [Christchurch Open Data](https://opendata-christchurchcity.hub.arcgis.com/)

