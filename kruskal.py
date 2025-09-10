import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import folium
from osmnx import routing

# Baca file hasil clustering dan centroid
data = pd.read_csv('hasil_clustering.csv')  # Asumsi file hasil clustering ada
centroids = np.load('centroids.npy')        # Asumsi file centroids ada

# Hitung rata-rata lokasi untuk pusat graf jalan
center_lat = data['latitude'].mean()
center_lon = data['longitude'].mean()

# Buat graf jalan sekitar pusat area menggunakan bbox
# Menggunakan `dist_type="bbox"` untuk mencegah FutureWarning
G_osm = ox.graph_from_point(
    (center_lat, center_lon),
    dist=4000,
    dist_type="bbox",  # Tambahan untuk kompatibilitas masa depan
    network_type="all"
)

# Fungsi untuk menghitung panjang rute jalan dari satu titik ke titik lain
def calculate_route_length(G, origin, destination):
    origin_node = ox.distance.nearest_nodes(G, origin[1], origin[0])
    destination_node = ox.distance.nearest_nodes(G, destination[1], destination[0])
    route = nx.shortest_path(G, origin_node, destination_node, weight="length")
    
    # Dapatkan panjang rute menggunakan fungsi bawaan OSMnx
    route_length = sum(
        ox.utils_graph.get_route_edge_attributes(G, route, "length")
    )
    
    return route, route_length

# Buat peta menggunakan Folium untuk visualisasi semua rumah dan ODP
mymap = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=16,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery"
)

# Warna untuk tiap cluster
colors = ['blue', 'green', 'purple', 'orange', 'darkred']

# Total panjang kabel keseluruhan
total_cable_length = 0

# Tambahkan edge dari setiap rumah ke ODP terdekatnya dengan panjang rute terdekat
all_routes = []  # Menyimpan semua rute untuk visualisasi
for idx, row in data.iterrows():
    rumah_coords = (row['latitude'], row['longitude'])
    cluster_id = row['Cluster']
    odp_coords = centroids[cluster_id - 1]  # Dapatkan koordinat ODP sesuai cluster

    # Hitung rute dan panjang kabel dari rumah ke ODP
    route, route_length = calculate_route_length(G_osm, rumah_coords, odp_coords)
    total_cable_length += route_length

    # Simpan rute untuk visualisasi
    all_routes.append((route, route_length, rumah_coords, odp_coords))

    # Tambahkan marker untuk setiap rumah dengan informasi panjang kabel
    folium.Marker(
        location=rumah_coords,
        popup=f"Nama: {row['NAMA PELANGGAN']}<br>Cluster: {row['Cluster']}<br>Panjang Kabel: {route_length:.2f} meter",
        icon=folium.Icon(color=colors[row['Cluster'] % len(colors)])
    ).add_to(mymap)

# Visualisasikan semua rute dari rumah ke ODP pada peta
for route, route_length, rumah_coords, odp_coords in all_routes:
    route_coords = [(G_osm.nodes[node]['y'], G_osm.nodes[node]['x']) for node in route]
    folium.PolyLine(
        locations=route_coords,
        color="blue",
        weight=2,
        opacity=0.7
    ).add_to(mymap)

# Tambahkan marker untuk ODP di setiap cluster
for idx, (lat, lon) in enumerate(centroids):
    folium.Marker(
        location=(lat, lon),
        popup=f"ODP Cluster {idx+1}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(mymap)

# Tambahkan informasi total panjang kabel ke peta
total_cable_info = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 350px;
        height: 100px;
        background: linear-gradient(to right, #4CAF50, #81C784);
        color: white;
        font-family: Arial, sans-serif;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        padding: 20px;
        z-index:9999;
        font-size: 18px;
        text-align: center;">
        <b>Total Panjang Kabel Keseluruhan:</b><br>
        <span style="font-size: 24px; font-weight: bold;">{total_cable_length:.2f} meter</span>
    </div>
"""
mymap.get_root().html.add_child(folium.Element(total_cable_info))

# Simpan peta sebagai HTML
output_file = "peta_kmeans_odp_all_routes_satelit.html"
mymap.save(output_file)
print(f"Peta telah disimpan sebagai '{output_file}'.")
