from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import folium
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import networkx as nx

app = Flask(__name__)

# Konfigurasi folder upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}

# Variabel global untuk menyimpan data
uploaded_data = None
clustered_data = None

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    global uploaded_data

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "File tidak ditemukan."})
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Format file tidak didukung."})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            uploaded_data = pd.read_excel(filepath, engine="openpyxl")
            
            # Validasi kolom MAPS
            if 'MAPS' not in uploaded_data.columns:
                return jsonify({"success": False, "message": "Kolom 'MAPS' tidak ditemukan dalam file."})

            # Hitung jumlah data
            total_data = len(uploaded_data)

            map_file = generate_initial_map(uploaded_data)
            return render_template('index.html', map_file=map_file, total_data=total_data)
        except Exception as e:
            print(f"Error saat memproses file: {e}")
            return jsonify({"success": False, "message": str(e)})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template('index.html', map_file=None, total_data=None)

# Fungsi untuk membuat peta awal
def generate_initial_map(data):
    try:
        data[['latitude', 'longitude']] = data['MAPS'].str.split(' ', expand=True)
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data = data.dropna(subset=['latitude', 'longitude'])

        if data.empty:
            raise ValueError("Data latitude dan longitude tidak valid atau kosong.")

        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        mymap = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        for _, row in data.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Nama: {row.get('NAMA PELANGGAN', 'Tidak Ada')}<br>Koordinat: {row['latitude']}, {row['longitude']}"
            ).add_to(mymap)

        map_file = 'map_initial.html'
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], map_file)
        mymap.save(map_path)

        return f'uploads/{map_file}'
    except Exception as e:
        print(f"Error saat membuat peta awal: {e}")
        return None

# Fungsi untuk clustering
import random

def generate_unique_colors(n):
    """Generate n unique colors in hex format."""
    colors = []
    for _ in range(n):
        colors.append(f"#{random.randint(0, 0xFFFFFF):06x}")
    return colors

def process_clustering_and_generate_map(data, n_clusters):
    global clustered_data

    try:
        # Split coordinates
        data[['latitude', 'longitude']] = data['MAPS'].str.split(' ', expand=True)
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data = data.dropna(subset=['latitude', 'longitude'])

        if data.empty:
            raise ValueError("Data latitude dan longitude tidak valid atau kosong.")

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data[['latitude', 'longitude']])
        data['Cluster'] = kmeans.labels_ + 1  # Cluster labels start from 1
        clustered_data = data

        # Generate unique colors for each cluster
        cluster_colors = generate_unique_colors(n_clusters)

        # Create map
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        mymap = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Add markers for each point
        for _, row in data.iterrows():
            cluster_idx = row['Cluster'] - 1
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color=cluster_colors[cluster_idx],
                fill=True,
                fill_color=cluster_colors[cluster_idx],
                fill_opacity=0.7,
                popup=f"Nama: {row.get('NAMA PELANGGAN', 'Tidak Ada')}<br>Cluster: {row['Cluster']}<br>Koordinat: {row['latitude']}, {row['longitude']}"
            ).add_to(mymap)

        # Add legend for cluster colors
        legend_html = """
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            width: 200px;
            background-color: white;
            border: 1px solid black;
            z-index: 1000;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
        <strong>Legend:</strong><br>
        """
        for i, color in enumerate(cluster_colors, start=1):
            legend_html += f'<div style="margin: 5px 0;"><span style="display: inline-block; width: 20px; height: 20px; background-color: {color}; border-radius: 50%;"></span> Cluster {i}</div>'
        legend_html += "</div>"
        mymap.get_root().html.add_child(folium.Element(legend_html))

        # Save map
        map_file = 'map_clustered.html'
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], map_file)
        mymap.save(map_path)

        return f'uploads/{map_file}'
    except Exception as e:
        print(f"Error saat clustering: {e}")
        return None


@app.route('/cluster', methods=['POST'])
def cluster():
    global uploaded_data, clustered_data

    if uploaded_data is None:
        return jsonify({"success": False, "message": "Data belum diunggah."})

    try:
        # Evaluasi jumlah cluster optimal
        optimal_clusters, best_score, evaluation = find_optimal_clusters(uploaded_data)

        # Lakukan clustering
        clustered_data = uploaded_data.copy()
        map_file = process_clustering_and_generate_map(uploaded_data, optimal_clusters)

        # Informasi cluster
        cluster_info = clustered_data['Cluster'].value_counts().to_dict()

        if map_file:
            return jsonify({
                "success": True,
                "map_file": map_file,
                "optimal_clusters": optimal_clusters,
                "cluster_info": cluster_info,
                "silhouette_score": round(best_score, 4),
                "evaluation": evaluation
            })
        else:
            return jsonify({"success": False, "message": "Gagal membuat peta clustering."})
    except Exception as e:
        print(f"Error di endpoint /cluster: {e}")
        return jsonify({"success": False, "message": str(e)})



# Fungsi untuk menentukan titik koordinat ODP dengan logika pembatasan
from scipy.spatial import distance_matrix
import numpy as np
import folium

def process_odp_and_generate_map(data):
    global clustered_data

    try:
        if clustered_data is None:
            raise ValueError("Data belum di-cluster. Lakukan clustering terlebih dahulu.")

        # Validasi awal data latitude dan longitude
        if clustered_data['latitude'].isnull().any() or clustered_data['longitude'].isnull().any():
            raise ValueError("Data latitude atau longitude mengandung nilai NaN.")

        # Pastikan latitude dan longitude berada dalam rentang yang valid
        if not ((-90 <= clustered_data['latitude']).all() and (clustered_data['latitude'] <= 90).all()):
            raise ValueError("Latitude berada di luar rentang (-90 hingga 90).")
        if not ((-180 <= clustered_data['longitude']).all() and (clustered_data['longitude'] <= 180).all()):
            raise ValueError("Longitude berada di luar rentang (-180 hingga 180).")

        clusters = clustered_data.groupby('Cluster')
        odp_points = []  # Untuk menyimpan data ODP
        house_to_odp_mapping = []  # Untuk menyimpan hubungan rumah ke ODP

        # Pusat peta berdasarkan semua data
        center_lat = clustered_data['latitude'].mean()
        center_lon = clustered_data['longitude'].mean()
        mymap = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Generate unique colors for clusters
        unique_clusters = clustered_data['Cluster'].nunique()
        cluster_colors = generate_unique_colors(unique_clusters)

        # Proses setiap cluster
        for cluster_id, group in clusters:
            print(f"Memproses cluster: {cluster_id}")
            group_coords = group[['latitude', 'longitude']].values
            num_houses = len(group_coords)
            print(f"Jumlah rumah di cluster {cluster_id}: {num_houses}")

            # Buat centroid awal (sementara untuk setiap ODP)
            odp_centroids = []
            assigned_houses = set()  # Indeks rumah yang sudah dihubungkan

            while len(assigned_houses) < num_houses:
                # Validasi remaining_houses untuk menghindari indeks tidak valid
                remaining_houses = [i for i in range(num_houses) if i not in assigned_houses]
                if not remaining_houses:
                    print(f"Tidak ada rumah tersisa untuk cluster {cluster_id}.")
                    break  # Semua rumah sudah dihubungkan
                
                remaining_coords = group_coords[remaining_houses]

                # Tentukan centroid baru untuk ODP
                if not odp_centroids:
                    centroid = group_coords.mean(axis=0)  # Centroid awal
                else:
                    centroid = remaining_coords.mean(axis=0)

                print(f"Centroid baru untuk cluster {cluster_id}, ODP {len(odp_centroids) + 1}: {centroid}")

                # Tambahkan ODP ke daftar
                odp_id = len(odp_centroids) + 1
                odp_centroids.append(centroid)
                odp_points.append({
                    "Cluster": cluster_id,
                    "ODP": odp_id,
                    "latitude": centroid[0],
                    "longitude": centroid[1]
                })

                # Hitung jarak rumah ke centroid ODP ini
                distances = distance_matrix([centroid], group_coords)[0]
                sorted_indices = np.argsort(distances)

                # Hubungkan hingga 8 rumah terdekat ke ODP ini
                connected_house_count = 0
                for house_idx in sorted_indices:
                    if house_idx in assigned_houses:
                        continue  # Lewati rumah yang sudah dihubungkan
                    if connected_house_count >= 8:
                        break  # Maksimal 8 rumah per ODP

                    # Hubungkan rumah ke ODP
                    house_row = group.iloc[house_idx]
                    house_to_odp_mapping.append({
                        "Cluster": cluster_id,
                        "ODP": odp_id,
                        "House_Name": house_row.get("NAMA PELANGGAN", "Tidak Ada"),
                        "House_Latitude": house_row.latitude,
                        "House_Longitude": house_row.longitude
                    })

                    # Tambahkan garis antara rumah dan ODP
                    folium.PolyLine(
                        locations=[[house_row.latitude, house_row.longitude], [centroid[0], centroid[1]]],
                        color='blue',
                        weight=2,
                        opacity=0.7
                    ).add_to(mymap)

                    assigned_houses.add(house_idx)
                    connected_house_count += 1

                print(f"ODP {odp_id} pada cluster {cluster_id} terhubung dengan {connected_house_count} rumah.")

                # Tambahkan marker ODP ke peta
                folium.Marker(
                    location=centroid,
                    popup=f"Cluster {cluster_id}, ODP {odp_id}<br>Koordinat: {centroid[0]}, {centroid[1]}<br>Rumah Terhubung: {connected_house_count}",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(mymap)

            # Tambahkan marker rumah dengan warna cluster dan keterangan ODP
            for idx, row in group.iterrows():
                cluster_idx = row['Cluster'] - 1

                # Cari ODP yang terhubung ke rumah ini
                connected_odp = next((odp for odp in house_to_odp_mapping
                                      if odp['House_Latitude'] == row['latitude']
                                      and odp['House_Longitude'] == row['longitude']), None)

                odp_info = f"Cluster: {row['Cluster']}, ODP: {connected_odp['ODP']}" if connected_odp else "Tidak ada informasi ODP"

                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    color=cluster_colors[cluster_idx],
                    fill=True,
                    fill_color=cluster_colors[cluster_idx],
                    fill_opacity=0.7,
                    popup=f"Nama: {row.get('NAMA PELANGGAN', 'Tidak Ada')}<br>{odp_info}<br>Koordinat: {row['latitude']}, {row['longitude']}"
                ).add_to(mymap)

        # Simpan peta dengan ODP
        map_file = 'map_with_odp.html'
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], map_file)
        mymap.save(map_path)

        print("Peta ODP berhasil dibuat.")
        return f'uploads/{map_file}', odp_points, house_to_odp_mapping
    except Exception as e:
        print(f"Error saat menentukan ODP: {e}")
        return None, [], []


# Endpoint untuk menentukan titik ODP
@app.route('/odp', methods=['POST'])
def determine_odp():
    global clustered_data

    if clustered_data is None:
        return jsonify({"success": False, "message": "Data belum di-cluster. Lakukan clustering terlebih dahulu."})

    try:
        # Terima tiga nilai dari fungsi
        map_file, odp_points, house_to_odp_mapping = process_odp_and_generate_map(clustered_data)
        
        if map_file:
            return jsonify({
                "success": True,
                "map_file": map_file,
                "odp_points": odp_points,
                "house_to_odp_mapping": house_to_odp_mapping
            })
        else:
            return jsonify({"success": False, "message": "Gagal membuat peta ODP."})
    except Exception as e:
        print(f"Error di endpoint /odp: {e}")
        return jsonify({"success": False, "message": str(e)})

import osmnx as ox
import networkx as nx

def calculate_cable_per_house(clustered_data, odp_points, house_to_odp_mapping):
    try:
        cable_connections = []
        total_length = 0
        house_to_odp_lengths = []

        if clustered_data.empty:
            raise ValueError("Data cluster kosong.")
        if not odp_points:
            raise ValueError("Data ODP kosong.")

        # Ambil graf jalan dari OpenStreetMap
        center_lat = clustered_data['latitude'].mean()
        center_lon = clustered_data['longitude'].mean()

        print("📡 Mengambil data jalan dari OpenStreetMap...")
        G = ox.graph_from_point((center_lat, center_lon), dist=5000, network_type='drive')

        if G is None or len(G.nodes) == 0:
            raise ValueError("Graf jalan tidak ditemukan di area ini.")

        # Iterasi setiap ODP (menggunakan nomor ODP yang sudah ditentukan sebelumnya)
        for odp in odp_points:
            odp_index = odp["ODP"]
            try:
                odp_node = ox.nearest_nodes(G, odp["longitude"], odp["latitude"])
            except ValueError:
                print(f"🚨 Node ODP tidak ditemukan untuk ODP: {odp}")
                continue

            # Proses rumah secara batch untuk menghindari delay besar
            cluster_houses = clustered_data[clustered_data["Cluster"] == odp["Cluster"]]
            batch_size = 10  # Proses 10 rumah per iterasi
            house_list = cluster_houses.to_dict(orient='records')

            for i in range(0, len(house_list), batch_size):
                batch_houses = house_list[i : i + batch_size]

                for house in batch_houses:
                    try:
                        house_node = ox.nearest_nodes(G, house["longitude"], house["latitude"])
                    except ValueError:
                        print(f"🚨 Node rumah tidak ditemukan untuk: {house}")
                        continue

                    # Hitung rute terpendek dari rumah ke ODP
                    try:
                        path_coords = nx.shortest_path(G, house_node, odp_node, weight='length')
                        path_length = nx.shortest_path_length(G, house_node, odp_node, weight='length')

                        # Pastikan hanya satu jalur untuk setiap rumah
                        if any(house["latitude"] == entry["House_Latitude"] and house["longitude"] == entry["House_Longitude"]
                               for entry in house_to_odp_lengths):
                            continue  # Rumah ini sudah terhubung, lewati

                        # Tambahkan jalur ke koneksi kabel
                        cable_connections.append((path_coords, odp["Cluster"], odp_index, path_length))

                        # Simpan panjang kabel rumah ke ODP
                        house_to_odp_lengths.append({
                            "Cluster": odp["Cluster"],
                            "ODP": odp_index,
                            "House_Name": house["NAMA PELANGGAN"],
                            "House_Latitude": house["latitude"],
                            "House_Longitude": house["longitude"],
                            "ODP_Latitude": odp["latitude"],
                            "ODP_Longitude": odp["longitude"],
                            "Cable_Length": path_length
                        })

                        total_length += path_length
                    except nx.NetworkXNoPath:
                        print(f"🚨 Tidak ada jalur antara rumah {house_node} dan ODP {odp_node}")
                        continue

        return total_length, cable_connections, house_to_odp_lengths, G
    except Exception as e:
        print(f"❌ Error saat menghitung kabel per rumah: {e}")
        return 0, [], [], None


import folium

def visualize_house_to_odp_connections(clustered_data, odp_points, cable_connections, house_to_odp_lengths, G):
    try:
        center_lat = clustered_data['latitude'].mean()
        center_lon = clustered_data['longitude'].mean()
        mymap = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Warna untuk setiap cluster
        cluster_colors = [
            "#FF5733", "#33FF57", "#3357FF", "#F333FF", "#FF33A2", "#33FFF5", "#F5FF33", "#FFA533", "#33FFAA"
        ]

        # Tambahkan titik ODP ke peta
        for odp in odp_points:
            odp_index = odp["ODP"]
            folium.Marker(
                location=[odp["latitude"], odp["longitude"]],
                popup=f"<b>Cluster {odp['Cluster']}, ODP {odp_index}</b><br>Koordinat: {odp['latitude']}, {odp['longitude']}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(mymap)

        # Tambahkan titik rumah dengan informasi koneksi
        for house_info in house_to_odp_lengths:
            cluster_index = int(house_info['Cluster']) - 1
            color = cluster_colors[cluster_index % len(cluster_colors)]

            popup_info = (
                f"Nama: {house_info['House_Name']}<br>"
                f"Terhubung ke Cluster {house_info['Cluster']}, ODP {house_info['ODP']}<br>"
                f"Panjang Kabel: {house_info['Cable_Length']:.2f} meter"
            )

            folium.CircleMarker(
                location=[house_info["House_Latitude"], house_info["House_Longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_info
            ).add_to(mymap)

        # Tambahkan jalur kabel dari rumah ke ODP
        for path_coords, cluster, odp_index, path_length in cable_connections:
            path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path_coords]

            folium.PolyLine(
                path_coords,
                color='black',
                weight=2,
                opacity=0.8,
                popup=f"Cluster {cluster}, ODP {odp_index}<br>Panjang Kabel: {path_length:.2f} meter"
            ).add_to(mymap)

        # Simpan peta
        map_file = os.path.join(app.config['UPLOAD_FOLDER'], 'map_with_cables.html')
        mymap.save(map_file)
        return os.path.relpath(map_file, app.config['UPLOAD_FOLDER'])
    except Exception as e:
        print(f"Error saat visualisasi kabel dengan jalan: {e}")
        return None


from flask import send_file
import io

# Variabel global untuk menyimpan data akhir
uploaded_data = None
clustered_data = None
odp_data = None
cable_data = None
total_cable_length = 0
final_data = {}

@app.route('/hitung_kabel', methods=['POST'])
def hitung_kabel():
    global clustered_data, odp_data, cable_data, total_cable_length

    if clustered_data is None:
        return jsonify({"success": False, "message": "Data belum di-cluster atau ODP belum ditentukan."})

    try:
        # Pastikan ODP dihitung dan `odp_points` dihasilkan
        map_file, odp_points, house_to_odp_mapping = process_odp_and_generate_map(clustered_data)

        if not odp_points:
            return jsonify({"success": False, "message": "Gagal menentukan titik ODP."})

        # Hitung panjang kabel dan jalur
        total_length, cable_connections, house_to_odp_lengths, G = calculate_cable_per_house(
            clustered_data, odp_points, house_to_odp_mapping
        )

        if not G:
            raise ValueError("Graf jalan tidak ditemukan.")

        if not cable_connections or not house_to_odp_lengths:
            raise ValueError("Koneksi kabel atau data panjang kabel kosong.")

        # Simpan data ODP dan kabel ke variabel global
        odp_data = pd.DataFrame(odp_points)
        cable_data = pd.DataFrame(house_to_odp_lengths)
        total_cable_length = total_length

        # Visualisasikan hasil pada peta
        map_file = visualize_house_to_odp_connections(clustered_data, odp_points, cable_connections, house_to_odp_lengths, G)

        if map_file:
            return jsonify({
                "success": True,
                "total_length": total_length,  # Menambahkan total panjang kabel
                "map_file": map_file
            })
        else:
            return jsonify({"success": False, "message": "Gagal menyimpan peta hasil."})
    except Exception as e:
        print(f"Error di endpoint /hitung_kabel: {e}")
        return jsonify({"success": False, "message": f"Error: {e}"})

#download
@app.route('/download_results', methods=['GET'])
def download_results():
    global clustered_data, odp_data, cable_data, total_cable_length

    # Pastikan data sudah tersedia
    if clustered_data is None or odp_data is None or cable_data is None:
        return jsonify({"success": False, "message": "Proses belum selesai. Harap jalankan proses terlebih dahulu."})

    try:
        # Simpan hasil ke dalam file Excel
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "Final_Results.xlsx")

        # Menyimpan hasil clustering, ODP, dan kabel ke dalam file Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            clustered_data.to_excel(writer, sheet_name="Clustering", index=False)
            odp_data.to_excel(writer, sheet_name="ODP", index=False)
            cable_data.to_excel(writer, sheet_name="Cable Length", index=False)

            # Menambahkan sheet dengan total panjang kabel
            summary_df = pd.DataFrame({
                "Total Cable Length": [total_cable_length]
            })
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Periksa apakah file Excel berhasil dibuat
        if not os.path.exists(file_path):
            raise ValueError("File Excel gagal dibuat.")

        # Kirim file sebagai respons
        return send_file(
            file_path,
            as_attachment=True,
            download_name="Final_Results.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        print(f"Error saat membuat file Excel: {e}")
        return jsonify({"success": False, "message": str(e)})

#silhouette
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import numpy as np

def find_optimal_clusters(data, max_clusters=10):
    try:
        # Pisahkan latitude & longitude dari kolom 'MAPS'
        data[['latitude', 'longitude']] = data['MAPS'].str.split(' ', expand=True)
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        data = data.dropna(subset=['latitude', 'longitude'])

        if data.empty:
            raise ValueError("Data latitude dan longitude tidak valid atau kosong.")

        data_matrix = data[['latitude', 'longitude']].values

        # Hitung jarak rata-rata antar titik untuk mempertimbangkan sebaran data
        dist_matrix = distance_matrix(data_matrix, data_matrix)
        avg_distance = np.mean(dist_matrix)
        max_distance = np.max(dist_matrix)

        best_score = -1
        optimal_clusters = 2
        scores = {}

        # Tetapkan jumlah cluster minimum berdasarkan jumlah data
        min_clusters = min(2, len(data) // 10)  # Minimal 1 cluster per 10 data

        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_matrix)
            labels = kmeans.labels_
            score = silhouette_score(data_matrix, labels)

            scores[n_clusters] = score

            # Perbaikan logika pemilihan cluster:
            # 1. Jika Silhouette Score lebih tinggi, pilih cluster itu
            # 2. Jika jarak antar titik besar, buat lebih banyak cluster
            if score > best_score:
                best_score = score
                optimal_clusters = n_clusters
            elif avg_distance > 0.1 and score > 0.5:
                optimal_clusters = max(optimal_clusters, n_clusters)
            
            # 3. Jika jarak terjauh dalam data sangat besar, naikkan jumlah cluster
            if max_distance > 0.5 and n_clusters > optimal_clusters:
                optimal_clusters = n_clusters

        # Interpretasi kualitas clustering berdasarkan Silhouette Score
        if best_score >= 0.7:
            evaluation = "Sangat Bagus (Cluster sangat terpisah)"
        elif best_score >= 0.5:
            evaluation = "Baik (Cluster cukup terpisah)"
        elif best_score >= 0.2:
            evaluation = "Kurang Baik (Beberapa cluster tumpang tindih)"
        else:
            evaluation = "Buruk (Cluster tidak terpisah dengan baik)"

        print(f"Cluster optimal: {optimal_clusters} dengan silhouette score {best_score:.4f} ({evaluation})")

        return optimal_clusters, best_score, evaluation

    except Exception as e:
        print(f"Error saat mencari jumlah cluster optimal: {e}")
        return 2, -1, "Error"



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
