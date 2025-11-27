# app_desktop_pyqt6.py
import sys
import os
import math
import tempfile
import threading
from random import uniform, randint, seed

import folium
import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import LineString
from sklearn.ensemble import RandomForestRegressor

from PyQt6.QtCore import QUrl, Qt, pyqtSlot, QObject
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox, QSlider, QGroupBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel

# reproducibility
seed(42)
np.random.seed(42)
ox.settings.log_console = False
ox.settings.use_cache = True


# -----------------------------------------
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def edges_to_dataframe(G):
    rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get("highway", "")
        if isinstance(highway, list):
            highway = highway[0]
        rows.append({
            "u": u, "v": v, "key": k,
            "length": data.get("length", 0.0),
            "speed_kph": data.get("speed_kph", 40.0),
            "highway": highway,
            "geometry": data.get("geometry", None),
            "center_x": (G.nodes[u]['x'] + G.nodes[v]['x']) / 2.0,
            "center_y": (G.nodes[u]['y'] + G.nodes[v]['y']) / 2.0
        })
    return pd.DataFrame(rows)


def build_and_train_model(_G, edges_df, sample_edges=1500, n_anchor_pairs=30):
    G = _G
    if len(edges_df) > sample_edges:
        ed_sample = edges_df.sample(sample_edges, random_state=42).reset_index(drop=True)
    else:
        ed_sample = edges_df.copy().reset_index(drop=True)

    nodes = list(G.nodes)
    anchors = [(nodes[np.random.randint(0, len(nodes))], nodes[np.random.randint(0, len(nodes))]) for _ in
               range(n_anchor_pairs)]

    X_list = []
    y_list = []
    for hour in range(24):
        for (a, b) in anchors:
            a_xy = (G.nodes[a]['y'], G.nodes[a]['x'])
            b_xy = (G.nodes[b]['y'], G.nodes[b]['x'])
            for idx, r in ed_sample.iterrows():
                length = r["length"]
                speed = r["speed_kph"] if not np.isnan(r["speed_kph"]) else 40.0
                road_type_score = 0
                hwy = r["highway"]
                if isinstance(hwy, str):
                    if "motorway" in hwy:
                        road_type_score = 3
                    elif "trunk" in hwy:
                        road_type_score = 2
                    elif "primary" in hwy:
                        road_type_score = 2
                    elif "secondary" in hwy:
                        road_type_score = 1
                    elif "residential" in hwy:
                        road_type_score = 0
                dist_to_a = haversine_meters(r["center_y"], r["center_x"], a_xy[0], a_xy[1])
                dist_to_b = haversine_meters(r["center_y"], r["center_x"], b_xy[0], b_xy[1])
                base = 0.2 + 0.05 * road_type_score
                rush = 0.0
                rush += 0.5 * math.exp(-((hour - 9) ** 2) / (2 * 3 ** 2))
                rush += 0.5 * math.exp(-((hour - 18) ** 2) / (2 * 3 ** 2))
                prox_effect = 0.0
                prox_effect += 0.6 * math.exp(-dist_to_a / 200.0)
                prox_effect += 0.4 * math.exp(-dist_to_b / 200.0)
                noise = np.random.normal(0, 0.05)
                target = float(np.clip(base + 0.6 * rush + prox_effect + noise, 0.1, 1.0))
                feat = [length, speed, road_type_score, math.sin(2 * math.pi * hour / 24.0),
                        math.cos(2 * math.pi * hour / 24.0), dist_to_a, dist_to_b]
                X_list.append(feat)
                y_list.append(target)
    X = np.array(X_list)
    y = np.array(y_list)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def make_features_for_prediction(row, hour, src_xy, dst_xy):
    length = row["length"]
    speed = row["speed_kph"] if not np.isnan(row["speed_kph"]) else 40.0
    road_type_score = 0
    hwy = row["highway"]
    if isinstance(hwy, str):
        if "motorway" in hwy:
            road_type_score = 3
        elif "trunk" in hwy:
            road_type_score = 2
        elif "primary" in hwy:
            road_type_score = 2
        elif "secondary" in hwy:
            road_type_score = 1
        elif "residential" in hwy:
            road_type_score = 0
    dist_to_a = haversine_meters(row["center_y"], row["center_x"], src_xy[0], src_xy[1])
    dist_to_b = haversine_meters(row["center_y"], row["center_x"], dst_xy[0], dst_xy[1])
    feat = [length, speed, road_type_score, math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
            dist_to_a, dist_to_b]
    return np.array(feat)


# -----------------------------------------
def generate_synthetic_weights(G, edges_df, center_lat, center_lon,
                               hotspots_count=3, r_min=50, r_max=300, base_prob=0.02,
                               center_boost=2.5, max_chain=4):
    n = len(edges_df)
    base = np.random.uniform(0.05, 0.4, size=n)
    centers_dist = edges_df.apply(lambda r: haversine_meters(r["center_y"], r["center_x"], center_lat, center_lon),
                                  axis=1).values
    center_boost_vals = center_boost * np.exp(-centers_dist / 1500.0)
    prob = np.clip(base_prob + center_boost_vals, 0.001, 0.6)

    hotspots = []
    meters_per_deg_lat = 111139
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))
    for _ in range(hotspots_count):
        dist_m = uniform(200, 2000) ** 0.9
        angle = uniform(0, 2 * math.pi)
        dy, dx = dist_m * math.sin(angle), dist_m * math.cos(angle)
        dlat, dlon = dy / meters_per_deg_lat, dx / meters_per_deg_lon
        lat, lon = center_lat + dlat, center_lon + dlon
        radius = uniform(r_min, r_max)
        hotspots.append({"lat": lat, "lon": lon, "radius": radius})

    hotspot_add = np.zeros(n)
    for h in hotspots:
        dists = edges_df.apply(lambda r: haversine_meters(r["center_y"], r["center_x"], h["lat"], h["lon"]),
                               axis=1).values
        inside = dists <= h["radius"]
        if inside.any():
            hotspot_add[inside] += 0.6 * (1 - dists[inside] / h["radius"])

    synthetic = np.clip(base + hotspot_add, 0.0, 1.2)
    seeds_mask = np.random.rand(n) < prob
    seed_indices = np.nonzero(seeds_mask)[0]

    adjacency = [[] for _ in range(n)]
    for idx, row in edges_df.iterrows():
        u, v = int(row["u"]), int(row["v"])
        outgoing = edges_df[edges_df["u"] == v]
        for _, out_row in outgoing.iterrows():
            adjacency[idx].append(int(out_row.name))

    for s in seed_indices:
        length = randint(1, max_chain)
        current = s
        decay = 1.0
        for _ in range(length):
            synthetic[current] += min(0.5 * decay, 1.5 - synthetic[current])
            if adjacency[current]:
                current = int(np.random.choice(adjacency[current]))
                decay *= 0.6
            else:
                break
    return np.clip(synthetic, 0.0, 1.5), hotspots


# -----------------------------------------
class MapClickReceiver(QObject):
    """Receiver exposed to JS via QWebChannel"""
    def __init__(self, app_window):
        super().__init__()
        self.app = app_window
        self.state = 0  # 0 -> next click sets source, 1 -> next sets dest

    @pyqtSlot(float, float)
    def map_clicked(self, lat, lon):
        # Called from JS
        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            return
        if self.state == 0:
            self.app.src_lat.setText(f"{lat:.6f}")
            self.app.src_lon.setText(f"{lon:.6f}")
            self.app.set_status("Source selected by click")
            self.state = 1
        else:
            self.app.dst_lat.setText(f"{lat:.6f}")
            self.app.dst_lon.setText(f"{lon:.6f}")
            self.app.set_status("Destination selected by click")
            self.state = 0
        # re-render map so markers appear
        self.app._render_map()


class TrafficSimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic sim — Dijkstra + ML (PyQt6)")
        self.resize(1200, 800)

        self.G = None
        self.edges_df = None
        self.model = None
        self.synthetic_mult = None
        self.hotspots = None

        self.map_html = os.path.join(tempfile.gettempdir(), "traffic_map.html")

        # UI build and webchannel setup
        self._build_ui()
        # Create and register receiver for JS clicks
        self.channel = QWebChannel()
        self.map_receiver = MapClickReceiver(self)
        self.channel.registerObject("pyReceiver", self.map_receiver)
        self.view.page().setWebChannel(self.channel)

    def _build_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        ctrl = QVBoxLayout()
        layout.addLayout(ctrl, 0)

        box_map = QGroupBox("Map / Graph")
        vmap = QVBoxLayout()
        box_map.setLayout(vmap)
        ctrl.addWidget(box_map)

        self.center_lat_input = QDoubleSpinBox()
        self.center_lat_input.setDecimals(6)
        self.center_lat_input.setRange(-90, 90)
        self.center_lat_input.setValue(44.85)
        self.center_lon_input = QDoubleSpinBox()
        self.center_lon_input.setDecimals(6)
        self.center_lon_input.setRange(-180, 180)
        self.center_lon_input.setValue(65.50)
        self.radius_km_input = QSpinBox()
        self.radius_km_input.setRange(5, 60)
        self.radius_km_input.setValue(20)

        vmap.addWidget(QLabel("Center lat:"))
        vmap.addWidget(self.center_lat_input)
        vmap.addWidget(QLabel("Center lon:"))
        vmap.addWidget(self.center_lon_input)
        vmap.addWidget(QLabel("Radius (km):"))
        vmap.addWidget(self.radius_km_input)

        btn_load = QPushButton("Load graph (OSMnx)")
        btn_load.clicked.connect(self.on_load_graph)
        vmap.addWidget(btn_load)

        # hotspots
        box_hot = QGroupBox("Hotspots / Synthetic traffic")
        vhot = QVBoxLayout()
        box_hot.setLayout(vhot)
        ctrl.addWidget(box_hot)

        self.hot_count = QSpinBox()
        self.hot_count.setRange(0, 10)
        self.hot_count.setValue(3)
        self.hot_rmin = QSpinBox()
        self.hot_rmin.setRange(10, 2000)
        self.hot_rmin.setValue(50)
        self.hot_rmax = QSpinBox()
        self.hot_rmax.setRange(20, 5000)
        self.hot_rmax.setValue(300)
        self.hour_slider = QSlider(Qt.Orientation.Horizontal)
        self.hour_slider.setRange(0, 23)
        self.hour_slider.setValue(9)

        vhot.addWidget(QLabel("Hotspots count:"))
        vhot.addWidget(self.hot_count)
        vhot.addWidget(QLabel("Hotspot radius min (m):"))
        vhot.addWidget(self.hot_rmin)
        vhot.addWidget(QLabel("Hotspot radius max (m):"))
        vhot.addWidget(self.hot_rmax)
        vhot.addWidget(QLabel("Hour of day:"))
        vhot.addWidget(self.hour_slider)

        btn_gen = QPushButton("Generate synthetic traffic")
        btn_gen.clicked.connect(self.on_generate_synthetic)
        vhot.addWidget(btn_gen)

        # Route inputs
        box_route = QGroupBox("Route")
        vroute = QVBoxLayout()
        box_route.setLayout(vroute)
        ctrl.addWidget(box_route)

        self.src_lat = QLineEdit()
        self.src_lon = QLineEdit()
        self.dst_lat = QLineEdit()
        self.dst_lon = QLineEdit()
        vroute.addWidget(QLabel("Source lat:"))
        vroute.addWidget(self.src_lat)
        vroute.addWidget(QLabel("Source lon:"))
        vroute.addWidget(self.src_lon)
        vroute.addWidget(QLabel("Dest lat:"))
        vroute.addWidget(self.dst_lat)
        vroute.addWidget(QLabel("Dest lon:"))
        vroute.addWidget(self.dst_lon)

        self.btn_build_model = QPushButton("Build & train ML model")
        self.btn_build_model.clicked.connect(self.on_build_model)
        vroute.addWidget(self.btn_build_model)

        self.btn_route = QPushButton("Compute route (Dijkstra + ML + Synthetic)")
        self.btn_route.clicked.connect(self.on_compute_route)
        vroute.addWidget(self.btn_route)

        # Web view
        self.view = QWebEngineView()

        # WebEngine settings
        page = self.view.page()
        settings = page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)

        # Debug prints
        self.view.loadStarted.connect(lambda: print("WebEngine: Load started"))
        self.view.loadProgress.connect(lambda p: print(f"WebEngine: Load progress {p}%"))
        self.view.loadFinished.connect(lambda ok: print(f"WebEngine: Load finished, success={ok}"))

        layout.addWidget(self.view, 1)

        self.status = QLabel("Ready")
        ctrl.addWidget(self.status)
        ctrl.addStretch(1)

        # initial small html
        self._test_webview()

    def set_status(self, text):
        self.status.setText(text)
        QApplication.processEvents()

    def _test_webview(self):
        test_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { 
                    font-family: Arial; 
                    padding: 50px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-align: center;
                }
                h1 { font-size: 48px; }
            </style>
        </head>
        <body>
            <h1>🗺️ Traffic Simulator</h1>
            <p style="font-size: 20px;">Load a graph to see the map here</p>
        </body>
        </html>
        """
        self.view.setHtml(test_html)
        print("Test HTML loaded into WebEngineView")

    # -------------------------
    def on_load_graph(self):
        lat = float(self.center_lat_input.value())
        lon = float(self.center_lon_input.value())
        radius = int(self.radius_km_input.value() * 1000)
        self.set_status("Loading graph from OSM (this may take a few seconds)...")
        QApplication.processEvents()

        try:
            print(f"Loading graph at ({lat}, {lon}) with radius {radius}m")
            G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")

            print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
            print(f"Original CRS: {G.graph.get('crs', 'unknown')}")

            # Если граф не в lat/lon, проецируем его
            if G.graph.get('crs') != 'epsg:4326':
                print("Converting to EPSG:4326...")
                G = ox.project_graph(G, to_crs='epsg:4326')

            G = ox.distance.add_edge_lengths(G)
            G = ox.add_edge_speeds(G, fallback=40)

            for u, v, k, data in G.edges(keys=True, data=True):
                speed = data.get("speed_kph", 40.0)
                length = data.get("length", 0.0)
                travel_time = length * 3.6 / max(speed, 0.1)
                G[u][v][k]["travel_time"] = travel_time

            self.G = G
            self.edges_df = edges_to_dataframe(G)

            self.set_status(f"Graph loaded. Nodes: {len(G.nodes)}, Edges: {len(self.edges_df)}")
            print(f"Sample node: {list(G.nodes(data=True))[0]}")

            # render map
            self._render_map()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error loading graph", str(e))
            self.set_status("Error loading graph")

    def on_build_model(self):
        if self.G is None or self.edges_df is None:
            QMessageBox.warning(self, "No graph", "Load graph first")
            return
        self.set_status("Training ML model...")
        QApplication.processEvents()

        def job():
            try:
                self.model = build_and_train_model(self.G, self.edges_df)
                self.set_status("Model trained")
            except Exception as e:
                self.set_status("Model training error")
                QMessageBox.critical(self, "Model error", str(e))

        threading.Thread(target=job, daemon=True).start()

    def on_generate_synthetic(self):
        if self.G is None or self.edges_df is None:
            QMessageBox.warning(self, "No graph", "Load graph first")
            return
        lat, lon = float(self.center_lat_input.value()), float(self.center_lon_input.value())
        count, rmin, rmax = int(self.hot_count.value()), int(self.hot_rmin.value()), int(self.hot_rmax.value())
        self.set_status("Generating synthetic traffic...")
        QApplication.processEvents()
        synthetic, hotspots = generate_synthetic_weights(self.G, self.edges_df, lat, lon,
                                                         hotspots_count=count, r_min=rmin, r_max=rmax)
        self.synthetic_mult = synthetic
        self.hotspots = hotspots
        self.set_status("Synthetic traffic generated")
        self._render_map()

    def _predict_model_on_edges(self, hour, src_node, dst_node):
        node_xy = {n: (data['y'], data['x']) for n, data in self.G.nodes(data=True)}
        src_xy, dst_xy = node_xy[src_node], node_xy[dst_node]
        feats = [make_features_for_prediction(row, hour, src_xy, dst_xy) for _, row in self.edges_df.iterrows()]
        X = np.vstack(feats)
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.5)

    def on_compute_route(self):
        if self.G is None or self.edges_df is None:
            QMessageBox.warning(self, "No graph", "Load graph first")
            return
        if self.model is None:
            QMessageBox.warning(self, "No model", "Train ML model first")
            return
        try:
            src_lat, src_lon = float(self.src_lat.text()), float(self.src_lon.text())
            dst_lat, dst_lon = float(self.dst_lat.text()), float(self.dst_lon.text())
        except Exception:
            QMessageBox.warning(self, "Coords error", "Enter valid numeric coordinates")
            return

        src_node = ox.nearest_nodes(self.G, src_lon, src_lat)
        dst_node = ox.nearest_nodes(self.G, dst_lon, dst_lat)
        hour = int(self.hour_slider.value())
        self.set_status("Predicting model on edges...")
        QApplication.processEvents()

        preds = self._predict_model_on_edges(hour, src_node, dst_node)
        synthetic = self.synthetic_mult if self.synthetic_mult is not None else np.zeros(len(self.edges_df))
        combined_mult = np.clip(preds + synthetic, 0.0, 2.5)

        for i, row in self.edges_df.iterrows():
            u, v, k = int(row["u"]), int(row["v"]), int(row["key"])
            if u in self.G and v in self.G and k in self.G[u][v]:
                base_tt = self.G[u][v][k].get("travel_time", row["length"] * 3.6 / max(row["speed_kph"], 0.1))
                mult = float(combined_mult[i])
                self.G[u][v][k]["pred_mult"] = mult
                self.G[u][v][k]["pred_cost"] = base_tt * (1 + mult)

        self.set_status("Searching shortest path...")
        QApplication.processEvents()
        try:
            path = nx.shortest_path(self.G, source=src_node, target=dst_node, weight="pred_cost")
            self.set_status("Route found — rendering map")
            self._render_map(route=path, show_pred=True)
        except nx.NetworkXNoPath:
            QMessageBox.warning(self, "No path", "No path found")
            self.set_status("No path")
        except Exception as e:
            QMessageBox.critical(self, "Routing error", str(e))
            self.set_status("Routing error")

    def _render_map(self, route=None, show_pred=False):
        try:
            # Determine center
            if self.G is None or len(self.G.nodes) == 0:
                center_lat, center_lon = float(self.center_lat_input.value()), float(self.center_lon_input.value())
            else:
                lats = [data['y'] for _, data in self.G.nodes(data=True)]
                lons = [data['x'] for _, data in self.G.nodes(data=True)]
                center_lat = sum(lats) / len(lats)
                center_lon = sum(lons) / len(lons)

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                control_scale=True,
                prefer_canvas=True,
                tiles='OpenStreetMap'
            )

            # Draw all edges (light gray)
            edges_drawn = 0
            if self.edges_df is not None:
                for i, row in self.edges_df.iterrows():
                    u, v = int(row["u"]), int(row["v"])
                    if u not in self.G.nodes or v not in self.G.nodes:
                        continue
                    u_node, v_node = self.G.nodes[u], self.G.nodes[v]
                    if row["geometry"] is not None:
                        coords = [(lat_, lon_) for lon_, lat_ in row["geometry"].coords]
                    else:
                        coords = [(u_node['y'], u_node['x']), (v_node['y'], v_node['x'])]
                    folium.PolyLine(coords, color='gray', weight=2, opacity=0.7).add_to(m)
                    edges_drawn += 1

            # Hotspots
            if self.hotspots:
                for h in self.hotspots:
                    folium.Circle(
                        location=[h["lat"], h["lon"]],
                        radius=h["radius"],
                        color='orange',
                        fill=True,
                        fillColor='orange',
                        fillOpacity=0.2,
                        weight=2,
                        opacity=0.8
                    ).add_to(m)

            # Synthetic traffic overlay
            if self.synthetic_mult is not None:
                thr = 0.25
                for i, row in self.edges_df.iterrows():
                    mult = float(self.synthetic_mult[i])
                    if mult > thr:
                        u, v = int(row["u"]), int(row["v"])
                        if u not in self.G.nodes or v not in self.G.nodes:
                            continue
                        u_node, v_node = self.G.nodes[u], self.G.nodes[v]
                        if row["geometry"] is not None:
                            coords = [(lat_, lon_) for lon_, lat_ in row["geometry"].coords]
                        else:
                            coords = [(u_node['y'], u_node['x']), (v_node['y'], v_node['x'])]
                        if mult > 1.0:
                            color = 'darkred'; width = 5
                        elif mult > 0.5:
                            color = 'red'; width = 4
                        else:
                            color = 'orange'; width = 3
                        folium.PolyLine(coords, color=color, weight=width, opacity=0.9).add_to(m)

            # Route overlay
            if show_pred and route is not None:
                route_coords = []
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    data = self.G.get_edge_data(u, v)
                    if not data:
                        continue
                    k = list(data.keys())[0]
                    geom = data[k].get("geometry")
                    if geom is not None:
                        route_coords.extend([(lat_, lon_) for lon_, lat_ in geom.coords])
                    else:
                        u_node, v_node = self.G.nodes[u], self.G.nodes[v]
                        route_coords.extend([(u_node['y'], u_node['x']), (v_node['y'], v_node['x'])])
                if route_coords:
                    folium.PolyLine(route_coords, color='blue', weight=6, opacity=0.9, tooltip="Route").add_to(m)

                # start/end markers
                if route:
                    start_node = self.G.nodes[route[0]]
                    end_node = self.G.nodes[route[-1]]
                    folium.Marker(
                        location=[start_node['y'], start_node['x']],
                        popup='Start',
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(m)
                    folium.Marker(
                        location=[end_node['y'], end_node['x']],
                        popup='End',
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(m)

                # per-edge predicted multiplier markers
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    data = self.G.get_edge_data(u, v)
                    if not data:
                        continue
                    k = list(data.keys())[0]
                    mult = data[k].get("pred_mult", 0.0)
                    nu, nv = self.G.nodes[u], self.G.nodes[v]
                    mid_lat = (nu['y'] + nv['y']) / 2.0
                    mid_lon = (nu['x'] + nv['x']) / 2.0
                    folium.CircleMarker(
                        location=[mid_lat, mid_lon],
                        radius=4 + mult * 5,
                        color='purple',
                        fill=True,
                        fillColor='purple',
                        fillOpacity=0.7,
                        weight=2,
                        tooltip=f"Traffic multiplier: {mult:.2f}"
                    ).add_to(m)

            # Show src/dst markers if set via inputs (or clicks)
            try:
                if self.src_lat.text() and self.src_lon.text():
                    folium.Marker(
                        [float(self.src_lat.text()), float(self.src_lon.text())],
                        popup='Selected Source',
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(m)
                if self.dst_lat.text() and self.dst_lon.text():
                    folium.Marker(
                        [float(self.dst_lat.text()), float(self.dst_lon.text())],
                        popup='Selected Destination',
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(m)
            except Exception:
                pass

            # Insert JS for click handling via QWebChannel
            map_var = m.get_name()  # folium map var, e.g. "map_123456"
            click_js = f"""
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
            <script>
                (function() {{
                    // Try to get map variable created by folium
                    var mapObj = window['{map_var}'] || null;
                    function setupChannel() {{
                        if (typeof QWebChannel === 'undefined') {{
                            console.warn("QWebChannel not ready");
                            return;
                        }}
                        new QWebChannel(qt.webChannelTransport, function(channel) {{
                            window.pyReceiver = channel.objects.pyReceiver;
                        }});
                    }}
                    if (typeof qt !== "undefined" && qt.webChannelTransport) {{
                        setupChannel();
                    }} else {{
                        document.addEventListener("DOMContentLoaded", setupChannel);
                    }}

                    function onMapClick(e) {{
                        if (window.pyReceiver && mapObj) {{
                            window.pyReceiver.map_clicked(e.latlng.lat, e.latlng.lng);
                        }}
                    }}
                    // wait until map exists
                    var tries = 0;
                    function attach() {{
                        mapObj = window['{map_var}'] || mapObj;
                        if (mapObj && mapObj.on) {{
                            mapObj.on('click', onMapClick);
                        }} else {{
                            tries++;
                            if (tries < 50) setTimeout(attach, 200);
                        }}
                    }}
                    attach();
                }})();
            </script>
            """
            m.get_root().html.add_child(folium.Element(click_js))

            # Save and load
            m.save(self.map_html)
            file_url = QUrl.fromLocalFile(os.path.abspath(self.map_html))
            self.view.setUrl(file_url)
            self.set_status("Map rendered")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.set_status(f"Map rendering error: {e}")


# -----------------------------------------
def main():
    # Settings to make WebEngine more permissive when loading local files
    os.environ['QTWEBENGINE_DISABLE_SANDBOX'] = '1'
    os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-web-security --allow-file-access-from-files'

    app = QApplication(sys.argv)
    win = TrafficSimulatorApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
