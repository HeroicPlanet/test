# app.py
import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from streamlit_folium import st_folium
import folium
from shapely.geometry import Point, LineString
from random import uniform, randint, seed

# --- Настройки ---
ox.settings.log_console = False
ox.settings.use_cache = True
st.set_page_config(layout="wide", page_title="Маршрут: Dijkstra + ML (Kyzylorda)")

# фиксируем seed для воспроизводимости (можно убрать)
seed(42)
np.random.seed(42)


# ----------------- Вспомогательные функции -----------------
@st.cache_resource(show_spinner=True)
def load_graph(center_point=(44.85, 65.50), dist_m=20000):
    # Загружаем дорожный граф
    G = ox.graph_from_point(center_point, dist=dist_m, network_type="drive")

    # Добавляем длину рёбер
    G = ox.distance.add_edge_lengths(G)

    # Добавляем скорости (если нет, ставим 40 км/ч)
    G = ox.add_edge_speeds(G, fallback=40)

    # Добавляем время проезда
    for u, v, k, data in G.edges(keys=True, data=True):
        speed = data.get("speed_kph", 40.0)
        length = data.get("length", 0.0)
        # travel_time = (длина [м] / скорость [км/ч]) * 3600 / 1000
        travel_time = length * 3.6 / max(speed, 0.1)
        G[u][v][k]["travel_time"] = travel_time

    return G


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


def haversine_meters(lat1, lon1, lat2, lon2):
    # approximate haversine
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ----------------- Генерация пробок (hotspots) -----------------
def generate_hotspots(center_lat, center_lon, count=3, r_min=10, r_max=50):
    """
    ИЗМЕНЕНО: Генерирует точки в 'кольце' 3-5 км от центра.
    """
    hotspots = []

    # Приблизительные метры на градус
    meters_per_deg_lat = 111139
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(center_lat))

    for _ in range(count):
        # 1. Выбираем случайное расстояние (3-5 км) и угол
        dist_m = uniform(3000, 5000)  # 3-5 км в метрах
        angle_rad = uniform(0, 2 * math.pi)

        # 2. Конвертируем полярные в декартовы (смещения в метрах)
        dy_meters = dist_m * math.sin(angle_rad)
        dx_meters = dist_m * math.cos(angle_rad)

        # 3. Конвертируем смещения в метрах в смещения в градусах
        dlat = dy_meters / meters_per_deg_lat
        dlon = dx_meters / meters_per_deg_lon

        # 4. Новые координаты
        lat = center_lat + dlat
        lon = center_lon + dlon

        radius_m = uniform(r_min, r_max)
        hotspots.append({"lat": lat, "lon": lon, "radius": radius_m})
    return hotspots


def apply_hotspots_to_edges(edges_df, hotspots):
    # базовый вес случайно 0.1 - 0.6
    base_weights = np.random.uniform(0.1, 0.6, size=len(edges_df))
    total = base_weights.copy()
    # для каждого hotspot добавляем вес к ребрам в радиусе (чем ближе - тем сильнее)
    for h in hotspots:
        # расстояния от центра ребра до hotspot (метры)
        dists = edges_df.apply(
            lambda r: haversine_meters(r["center_y"], r["center_x"], h["lat"], h["lon"]),
            axis=1
        )
        # для ребер в радиусе добавим вклад (линейно убывающий)
        within = dists <= h["radius"]
        # если строго внутри — add up to 0.4, scaled by proximity
        add = np.zeros(len(edges_df))
        if within.any():
            add[within] = 0.4 * (1 - dists[within] / h["radius"])
        total += add
    # clamp to 0.1 .. 1.0
    total = np.clip(total, 0.1, 1.0)
    return total


# ----------------- Создание синтетической обучающей выборки и обучение -----------------
@st.cache_resource(show_spinner=True)
def build_and_train_model(_G, edges_df, sample_edges=1500, n_anchor_pairs=30):
    G = _G
    if len(edges_df) > sample_edges:
        ed_sample = edges_df.sample(sample_edges, random_state=42).reset_index(drop=True)
    else:
        ed_sample = edges_df.copy().reset_index(drop=True)

    # создаём "якорные" пары (A,B), чтобы модель училась учитывать dist_to_A/B
    nodes = list(G.nodes)
    anchors = []
    for _ in range(n_anchor_pairs):
        a = nodes[np.random.randint(0, len(nodes))]
        b = nodes[np.random.randint(0, len(nodes))]
        anchors.append((a, b))

    X_list = []
    y_list = []
    # симуляция: для каждого anchor и часа создаём целевой weight
    for hour in range(24):
        for (a, b) in anchors:
            a_xy = (G.nodes[a]['y'], G.nodes[a]['x'])  # lat, lon
            b_xy = (G.nodes[b]['y'], G.nodes[b]['x'])
            # подсчёт target weight как смесь road_type, hour and proximity to anchors
            for idx, r in ed_sample.iterrows():
                # features
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
                # distances (meters)
                dist_to_a = haversine_meters(r["center_y"], r["center_x"], a_xy[0], a_xy[1])
                dist_to_b = haversine_meters(r["center_y"], r["center_x"], b_xy[0], b_xy[1])
                # synthetic target: base 0.2 + effect by road_type + rush hour peaks + proximity to anchors
                base = 0.2 + 0.05 * road_type_score
                # simulate rush hours (9am and 18pm)
                rush = 0.0
                rush += 0.5 * math.exp(-((hour - 9) ** 2) / (2 * 3 ** 2))
                rush += 0.5 * math.exp(-((hour - 18) ** 2) / (2 * 3 ** 2))
                # proximity effect: if edge is near either A or B, add congestion (people starting/ending trips)
                prox_effect = 0.0
                prox_effect += 0.6 * math.exp(-dist_to_a / 200.0)  # strong near A
                prox_effect += 0.4 * math.exp(-dist_to_b / 200.0)
                noise = np.random.normal(0, 0.05)
                target = base + 0.6 * rush + prox_effect + noise
                target = float(np.clip(target, 0.1, 1.0))
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


# ----------------- ИЗМЕНЕНО: Поиск пути (Dijkstra + ML) -----------------
def calculate_route(G, source, target, hour, model, edges_df):
    """
    ИЗМЕНЕНО:
    1. Эта функция теперь сперва вычисляет ВСЕ веса для ML-модели.
    2. Затем она использует nx.shortest_path (Dijkstra) для поиска
       оптимального пути, используя эти веса.
    Это решает проблему "тупого" жадного алгоритма.
    """

    node_xy = {n: (data['y'], data['x']) for n, data in G.nodes(data=True)}  # lat, lon
    src_xy = node_xy[source]
    dst_xy = node_xy[target]

    # 1. Рассчитываем признаки и предсказания для ВСЕХ ребер в графе
    feats = []
    idxs = []
    for i, row in edges_df.iterrows():
        feat = make_features_for_prediction(row, hour, src_xy, dst_xy)
        feats.append(feat)
        idxs.append((int(row["u"]), int(row["v"]), int(row["key"])))

    X = np.vstack(feats)
    preds = model.predict(X)
    preds = np.clip(preds, 0.1, 1.0)

    # 2. Применяем предсказанные веса (pred_cost) к графу G
    # Это "стоимость" проезда по ребру с учетом пробок
    for i, uvk in enumerate(idxs):
        u, v, k = uvk
        if u in G and v in G and k in G[u][v]:
            base_tt = G[u][v][k].get("travel_time",
                                     edges_df.loc[i, "length"] * 3.6 / max(edges_df.loc[i, "speed_kph"], 0.1))
            pred_mult = float(preds[i])
            # Итоговая стоимость = базовое время * (1 + вес пробки)
            pred_cost = base_tt * (1 + pred_mult)
            G[u][v][k]["pred_mult"] = pred_mult
            G[u][v][k]["pred_cost"] = pred_cost

    # 3. ИСПОЛЬЗУЕМ DIJKSTRA (встроенный) для поиска пути
    # Он найдет оптимальный путь, используя вес "pred_cost"
    try:
        path = nx.shortest_path(G, source=source, target=target, weight="pred_cost")
        return path
    except nx.NetworkXNoPath:
        st.error("Путь между точками не найден (нет связи в графе).")
        return None
    except Exception as e:
        st.error(f"Ошибка при поиске пути (Dijkstra): {e}")
        return None


def route_geometry(G, route):
    points = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        data = G.get_edge_data(u, v)
        if not data:
            continue
        # Берем первое попавшееся ребро (на случай параллельных)
        k = list(data.keys())[0]
        geom = data[k].get("geometry", None)
        if geom is None:
            # Если геометрии нет (редко), просто берем узлы
            points.append((G.nodes[u]['y'], G.nodes[u]['x']))
            points.append((G.nodes[v]['y'], G.nodes[v]['x']))
        else:
            # Используем геометрию ребра
            for coord in geom.coords:
                points.append((coord[1], coord[0]))  # (lat, lon)
    if not points:
        return None

    # Создаем LineString из (lon, lat) пар
    ls = LineString([(lon, lat) for lat, lon in points])
    return ls


# ----------------- Streamlit UI -----------------
st.title("Оптимизация маршрута — Dijkstra + ML (Kyzylorda)")

with st.sidebar:
    st.markdown("### Параметры карты/симуляции")
    center_lat = st.number_input("Center lat", value=44.85, format="%.6f")
    center_lon = st.number_input("Center lon", value=65.50, format="%.6f")
    radius_km = st.number_input("Radius (km)", value=20, min_value=5, max_value=60)
    hotspots_count = st.slider("Число пробок (hotspots)", 1, 5, value=3)
    hotspot_radius_min = st.number_input("Min радиус пробки (m)", value=10, min_value=1)
    hotspot_radius_max = st.number_input("Max радиус пробки (m)", value=50, min_value=10)
    hour = st.slider("Час суток для предсказания", 0, 23, 9)
    if st.button("Перезагрузить граф (очистить кэш)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

# load graph
G = load_graph(center_point=(center_lat, center_lon), dist_m=int(radius_km * 1000))
edges_df = edges_to_dataframe(G)

# hotspots
hotspots = generate_hotspots(center_lat, center_lon, count=hotspots_count, r_min=hotspot_radius_min,
                             r_max=hotspot_radius_max)
# Это "симуляция пробок" для UI. Она НЕ используется в ML-модели.
synthetic_weights = apply_hotspots_to_edges(edges_df, hotspots)

# build & train model (cached)
with st.spinner("Обучаем ML модель на синтетике (несколько секунд)..."):
    model = build_and_train_model(G, edges_df)

# Map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)
folium.TileLayer("cartodbpositron").add_to(m)

# ----------------- ИЗМЕНЕНО: Рисуем пробки линиями -----------------
# (Эти пробки - симуляция для UI, они НЕ используются ML-моделью для маршрута)
edges_df["synthetic_weight"] = synthetic_weights
# Возьмем ребра с весом > 0.6 (0.1 - 1.0)
congested_edges = edges_df[edges_df["synthetic_weight"] > 0.6]

# Стиль для красных линий
congestion_style = lambda x: {
    'color': 'red',
    'weight': 4,
    'opacity': 0.7
}

for idx, row in congested_edges.iterrows():
    geom = row.get("geometry")
    if geom:
        folium.GeoJson(
            geom.__geo_interface__,
            style_function=congestion_style,
            tooltip=f"Пробка (симуляция) (вес {row['synthetic_weight']:.2f})"
        ).add_to(m)
# ----------------- Конец блока пробок -----------------


st.markdown("Кликните на карту, чтобы поставить точки: 1) старт 2) цель.")
if "clicks" not in st.session_state:
    st.session_state["clicks"] = []

output = st_folium(m, width=900, height=600, returned_objects=["last_clicked"])
last_clicked = output.get("last_clicked", None)
if last_clicked:
    pt = (last_clicked["lat"], last_clicked["lng"])
    if len(st.session_state["clicks"]) == 0 or (abs(st.session_state["clicks"][-1][0] - pt[0]) > 1e-6 or abs(
            st.session_state["clicks"][-1][1] - pt[1]) > 1e-6):
        st.session_state["clicks"].append(pt)
    if len(st.session_state["clicks"]) > 2:
        st.session_state["clicks"] = st.session_state["clicks"][-2:]
if st.button("Очистить маркеры"):
    st.session_state["clicks"] = []

if st.session_state["clicks"]:
    st.markdown("#### Выбранные точки:")
    for i, pt in enumerate(st.session_state["clicks"]):
        st.write(f"{i + 1}. {pt}")

# если есть две точки - строим маршрут
if len(st.session_state["clicks"]) >= 2:
    src_pt = st.session_state["clicks"][0]
    dst_pt = st.session_state["clicks"][1]
    src_node = ox.nearest_nodes(G, src_pt[1], src_pt[0])
    dst_node = ox.nearest_nodes(G, dst_pt[1], dst_pt[0])
    st.write("Найденные узлы:", src_node, dst_node)

    # ----------------- ИЗМЕНЕНО: Вызов функции -----------------
    try:
        # Теперь функция называется calculate_route
        route = calculate_route(G, src_node, dst_node, hour, model, edges_df)
        used_method = "Dijkstra + ML"  # Более точное название
    except Exception as e:
        st.error(f"Ошибка при вызове 'calculate_route': {e}")
        route = None
        used_method = "Error"
    # ----------------- Конец блока -----------------

    if route:
        # покажем метки и маршрут на карте
        folium.Marker(location=src_pt, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(location=dst_pt, tooltip="Goal", icon=folium.Icon(color="red")).add_to(m)
        route_geom = route_geometry(G, route)
        if route_geom:
            folium.GeoJson(route_geom.__geo_interface__, name="route", tooltip="Маршрут").add_to(m)

        # Перерисовываем карту с маршрутом
        st_folium(m, width=900, height=600)

        # подсчёт длины, времени, средней maxspeed
        total_seconds = 0.0
        total_meters = 0.0
        speeds = []
        for u, v in zip(route[:-1], route[1:]):
            data = G.get_edge_data(u, v)
            if not data: continue
            k = list(data.keys())[0]
            # Суммируем pred_cost, т.к. это и есть прогнозируемое время
            total_seconds += data[k].get("pred_cost", data[k].get("travel_time", 0.0))
            total_meters += data[k].get("length", 0.0)
            speeds.append(data[k].get("speed_kph", 40.0))

        avg_speed = np.mean(speeds) if speeds else 0.0
        st.success(
            f"Маршрут ({used_method}) — длина ≈ {total_meters / 1000:.2f} km, прогнозируемое время ≈ {total_seconds / 60:.1f} min")
        st.info(f"Средняя maxspeed по маршруту ≈ {avg_speed:.1f} km/h (усреднение maxspeed рёбер)")

        # показать распределение predicted weights по пройденным ребрам
        weights = [G[u][v][list(G[u][v].keys())[0]].get("pred_mult", None) for u, v in zip(route[:-1], route[1:])]
        if any(w is not None for w in weights):
            st.write("Predicted traffic weights (по ребрам в маршруте):")
            st.write([round(float(w), 3) if w is not None else None for w in weights])

        # download GeoJSON
        if route_geom:
            gj = route_geom.__geo_interface__
            st.download_button("Скачать маршрут (GeoJSON)", data=str(gj), file_name="route.geojson",
                               mime="application/json")
    else:
        st.warning("Не удалось построить маршрут.")

st.markdown("---")
st.markdown("### Примечания")
st.markdown("""
- "Weight" у нас в диапазоне 0.1..1.0 — модель предсказывает его; итоговый мультипликатор времени = 1 + weight.
- Если maxspeed отсутствует — используется 40 km/h.
- ML обучается на синтетических данных (см. код). При наличии реальных данных пробок (CSV или API) модель легко заменяется / дообучается.
- **ИЗМЕНЕНО:** Используется Dijkstra (вместо Greedy) для поиска оптимального пути по весам, предсказанным ML.
""")