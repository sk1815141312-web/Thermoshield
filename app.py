"""
ThermoShield AI - Heat-Aware Navigation Application
=====================================================
FortyGuard AI Challenge Hackathon Submission

A Streamlit application that calculates optimal driving routes by balancing:
  - Thermal Risk Score (heat exposure)
  - Real driving distance (km)
  - Estimated Time of Arrival (ETA)

Tech Stack:
  - Streamlit (UI)
  - Folium + streamlit-folium (interactive maps)
  - OSRM API (real highway routing)
  - Nominatim API (geocoding)
  - Open-Meteo API (live air temperature, no key required)

Run with:  streamlit run app.py

Requirements (requirements.txt):
  streamlit
  requests
  folium
  streamlit-folium
  pandas
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL CONSTANTS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ThermoShield AI | Heat-Aware Navigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL_TEMPLATE = "http://router.project-osrm.org/route/v1/driving/{coords}"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Custom User-Agent is REQUIRED by Nominatim's usage policy.
HEADERS = {"User-Agent": "ThermoShieldAI-Hackathon/1.0 (contact: hackathon@fortyguard.ai)"}

# Route archetypes: name, surface-heat offset over ambient air temp (°C),
# and synthetic distance/duration multipliers used ONLY as a fallback when
# OSRM does not return enough real alternative routes.
ROUTE_PROFILES = [
    {"name": "Direct Highway",      "icon": "🛣️", "surface_offset": 9.0, "dist_mult": 1.00, "dur_mult": 1.00, "color": None},
    {"name": "Shaded Boulevard",    "icon": "🌳", "surface_offset": 4.5, "dist_mult": 1.12, "dur_mult": 1.15, "color": None},
    {"name": "Eco/Green Corridor",  "icon": "🍃", "surface_offset": 1.0, "dist_mult": 1.26, "dur_mult": 1.32, "color": None},
]

RISK_THRESHOLDS = {"critical": 42.0, "warning": 35.0}
RISK_COLORS = {"Critical": "#FF3B3B", "Warning": "#FFA726", "Safe": "#2ECC71"}


# ---------------------------------------------------------------------------
# DARK / CYBERSECURITY THEME CSS
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0B0F19;
            color: #E6EDF3;
        }
        section[data-testid="stSidebar"] {
            background-color: #0D1321;
            border-right: 1px solid #1F2937;
        }
        div[data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid #1F2A3C;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            color: #8FA3BF !important;
        }
        h1, h2, h3 {
            color: #00E5FF;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        .thermo-banner {
            background: linear-gradient(90deg, #001B2E 0%, #0B0F19 100%);
            border: 1px solid #00E5FF33;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 18px;
        }
        .advisory-box {
            background-color: #0F1B2B;
            border-left: 4px solid #00E5FF;
            border-radius: 6px;
            padding: 14px 18px;
            margin-top: 10px;
            font-size: 0.95rem;
            line-height: 1.5em;
        }
        .stButton>button {
            background-color: #00E5FF;
            color: #001B2E;
            font-weight: 700;
            border: none;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #33ecff;
            color: #001B2E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# GEOCODING (robust, prevents Nominatim from returning tiny streets/POIs)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(raw_query: str):
    """
    Resolves a place name into precise (lat, lon) coordinates.

    Fixes the classic 'Dubai -> tiny street center' bug by:
      1. Preferring structured queries biased toward city/settlement features.
      2. Filtering Nominatim results to place/boundary classes (cities, towns,
         administrative regions) instead of accepting the first raw hit.
      3. Ranking surviving candidates by Nominatim's own 'importance' score.
      4. Falling back through progressively looser queries if the strict
         attempt returns nothing.

    Returns a dict {lat, lon, display_name} or None if resolution fails.
    """
    query = raw_query.strip()
    if not query:
        return None

    # Ordered fallback strategies: most restrictive -> most permissive.
    attempts = [
        {"q": query, "featuretype": "city"},
        {"q": query, "featuretype": "settlement"},
        {"q": query, "featuretype": "state"},
        {"q": query},  # last resort: unrestricted search
    ]

    preferred_types = {
        "city", "town", "village", "administrative", "municipality",
        "state", "county", "hamlet", "region",
    }
    preferred_classes = {"place", "boundary"}

    for extra_params in attempts:
        params = {
            "format": "json",
            "limit": 6,
            "addressdetails": 1,
            **extra_params,
        }
        try:
            resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            results = resp.json()
        except (requests.RequestException, ValueError):
            continue

        if not results:
            continue

        # Prefer proper places (cities/towns/regions) over shops, streets, POIs.
        filtered = [
            r for r in results
            if r.get("type") in preferred_types or r.get("class") in preferred_classes
        ]
        candidates = filtered if filtered else results

        # Rank by Nominatim's importance score (higher = more globally significant).
        candidates.sort(key=lambda r: float(r.get("importance", 0.0)), reverse=True)
        best = candidates[0]

        try:
            lat, lon = float(best["lat"]), float(best["lon"])
        except (KeyError, ValueError, TypeError):
            continue

        return {
            "lat": lat,
            "lon": lon,
            "display_name": best.get("display_name", query),
        }

        # Being polite to the free Nominatim endpoint between attempts.
    return None


# ---------------------------------------------------------------------------
# OSRM ROUTING
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_osrm_routes(origin_lat, origin_lon, dest_lat, dest_lon, alternatives=True):
    """
    Calls the public OSRM demo server for real driving directions.

    Returns a list of route dicts, each containing:
      - distance_m, duration_s
      - geometry: list of [lon, lat] coordinate pairs (GeoJSON order)

    Raises a RuntimeError with a user-friendly message on failure.
    """
    coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    url = OSRM_URL_TEMPLATE.format(coords=coords)
    params = {
        "overview": "full",
        "geometries": "geojson",
        "alternatives": "true" if alternatives else "false",
        "steps": "false",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the OSRM routing service: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("OSRM returned an unreadable response.") from exc

    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(
            "OSRM could not find a drivable road route between these two points. "
            "They may be separated by water, unmapped roads, or an invalid location match."
        )

    routes = []
    for r in data["routes"]:
        routes.append({
            "distance_m": r["distance"],
            "duration_s": r["duration"],
            "geometry": r["geometry"]["coordinates"],  # [[lon, lat], ...]
        })

    # Sort fastest first so index 0 is always the "Direct Highway" baseline.
    routes.sort(key=lambda r: r["duration_s"])
    return routes


# ---------------------------------------------------------------------------
# LIVE WEATHER (Open-Meteo — free, no API key)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def get_air_temperature(lat, lon):
    """Fetches current ambient air temperature (°C) for a coordinate."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data["current"]["temperature_2m"])
    except (requests.RequestException, KeyError, ValueError, TypeError):
        # Safe fallback if the weather service is unreachable — avoids crashing
        # the whole app; clearly flagged to the user in the UI.
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_hourly_forecast(lat, lon, hours_ahead=12):
    """
    Fetches an hourly air-temperature forecast for the next `hours_ahead` hours
    at the given coordinate. Powers the Smart Departure Advisor — the core
    predictive feature that tells the driver WHEN to leave, not just WHICH road.
    Returns a list of {time: datetime, air_temp: float}, or [] on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "forecast_days": 2,
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return []

    now = datetime.now()
    forecast = []
    for t_str, temp in zip(times, temps):
        try:
            t = datetime.fromisoformat(t_str)
        except ValueError:
            continue
        if t >= now.replace(minute=0, second=0, microsecond=0) and temp is not None:
            forecast.append({"time": t, "air_temp": float(temp)})
        if len(forecast) >= hours_ahead:
            break

    return forecast


# ---------------------------------------------------------------------------
# THERMAL RISK ENGINE
# ---------------------------------------------------------------------------

def classify_risk(score: float) -> str:
    if score > RISK_THRESHOLDS["critical"]:
        return "Critical"
    elif score >= RISK_THRESHOLDS["warning"]:
        return "Warning"
    return "Safe"


def thermal_risk_score(surface_temp: float, air_temp: float) -> float:
    """Core formula mandated by the challenge spec."""
    return round((surface_temp * 0.6) + (air_temp * 0.4), 1)


def compute_departure_advisory(hourly_forecast, surface_offset, current_air_temp):
    """
    Projects the Thermal Risk Score for a given route across the next several
    hours (using each hour's forecast air temp + the route's fixed surface-heat
    offset), and identifies the coolest realistic departure window.

    This is what makes ThermoShield predictive rather than just descriptive:
    instead of only comparing roads right now, it tells the driver WHEN to go.
    """
    if not hourly_forecast:
        return None

    scored = []
    for entry in hourly_forecast:
        score = thermal_risk_score(entry["air_temp"] + surface_offset, entry["air_temp"])
        scored.append({
            "time": entry["time"],
            "air_temp": round(entry["air_temp"], 1),
            "score": score,
            "category": classify_risk(score),
        })

    current_score = thermal_risk_score(current_air_temp + surface_offset, current_air_temp)
    best = min(scored, key=lambda s: s["score"])

    return {
        "hourly": scored,
        "best": best,
        "current_score": current_score,
        "current_category": classify_risk(current_score),
        "savings": round(current_score - best["score"], 1),
    }


def build_route_options(osrm_routes, air_temp):
    """
    Merges real OSRM route data with the 3 ThermoShield route archetypes,
    computing thermal exposure for each. If OSRM returns fewer real
    alternatives than needed, missing ones are estimated from the primary
    route using conservative distance/duration multipliers and flagged
    as 'Estimated' in the UI.
    """
    baseline = osrm_routes[0]
    options = []

    for i, profile in enumerate(ROUTE_PROFILES):
        if i < len(osrm_routes):
            r = osrm_routes[i]
            distance_km = r["distance_m"] / 1000.0
            duration_min = r["duration_s"] / 60.0
            geometry = r["geometry"]
            is_estimated = False
        else:
            distance_km = (baseline["distance_m"] / 1000.0) * profile["dist_mult"]
            duration_min = (baseline["duration_s"] / 60.0) * profile["dur_mult"]
            geometry = baseline["geometry"]
            is_estimated = True

        surface_temp = round(air_temp + profile["surface_offset"], 1)
        score = thermal_risk_score(surface_temp, air_temp)
        category = classify_risk(score)

        options.append({
            "name": f"{profile['icon']} {profile['name']}",
            "raw_name": profile["name"],
            "distance_km": round(distance_km, 1),
            "duration_min": round(duration_min, 1),
            "surface_temp": surface_temp,
            "air_temp": round(air_temp, 1),
            "thermal_score": score,
            "risk_category": category,
            "risk_color": RISK_COLORS[category],
            "geometry": geometry,
            "estimated": is_estimated,
        })

    return options


# ---------------------------------------------------------------------------
# MAP RENDERING
# ---------------------------------------------------------------------------

def build_map(origin, destination, route_options, selected_name):
    """Builds a dark-themed Folium map with all 3 route polylines drawn,
    auto-fitted so it works equally well for local and international trips."""

    all_lats = [origin["lat"], destination["lat"]]
    all_lons = [origin["lon"], destination["lon"]]

    fmap = folium.Map(location=[origin["lat"], origin["lon"]], zoom_start=8, tiles="CartoDB dark_matter")

    for opt in route_options:
        coords_latlon = [[lat, lon] for lon, lat in opt["geometry"]]
        all_lats.extend([c[0] for c in coords_latlon])
        all_lons.extend([c[1] for c in coords_latlon])

        is_selected = opt["raw_name"] == selected_name
        folium.PolyLine(
            locations=coords_latlon,
            color=opt["risk_color"],
            weight=7 if is_selected else 4,
            opacity=0.95 if is_selected else 0.45,
            tooltip=(
                f"{opt['name']} | {opt['distance_km']} km | {opt['duration_min']:.0f} min | "
                f"Thermal Score: {opt['thermal_score']}°C ({opt['risk_category']})"
            ),
        ).add_to(fmap)

    # Origin / Destination markers
    folium.Marker(
        [origin["lat"], origin["lon"]],
        tooltip=f"Origin: {origin['display_name']}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fmap)

    folium.Marker(
        [destination["lat"], destination["lon"]],
        tooltip=f"Destination: {destination['display_name']}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(fmap)

    # Dynamic zoom/center — works for both local (few km) and international (100s of km) trips.
    south, north = min(all_lats), max(all_lats)
    west, east = min(all_lons), max(all_lons)
    fmap.fit_bounds([[south, west], [north, east]], padding=(30, 30))

    return fmap


# ---------------------------------------------------------------------------
# AI ADVISORY MESSAGE
# ---------------------------------------------------------------------------

def generate_advisory(route_options, selected_name):
    fastest = min(route_options, key=lambda r: r["duration_min"])
    coolest = min(route_options, key=lambda r: r["thermal_score"])
    selected = next(r for r in route_options if r["raw_name"] == selected_name)

    if fastest["raw_name"] == coolest["raw_name"]:
        return (
            f"✅ **{selected['name']}** is both the fastest and coolest option available right now — "
            f"no trade-off needed. Thermal exposure is {selected['thermal_score']}°C "
            f"({selected['risk_category']})."
        )

    extra_time = round(coolest["duration_min"] - fastest["duration_min"], 1)
    heat_saved = round(fastest["thermal_score"] - coolest["thermal_score"], 1)

    if selected["raw_name"] == coolest["raw_name"]:
        return (
            f"🛡️ You're on the **coolest** route. Compared to **{fastest['name']}**, you're spending "
            f"**{extra_time:.0f} extra minutes** on the road but avoiding **{heat_saved}°C** of surface "
            f"heat exposure — a solid trade for passenger comfort and safety."
        )
    elif selected["raw_name"] == fastest["raw_name"]:
        return (
            f"⚡ You're on the **fastest** route. Switching to **{coolest['name']}** would cost "
            f"**{extra_time:.0f} extra minutes** but reduce thermal exposure by **{heat_saved}°C** "
            f"(down to {coolest['risk_category']} risk)."
        )
    else:
        return (
            f"⚖️ **{selected['name']}** is a balanced middle-ground: {selected['duration_min']:.0f} min ETA "
            f"and {selected['thermal_score']}°C thermal exposure ({selected['risk_category']}). "
            f"The coolest option, **{coolest['name']}**, saves an extra {heat_saved}°C but adds "
            f"{extra_time:.0f} more minutes versus the fastest route."
        )


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    inject_custom_css()

    st.markdown(
        """
        <div class="thermo-banner">
            <h1 style="margin-bottom:0;">🛡️ ThermoShield AI</h1>
            <p style="color:#8FA3BF; margin-top:4px;">
                Heat-aware navigation — balancing real-world routing, live weather, and thermal safety.
                Built for the FortyGuard AI Challenge.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Sidebar controls ----------------
    with st.sidebar:
        st.header("🧭 Route Planner")
        origin_query = st.text_input("Origin", placeholder="e.g. Dubai")
        dest_query = st.text_input("Destination", placeholder="e.g. Abu Dhabi")
        calculate = st.button("🔍 Calculate Thermal-Safe Routes", use_container_width=True)
        st.markdown("---")
        st.caption(
            "Thermal Risk = (Surface Temp × 0.6) + (Air Temp × 0.4)\n\n"
            "🟢 Safe < 35°C  🟠 Warning 35–42°C  🔴 Critical > 42°C"
        )

    if "route_options" not in st.session_state:
       
