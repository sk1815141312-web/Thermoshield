🛡️ ThermoShield AI — Heat-Aware Navigation

FortyGuard AI Challenge Hackathon'26 Submission · Interactive Maps Track

ThermoShield AI is a navigation app that doesn't just ask "how fast can I get there?" — it asks "which route exposes me to less heat, and when should I travel?". It compares real driving routes on live heat-risk exposure, not just speed, and recommends the safest realistic option.

🔗 Live Demo

https://mzjxbr6q4h.streamlit.app

⚡ Quick Start (No Setup Needed)

Open the live demo above and tap any Demo Mode button (e.g. "Phoenix → Houston") for an instant, fully working example — no login or API key required.

🧰 Run It Locally
bash
git clone https://github.com/sk1815141312-web/Thermoshield.git
cd Thermoshield
pip install -r requirements.txt
streamlit run app.py

Open the local URL Streamlit prints (usually http://localhost:8501).

Optional: enable extra live data sources

The app runs fully without any keys (using Open-Meteo for weather and a rule-based engine for recommendations). To enable the optional live upgrades, create .streamlit/secrets.toml in the project folder:

toml
FORTYGUARD_API_KEY = "your_fortyguard_temperature_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
FortyGuard Temperature API — used for U.S. locations only (its supported coverage area); the app automatically falls back to Open-Meteo everywhere else, or if this key isn't set.
Gemini API — used for the natural-language AI Recommendation and "Ask ThermoShield" chat; the app automatically falls back to a deterministic rule-based engine if this key isn't set.
🌡️ How It Works
Geocoding — Nominatim (OpenStreetMap) resolves the entered city names to coordinates.
Routing — OSRM computes up to 3 real driving routes (Direct Highway, Shaded Boulevard, Eco/Green Corridor).
Weather — Open-Meteo (and optionally FortyGuard's Temperature API in the U.S.) supplies live air temperature.
Thermal Risk Score — computed per the challenge formula: Surface Temp × 0.6 + Air Temp × 0.4, where Surface Temp is estimated from air temperature plus a route-type offset (Critical > 42°C, Warning 35–42°C, Safe < 35°C).
Smart Departure Advisor — projects the next 12 hours of forecasted risk and recommends the coolest realistic departure window.
Cooling shelters & road segments — real OpenStreetMap places and OSRM road-name data, never invented locations or numbers.

Full data-source breakdown is available in the app's own Methodology & Data Sources panel.

🏗️ Tech Stack

Streamlit · Folium · OSRM · Nominatim · Open-Meteo · FortyGuard Temperature API (optional) · Gemini API (optional)

📁 Files
app.py — the complete application
requirements.txt — Python dependencies

Built for FortyGuard Hackathon'26 — Building the World's Temperature AI.
