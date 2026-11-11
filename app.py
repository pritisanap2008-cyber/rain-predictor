import streamlit as st          # builds the web page
import requests                 # asks websites for data
import pandas as pd             # tables
import numpy as np              # maths
import lightgbm as lgb          # loads our trained model
import altair as alt   # chart library that comes with streamlit

st.set_page_config(page_title="Rain Predictor")   # sets the browser tab title
st.markdown("""
<style>
.stApp {                                          /* the whole app background */
    background: linear-gradient(180deg, #0B1320 0%, #1B2A49 55%, #0B1320 100%) !important;  /* dark stormy sky, navy to blue to navy */
}

.stApp::before, .stApp::after {                   /* two invisible layers that will hold the rain */
    content: "";                                  /* needed so the layer exists */
    position: fixed;                              /* stays in place when you scroll */
    top: 0; left: 0; width: 100%; height: 100%;   /* covers the whole screen */
    pointer-events: none;                         /* lets your mouse click through the rain */
    z-index: 0;                                   /* sits behind your content */
}

.stApp::before {                                  /* rain layer 1: bigger drops */
    background: repeating-linear-gradient(90deg, transparent 0 37px, rgba(174,194,224,0.55) 37px 39px);  /* thin vertical lines every 39 pixels */
    -webkit-mask-image: linear-gradient(180deg, #000 0 40px, transparent 40px 130px);  /* cuts each line into short pieces, so they look like streaks */
    mask-image: linear-gradient(180deg, #000 0 40px, transparent 40px 130px);          /* same thing for other browsers */
    -webkit-mask-size: 100% 130px;                /* one pattern of 130 pixels high */
    mask-size: 100% 130px;                        /* same thing for other browsers */
    animation: rain1 0.8s linear infinite;        /* moves the streaks down forever */
}

.stApp::after {                                   /* rain layer 2: smaller and softer drops */
    background: repeating-linear-gradient(90deg, transparent 0 52px, rgba(174,194,224,0.3) 52px 53px);  /* thinner lines, different spacing */
    -webkit-mask-image: linear-gradient(180deg, #000 0 60px, transparent 60px 170px);
    mask-image: linear-gradient(180deg, #000 0 60px, transparent 60px 170px);
    -webkit-mask-size: 100% 170px;
    mask-size: 100% 170px;
    animation: rain2 1.2s linear infinite;        /* slower than layer 1, so it looks natural */
}

@keyframes rain1 {                                /* the movement of layer 1 */
    from { -webkit-mask-position: 0 0; mask-position: 0 0; }          /* start position */
    to   { -webkit-mask-position: 0 130px; mask-position: 0 130px; }  /* move down by one pattern, then repeat */
}
@keyframes rain2 {                                /* the movement of layer 2 */
    from { -webkit-mask-position: 0 0; mask-position: 0 0; }
    to   { -webkit-mask-position: 0 170px; mask-position: 0 170px; }
}

.block-container {                                /* your app content area */
    position: relative;                           /* needed for z-index to work */
    z-index: 1;                                   /* puts the content in front of the rain */
    max-width: 850px;                             /* keeps the page from getting too wide */
}

h1 {                                              /* the big title */
    background: linear-gradient(90deg, #4FC3F7, #B388FF);   /* blue to purple colour */
    -webkit-background-clip: text;                /* colour shows only inside the letters */
    -webkit-text-fill-color: transparent;         /* makes the letters see-through so the colour shows */
    font-weight: 800;                             /* extra bold */
}

[data-testid="stMetric"] {                        /* the box around the big percentage */
    background: rgba(255,255,255,0.07);           /* faint glass colour */
    border: 1px solid rgba(255,255,255,0.15);     /* thin light border */
    border-radius: 18px;                          /* round corners */
    padding: 18px 24px;                           /* space inside the box */
    backdrop-filter: blur(8px);                   /* blurs the rain behind the box, like frosted glass */
}

[data-testid="stMetricValue"] {                   /* the big number */
    font-size: 3.2rem;                            /* makes it large */
}

.stButton > button {                              /* the Predict button */
    background: linear-gradient(90deg, #4FC3F7, #7C4DFF);   /* blue to purple */
    color: white;                                 /* white text */
    border: none;                                 /* no border */
    border-radius: 12px;                          /* round corners */
    padding: 0.6rem 2.2rem;                       /* bigger button */
    font-weight: 600;                             /* semi-bold text */
}

footer {visibility: hidden;}                      /* hides the small "Made with Streamlit" text */
</style>
""", unsafe_allow_html=True)                      # puts this style into the page; without unsafe_allow_html the CSS would show as plain text
st.title("Rain predictor")                         # big title on the page
st.write("Chance of rain in the next 3 hours at any location.")  # short description

lat = st.number_input("Latitude", value=28.61, min_value=-90.0, max_value=90.0, format="%.4f")      # input box for latitude
lon = st.number_input("Longitude", value=77.21, min_value=-180.0, max_value=180.0, format="%.4f")   # input box for longitude

RAIN_LIMIT = 0.1   # same rain-hour cutoff we used in training

@st.cache_resource                                  # load the model once, not on every click
def load_model():
    return lgb.Booster(model_file="rain_model.txt") # reads the saved model file

@st.cache_data(ttl=900)                             # remember the downloaded data for 15 minutes
def get_recent_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"  # Open-Meteo live data service
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,precipitation",  # same 5 measurements as training
        "past_days": 2,                             # include the last 2 days (we need hours before now)
        "forecast_days": 1,                         # plus today
        "timezone": "auto",                         # local time of the chosen place
    }
    r = requests.get(url, params=params, timeout=30)  # send the request
    r.raise_for_status()                            # stop with an error if the request failed
    data = r.json()                                 # reply as a dictionary
    df = pd.DataFrame(data["hourly"])               # table of hourly values
    df["time"] = pd.to_datetime(df["time"])         # text to real dates
    df = df.set_index("time")                       # time as the row label
    return df, data["utc_offset_seconds"]           # also return the local time offset from UTC

def make_features(df):                              # same feature recipe as Step 4
    feat = pd.DataFrame(index=df.index)
    feat["pressure"] = df["surface_pressure"]
    feat["humidity"] = df["relative_humidity_2m"]
    feat["temp"] = df["temperature_2m"]
    feat["dew_gap"] = df["temperature_2m"] - df["dew_point_2m"]
    for h in [1, 3, 6, 12]:
        feat[f"pressure_change_{h}h"] = df["surface_pressure"].diff(h)
    for h in [1, 3, 6]:
        feat[f"humidity_change_{h}h"] = df["relative_humidity_2m"].diff(h)
    feat["temp_change_3h"] = df["temperature_2m"].diff(3)
    feat["dew_gap_change_3h"] = feat["dew_gap"].diff(3)
    rain_now = (df["precipitation"] >= RAIN_LIMIT).astype(int)
    feat["rain_last_3h"] = rain_now.rolling(3).sum()
    feat["rain_last_6h"] = rain_now.rolling(6).sum()
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return feat

model = load_model()                                # load the trained model

if st.button("Predict", type="primary"):                           # runs when the button is clicked
    df, offset = get_recent_weather(lat, lon)       # download recent hours
    now_local = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(seconds=offset)  # current time at the chosen place
    df = df.loc[:now_local.floor("h")]              # keep only hours up to now (no future data)
    feats = make_features(df).dropna()              # build features, remove rows with gaps
    row = feats.iloc[[-1]][model.feature_name()]    # latest row, columns in the exact order the model learned
    prob = float(model.predict(row)[0])             # probability of rain in the next 3 hours

    st.metric("Chance of rain in next 3 hours", f"{prob * 100:.0f}%")   # big number
    if prob >= 0.5:
        st.warning("Rain is likely.")               # shown when probability is 50% or more
    else:
        st.success("Rain is unlikely.")             # shown otherwise

    st.caption(f"Based on data up to {feats.index[-1]} (local time).")  # which hour was used
    st.subheader("Recent pressure (hPa)")
    chart = alt.Chart(df.reset_index()).mark_line().encode(   # draw a line chart with the altair library
    x="time:T",                                           # time on the horizontal axis
    y=alt.Y("surface_pressure:Q", scale=alt.Scale(zero=False)),  # pressure on the vertical axis, zoomed to the real range (not starting from 0)
)
    st.altair_chart(chart, use_container_width=True)          # show it, full width
    st.subheader("Recent humidity (%)")
    st.line_chart(df["relative_humidity_2m"])       # humidity graph

st.divider()                                        # a line to separate sections
st.caption("Model trained on historical hourly data. It is a learning project, not an official forecast.")  # honest note
