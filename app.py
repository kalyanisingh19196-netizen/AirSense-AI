from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import os
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AirSense AI | Environmental Intelligence", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------- DESIGN ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#07110f;--panel:#101e19;--line:#294238;--muted:#9ab2a6;--text:#edf7f1;--mint:#7ce5b2;--teal:#43cbbb}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 80% 0%,#17392d 0%,#07110f 48%);color:var(--text)}
[data-testid="stHeader"]{background:rgba(7,17,15,.78)}
[data-testid="stSidebar"]{background:#0a1712;border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:var(--text)}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1550px}
.hero{background:linear-gradient(115deg,rgba(25,65,48,.96),rgba(14,37,33,.92));border:1px solid #35634e;border-radius:24px;padding:25px 29px}
.hero h1{margin:.25rem 0;color:#f2fff7}
.hero p{color:#b6d0c2;margin:0;max-width:900px}
.eyebrow{color:#7ce5b2;text-transform:uppercase;letter-spacing:.15em;font-size:.72rem;font-weight:700}
.metric{background:linear-gradient(145deg,#162b22,#0d1b16);border:1px solid #315343;border-radius:18px;padding:16px;min-height:120px}
.metric-label{color:#a8c2b4;font-size:.75rem;font-weight:700}
.metric-value{color:#f1fff7;font-family:'Space Grotesk';font-size:1.9rem;font-weight:700;margin:.3rem 0}
.metric-foot{color:#8bd6ad;font-size:.75rem}
.panel{background:rgba(16,30,25,.85);border:1px solid #294238;border-radius:18px;padding:18px}
.stButton>button,.stDownloadButton>button{background:#174332;color:#effff5;border:1px solid #397b5b;border-radius:11px;font-weight:700}
div[data-testid="stMetric"]{background:#14271f;border:1px solid #315343;padding:12px;border-radius:14px}
.small-muted{color:#9ab2a6;font-size:.82rem}
</style>
""", unsafe_allow_html=True)

CITY_COORDS = {
"Agra":(27.1767,78.0081),"Ahmedabad":(23.0225,72.5714),"Amritsar":(31.6340,74.8723),
"Bengaluru":(12.9716,77.5946),"Bhopal":(23.2599,77.4126),"Bhubaneswar":(20.2961,85.8245),
"Chandigarh":(30.7333,76.7794),"Chennai":(13.0827,80.2707),"Coimbatore":(11.0168,76.9558),
"Delhi":(28.6139,77.2090),"Dehradun":(30.3165,78.0322),"Faridabad":(28.4089,77.3178),
"Ghaziabad":(28.6692,77.4538),"Gurugram":(28.4595,77.0266),"Guwahati":(26.1445,91.7362),
"Hyderabad":(17.3850,78.4867),"Indore":(22.7196,75.8577),"Jaipur":(26.9124,75.7873),
"Jodhpur":(26.2389,73.0243),"Kanpur":(26.4499,80.3319),"Kochi":(9.9312,76.2673),
"Kolkata":(22.5726,88.3639),"Lucknow":(26.8467,80.9462),"Ludhiana":(30.9010,75.8573),
"Mumbai":(19.0760,72.8777),"Mysuru":(12.2958,76.6394),"Nagpur":(21.1458,79.0882),
"Nashik":(19.9975,73.7898),"Patna":(25.5941,85.1376),"Pune":(18.5204,73.8567),
"Raipur":(21.2514,81.6296),"Ranchi":(23.3441,85.3096),"Surat":(21.1702,72.8311),
"Thiruvananthapuram":(8.5241,76.9366),"Varanasi":(25.3176,82.9739),"Visakhapatnam":(17.6868,83.2185)
}
POLLUTANT_MAP = {"PM2.5":"pm2_5","PM10":"pm10","NO2":"nitrogen_dioxide",
                 "SO2":"sulphur_dioxide","CO":"carbon_monoxide","O3":"ozone"}
UNITS = {"PM2.5":"µg/m³","PM10":"µg/m³","NO2":"µg/m³","SO2":"µg/m³","CO":"µg/m³","O3":"µg/m³"}

def aqi_category(x):
    if pd.isna(x): return "Unavailable"
    if x <= 50: return "Good"
    if x <= 100: return "Satisfactory"
    if x <= 200: return "Moderate"
    if x <= 300: return "Poor"
    if x <= 400: return "Very Poor"
    return "Severe"

def chart_style(fig,height=340):
    fig.update_layout(height=height,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e4f2e9",margin=dict(l=8,r=8,t=40,b=8),
        xaxis=dict(showgrid=False,color="#9ab2a6"),
        yaxis=dict(showgrid=True,gridcolor="rgba(154,178,166,.13)",color="#9ab2a6"),
        legend=dict(orientation="h",y=1.12,x=0))
    return fig

@st.cache_data(ttl=900,show_spinner=False)
def fetch_open_meteo(city,lat,lon):
    """CAMS modelled air-quality time series. Not a ground station observation."""
    url="https://air-quality-api.open-meteo.com/v1/air-quality"
    params={"latitude":lat,"longitude":lon,
        "hourly":"pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,european_aqi",
        "current":"pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,european_aqi",
        "past_days":2,"forecast_days":5,"timezone":"Asia/Kolkata","domains":"cams_global"}
    response=requests.get(url,params=params,timeout=30)
    response.raise_for_status()
    payload=response.json()
    hourly=payload.get("hourly",{})
    if not hourly or not hourly.get("time"): raise ValueError("Open-Meteo returned no hourly data.")
    frame=pd.DataFrame({"Time":pd.to_datetime(hourly["time"])})
    for out_col,api_col in [("PM2.5","pm2_5"),("PM10","pm10"),("NO2","nitrogen_dioxide"),
                            ("SO2","sulphur_dioxide"),("CO","carbon_monoxide"),("O3","ozone"),
                            ("US AQI","us_aqi"),("European AQI","european_aqi")]:
        values=hourly.get(api_col)
        if values is not None: frame[out_col]=values
    current=payload.get("current",{})
    return frame,current,payload.get("hourly_units",{}),payload.get("generationtime_ms")

@st.cache_data(ttl=900,show_spinner=False)
def fetch_cpcb(api_key,limit=1000):
    """Fetch CPCB's OGD real-time AQI resource. Requires a data.gov.in API key."""
    resource_id="3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    url=f"https://api.data.gov.in/resource/{resource_id}"
    params={"api-key":api_key,"format":"json","limit":limit}
    response=requests.get(url,params=params,timeout=35)
    response.raise_for_status()
    payload=response.json()
    return pd.DataFrame(payload.get("records",[])),payload

def metric(label,value,foot):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',unsafe_allow_html=True)

# ----------------------------- SIDEBAR ---------------------------------
with st.sidebar:
    st.markdown("## 🌿 AirSense AI")
    st.caption("REAL-DATA ENVIRONMENTAL INTELLIGENCE")
    st.divider()
    page=st.radio("WORKSPACE",["Command Center","Live Monitor","Forecast Lab","AI Explainability",
        "Pollution Detective","Health Intelligence","Interactive Map","Activity Planner","Data Explorer"])
    city=st.selectbox("SELECT CITY",list(CITY_COORDS),index=list(CITY_COORDS).index("Bengaluru"))
    st.divider()
    cpcb_key=st.text_input("data.gov.in API key (optional)",type="password",
        help="Paste your personal API key from the India Open Government Data portal.")
    st.caption("Open-Meteo model data works without a key. CPCB station data requires a data.gov.in key.")
    st.divider()
    if st.button("Clear cached API data",use_container_width=True):
        st.cache_data.clear()
        st.rerun()

lat,lon=CITY_COORDS[city]
st.markdown('<div class="eyebrow">AIR QUALITY · REAL DATA CONNECTORS</div>',unsafe_allow_html=True)
st.markdown(f'<div class="hero"><h1>{page}</h1><p>{city} · Environmental intelligence workspace · Live API ingestion with explicit source labels.</p></div>',unsafe_allow_html=True)

# Fetch selected city model data
try:
    with st.spinner(f"Fetching Open-Meteo air-quality model data for {city}..."):
        model_df,current,units,generation=fetch_open_meteo(city,lat,lon)
    api_error=None
except Exception as e:
    model_df=pd.DataFrame();current={};units={};generation=None;api_error=str(e)

cpcb_df=pd.DataFrame()
cpcb_error=None
if cpcb_key:
    try:
        with st.spinner("Fetching CPCB observations..."):
            cpcb_df,_=fetch_cpcb(cpcb_key)
    except Exception as e:
        cpcb_error=str(e)

# Select matching CPCB records where possible; preserve original source fields
city_cpcb=pd.DataFrame()
if not cpcb_df.empty and "city" in cpcb_df.columns:
    city_cpcb=cpcb_df[cpcb_df["city"].astype(str).str.casefold()==city.casefold()].copy()

if api_error:
    st.error(f"Open-Meteo API request failed: {api_error}")
if cpcb_error:
    st.error(f"CPCB API request failed: {cpcb_error}")

st.caption("Source labels matter: Open-Meteo is CAMS atmospheric model output, not a CPCB ground-station reading. CPCB records are shown only when the official OGD API key is configured.")

# ----------------------------- COMMAND CENTER ---------------------------
if page=="Command Center":
    st.subheader(f"{city} · air-quality overview")
    if not model_df.empty:
        latest=model_df.iloc[-1]
        # Use current US AQI only as a separately labelled model index, never call it CPCB AQI.
        us_aqi=current.get("us_aqi",latest.get("US AQI",np.nan))
        eu_aqi=current.get("european_aqi",latest.get("European AQI",np.nan))
        cols=st.columns(4)
        with cols[0]: metric("MODELLED US AQI",f"{us_aqi:.0f}" if pd.notna(us_aqi) else "N/A","Open-Meteo / CAMS")
        with cols[1]: metric("PM2.5",f"{latest.get('PM2.5',np.nan):.1f}","µg/m³ · modelled")
        with cols[2]: metric("PM10",f"{latest.get('PM10',np.nan):.1f}","µg/m³ · modelled")
        with cols[3]: metric("EUROPEAN AQI",f"{eu_aqi:.0f}" if pd.notna(eu_aqi) else "N/A","Different from Indian CPCB AQI")
        left,right=st.columns([1.6,1])
        with left:
            fig=px.line(model_df,x="Time",y=["PM2.5","PM10"],title="Hourly particulate matter · CAMS model")
            st.plotly_chart(chart_style(fig,350),use_container_width=True)
        with right:
            val=us_aqi if pd.notna(us_aqi) else 0
            fig=go.Figure(go.Indicator(mode="gauge+number",value=val,
                title={"text":"Modelled US AQI","font":{"color":"#b6d0c2"}},
                gauge={"axis":{"range":[0,500]},"bar":{"color":"#7ce5b2"},
                       "steps":[{"range":[0,50],"color":"#1b583e"},{"range":[50,100],"color":"#647c3d"},
                                {"range":[100,150],"color":"#947c36"},{"range":[150,300],"color":"#98513b"},
                                {"range":[300,500],"color":"#743d4b"}]}))
            fig.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",font_color="#e4f2e9",margin=dict(l=10,r=10,t=45,b=5))
            st.plotly_chart(fig,use_container_width=True)
        st.markdown("#### Multi-pollutant timeline")
        selected=st.multiselect("Select pollutant series",list(POLLUTANT_MAP),default=["PM2.5","PM10","NO2"])
        if selected:
            fig=px.line(model_df,x="Time",y=selected,markers=True,title="Hourly pollutant concentrations")
            st.plotly_chart(chart_style(fig,340),use_container_width=True)
        st.markdown("#### CPCB station observations")
        if not city_cpcb.empty:
            st.dataframe(city_cpcb,use_container_width=True,hide_index=True)
        elif cpcb_key:
            st.info(f"No CPCB records for {city} were returned in the current API page.")
        else:
            st.info("Add a data.gov.in API key in the sidebar to load official CPCB station records.")
    else:
        st.warning("No Open-Meteo data is available right now. Check your internet connection and retry.")

# ----------------------------- LIVE MONITOR ------------------------------
elif page=="Live Monitor":
    st.subheader("Current model conditions and official station readings")
    if current:
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[("PM2.5","pm2_5"),("PM10","pm10"),("NO2","nitrogen_dioxide"),("O3","ozone")]):
            val=current.get(key)
            with col: metric(label,f"{val:.2f}" if isinstance(val,(float,int)) else "N/A","Open-Meteo model · µg/m³")
        st.markdown("#### Hourly live model feed")
        st.dataframe(model_df.tail(30),use_container_width=True,hide_index=True)
    else: st.warning("No current model conditions returned.")
    st.markdown("#### CPCB station feed")
    if not city_cpcb.empty: st.dataframe(city_cpcb,use_container_width=True,hide_index=True)
    elif cpcb_key: st.info("No matching city records returned. Try checking the city spelling or API page limit.")
    else: st.info("Enter your data.gov.in API key in the sidebar to retrieve CPCB station data.")

# ----------------------------- FORECAST LAB -------------------------------
elif page=="Forecast Lab":
    st.subheader("Atmospheric model forecast")
    st.info("This forecast comes from the CAMS global atmospheric-composition model via Open-Meteo. It is not a Random Forest prediction and is not a CPCB station forecast.")
    if not model_df.empty:
        horizon=st.select_slider("Forecast window",options=[24,48,72,96,120],value=72,format_func=lambda h:f"{h} hours")
        cutoff=pd.Timestamp.now().floor("h")+pd.Timedelta(hours=1)
        future=model_df[model_df.Time>=cutoff].head(horizon)
        if future.empty: future=model_df.tail(horizon)
        fig=px.line(future,x="Time",y=["PM2.5","PM10"],markers=True,title="Forecast pollutant concentrations")
        st.plotly_chart(chart_style(fig,390),use_container_width=True)
        if "US AQI" in future:
            fig=px.area(future,x="Time",y="US AQI",title="Modelled US AQI forecast")
            st.plotly_chart(chart_style(fig,300),use_container_width=True)
        st.dataframe(future,use_container_width=True,hide_index=True)
        st.download_button("Download forecast CSV",future.to_csv(index=False),"airsense_openmeteo_forecast.csv","text/csv")
    else: st.warning("Forecast unavailable because the API did not return data.")

# ----------------------------- EXPLAINABILITY -----------------------------
elif page=="AI Explainability":
    st.subheader("Pollutant signal interpretation")
    st.info("This stage visualises measured/modelled pollutant concentrations. SHAP is not shown yet because no validated local forecasting model has been trained on station observations.")
    if not model_df.empty:
        last=model_df.iloc[-1]
        values=pd.DataFrame({"Pollutant":list(POLLUTANT_MAP),
            "Concentration":[last.get(k,np.nan) for k in POLLUTANT_MAP],
            "Unit":["µg/m³"]*6})
        fig=px.bar(values,x="Pollutant",y="Concentration",color="Pollutant",title="Latest modelled pollutant levels")
        st.plotly_chart(chart_style(fig,340),use_container_width=True)
        st.dataframe(values,use_container_width=True,hide_index=True)
        st.markdown("#### Interpretation")
        st.write("Concentration levels are not directly comparable across pollutants because each pollutant has different health-based thresholds and units. A future explainability layer will explain a trained model's predictions, not infer causation from these values.")
    else: st.warning("No pollutant data available.")

# ----------------------------- POLLUTION DETECTIVE -------------------------
elif page=="Pollution Detective":
    st.subheader("Investigate pollutant patterns")
    if not model_df.empty:
        signal=st.selectbox("Choose pollutant",list(POLLUTANT_MAP))
        series=model_df[["Time",signal]].dropna().copy()
        window=st.slider("Rolling baseline window (hours)",6,48,24,6)
        series["Rolling mean"]=series[signal].rolling(window,min_periods=max(3,window//3)).mean()
        series["Rolling std"]=series[signal].rolling(window,min_periods=max(3,window//3)).std()
        series["Z-score"]=(series[signal]-series["Rolling mean"])/series["Rolling std"].replace(0,np.nan)
        threshold=st.slider("Anomaly threshold",1.0,4.0,2.5,.1)
        anomalies=series[series["Z-score"].abs()>threshold]
        fig=px.line(series,x="Time",y=signal,title=f"{signal} hourly signal")
        if not anomalies.empty:
            fig.add_scatter(x=anomalies.Time,y=anomalies[signal],mode="markers",name="Potential anomaly",
                            marker=dict(color="#ff7777",size=10))
        st.plotly_chart(chart_style(fig,360),use_container_width=True)
        a,b,c=st.columns(3);a.metric("Potential anomalies",len(anomalies));b.metric("Mean",f"{series[signal].mean():.2f}");c.metric("Maximum",f"{series[signal].max():.2f}")
        st.dataframe(anomalies,use_container_width=True,hide_index=True)
        st.caption("Statistical outliers are investigation prompts, not proof of a pollution incident.")
    else: st.warning("No data available.")

# ----------------------------- HEALTH INTELLIGENCE -------------------------
elif page=="Health Intelligence":
    st.subheader("Exposure and health context")
    st.warning("General educational information only. Not a diagnosis or personal medical risk assessment.")
    if current:
        pm25=current.get("pm2_5"); pm10=current.get("pm10"); us_aqi=current.get("us_aqi")
        a,b,c=st.columns(3)
        with a: metric("PM2.5",f"{pm25:.1f}" if pm25 is not None else "N/A","µg/m³ · model estimate")
        with b: metric("PM10",f"{pm10:.1f}" if pm10 is not None else "N/A","µg/m³ · model estimate")
        with c: metric("US AQI",f"{us_aqi:.0f}" if us_aqi is not None else "N/A","Not Indian CPCB AQI")
        st.markdown("#### What these pollutants mean")
        explanations={
        "PM2.5":"Fine particles can reach deep into the lungs. Concentration and exposure duration both matter.",
        "PM10":"Inhalable particles can irritate the respiratory tract.",
        "NO2":"Combustion-related nitrogen dioxide can irritate airways.",
        "SO2":"Sulfur dioxide can trigger respiratory symptoms, particularly in people with asthma.",
        "CO":"Carbon monoxide can reduce oxygen delivery at sufficiently high exposure.",
        "O3":"Ground-level ozone can irritate lungs, especially during outdoor exertion."}
        for p,explanation in explanations.items():
            with st.expander(p): st.write(explanation)
        st.markdown("#### General precautions")
        st.write("- Check official local air-quality advisories before strenuous outdoor activity.")
        st.write("- People with asthma or heart/lung conditions should follow their clinician's advice.")
        st.write("- Reduce prolonged strenuous outdoor exertion when official local advisories recommend it.")
        st.caption("Open-Meteo values are atmospheric model estimates at a grid cell, not personal exposure measurements.")
    else: st.info("Current pollutant values unavailable.")

# ----------------------------- INTERACTIVE MAP ------------------------------
elif page=="Interactive Map":
    st.subheader("India air-quality model map")
    st.caption("Map points are city-centre model grid locations, not CPCB monitoring stations.")
    map_rows=[]
    progress=st.progress(0)
    for i,(name,(clat,clon)) in enumerate(CITY_COORDS.items()):
        try:
            _,cur,_,_=fetch_open_meteo(name,clat,clon)
            aqi=cur.get("us_aqi")
            pm=cur.get("pm2_5")
            if aqi is not None:
                map_rows.append({"City":name,"Latitude":clat,"Longitude":clon,"Model US AQI":aqi,"PM2.5":pm})
        except Exception:
            pass
        progress.progress((i+1)/len(CITY_COORDS))
    progress.empty()
    if map_rows:
        mdf=pd.DataFrame(map_rows)
        fig=px.scatter_geo(mdf,lat="Latitude",lon="Longitude",hover_name="City",
            hover_data={"Model US AQI":True,"PM2.5":True,"Latitude":False,"Longitude":False},
            color="Model US AQI",size="Model US AQI",color_continuous_scale=["#39c98b","#e8bd55","#e56c68"],
            scope="asia",projection="natural earth",title="Open-Meteo modelled air quality across Indian cities")
        fig.update_geos(lataxis_range=[5,38],lonaxis_range=[67,98],showland=True,landcolor="#152820",
            showocean=True,oceancolor="#081713",showcountries=True,countrycolor="#456254")
        st.plotly_chart(chart_style(fig,570),use_container_width=True)
        st.dataframe(mdf.sort_values("Model US AQI",ascending=False),use_container_width=True,hide_index=True)
    else: st.warning("Could not retrieve map data for any city. Check your connection and retry.")

# ----------------------------- ACTIVITY PLANNER -----------------------------
elif page=="Activity Planner":
    st.subheader("Activity planning with forecast context")
    st.warning("This is general guidance based on modelled air-quality values, not a guarantee that outdoor activity is safe.")
    activity=st.selectbox("Activity",["Walking","Running","Cycling","Outdoor sports","Gardening","Commuting"])
    duration=st.slider("Duration (minutes)",10,180,45,5)
    sensitive=st.checkbox("I want to consider sensitive-group precautions")
    if not model_df.empty:
        future=model_df[model_df.Time>pd.Timestamp.now()].head(24)
        if not future.empty:
            future["hour"]=future.Time.dt.strftime("%a %H:%M")
            fig=px.line(future,x="Time",y="PM2.5",markers=True,title="Next 24 hours · modelled PM2.5")
            st.plotly_chart(chart_style(fig,300),use_container_width=True)
            best=future.nsmallest(3,"PM2.5")[["Time","PM2.5"]]
            st.markdown("#### Lowest modelled PM2.5 periods in the next 24 hours")
            st.dataframe(best,use_container_width=True,hide_index=True)
            st.caption("These are lower modelled PM2.5 periods, not certified safe exercise windows.")
        st.write(f"Selected activity: **{activity}**, duration: **{duration} minutes**.")
        if sensitive:
            st.info("If you are in a sensitive group, follow clinician guidance and local public-health advisories rather than relying on a model forecast.")
    else: st.warning("Forecast unavailable.")

# ----------------------------- DATA EXPLORER --------------------------------
elif page=="Data Explorer":
    st.subheader("Explore and export API data")
    if not model_df.empty:
        available=model_df.copy()
        start=st.date_input("From",value=available.Time.min().date())
        end=st.date_input("To",value=available.Time.max().date())
        filtered=available[(available.Time.dt.date>=start)&(available.Time.dt.date<=end)]
        cols=st.multiselect("Columns",list(available.columns),default=["Time","PM2.5","PM10","US AQI"])
        if cols: st.dataframe(filtered[cols],use_container_width=True,hide_index=True)
        else: st.info("Select at least one column.")
        st.download_button("Download filtered API data CSV",filtered.to_csv(index=False),"airsense_openmeteo_data.csv","text/csv")
        st.caption("Data source: Open-Meteo Air Quality API / CAMS global model. Timestamp and grid-cell metadata are retained where available.")
    else: st.warning("No API data available to explore.")
    st.markdown("#### CPCB records")
    if not city_cpcb.empty:
        st.dataframe(city_cpcb,use_container_width=True,hide_index=True)
        st.download_button("Download CPCB city records",city_cpcb.to_csv(index=False),"airsense_cpcb_records.csv","text/csv")
    elif not cpcb_key:
        st.info("Configure your data.gov.in API key in the sidebar to enable official CPCB station data.")

st.divider()
st.markdown('<div class="small-muted">AirSense AI · Real API integration build. Open-Meteo/CAMS values are model estimates. CPCB observations are shown only from the official data.gov.in API when a valid key is supplied. Modelled AQI indices are explicitly labelled by their index standard; they are not relabelled as Indian CPCB AQI.</div>',unsafe_allow_html=True)
