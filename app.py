import io
import json
import math
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import httplib2
import google_auth_httplib2
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# Configuración de página Streamlit
st.set_page_config(
    page_title="Sistema QC Meteorológico | CCA-DOA-MARN",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "https://www.snet.gob.sv/Geologia/pcbase2/tabla2.php"

# Presión Barométrica Nominal de Referencia (hPa) por estación en El Salvador
REFERENCIA_PRESION_ESTACION = {
    "33": {"nombre": "Santa Ana", "bp_ref": 938.0},
    "289": {"nombre": "Ch. del Guayabo", "bp_ref": 995.0},
    "283": {"nombre": "Ahuachapán", "bp_ref": 930.0},
    "284": {"nombre": "Hachadura Met", "bp_ref": 1013.25},
    "31": {"nombre": "La Unión", "bp_ref": 1013.25},
    "279": {"nombre": "Jiquilisco", "bp_ref": 1013.25},
    "120": {"nombre": "Hda Melara", "bp_ref": 1013.25},
    "36": {"nombre": "PROCAFE", "bp_ref": 940.0},
    "280": {"nombre": "SANTA ROSA", "bp_ref": 940.0},
}

ESTACIONES = {info["nombre"]: id_code for id_code, info in REFERENCIA_PRESION_ESTACION.items()}
PARAMETROS = ["PP", "PC", "AT", "RH", "DP", "BP", "RI"]

# -----------------------------------------------------------------------------
# CLIENTE GOOGLE DRIVE
# -----------------------------------------------------------------------------
@st.cache_resource
def obtener_servicio_drive():
    """Autentica con la API de Google Drive usando AuthorizedHttp con timeout SSL."""
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    
    if "gcp_service_account" in st.secrets:
        cred_sec = st.secrets["gcp_service_account"]
        if isinstance(cred_sec, str):
            info_credenciales = json.loads(cred_sec)
        else:
            info_credenciales = dict(cred_sec)
    else:
        with open("credentials.json") as f:
            info_credenciales = json.load(f)

    creds = Credentials.from_service_account_info(info_credenciales, scopes=SCOPES)
    
    # Manejo HTTP con timeout para prevenir bloqueos por SSL
    http_base = httplib2.Http(timeout=15)
    authorized_http = google_auth_httplib2.AuthorizedHttp(creds, http=http_base)
    
    return build("drive", "v3", http=authorized_http)


def leer_csv_desde_drive(service, folder_id, nombre_archivo):
    """Busca y descarga un archivo CSV de la carpeta especificada en Drive."""
    try:
        query = f"'{folder_id}' in parents and name = '{nombre_archivo}' and trashed = false"
        
        results = service.files().list(
            q=query, 
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        items = results.get("files", [])

        if not items:
            return pd.DataFrame(), None

        file_id = items[0]["id"]
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        fh.seek(0)
        df = pd.read_csv(fh)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, file_id
    except Exception:
        # En caso de error al leer desde Drive, se retorna un DataFrame vacío para continuar con Scraping
        return pd.DataFrame(), None


def guardar_csv_en_drive(service, folder_id, nombre_archivo, df, file_id=None):
    """Intenta guardar o actualizar un CSV en Google Drive tolerando restricciones de cuota."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    media = MediaIoBaseUpload(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        mimetype="text/csv",
        resumable=True,
    )

    try:
        if file_id:
            service.files().update(
                fileId=file_id, 
                media_body=media, 
                supportsAllDrives=True
            ).execute()
        else:
            file_metadata = {"name": nombre_archivo, "parents": [folder_id]}
            service.files().create(
                body=file_metadata, 
                media_body=media, 
                fields="id", 
                supportsAllDrives=True
            ).execute()
    except Exception:
        # La Cuenta de Servicio no tiene cuota propia para subir/modificar archivos.
        # Capturamos la excepción para evitar que colapse la aplicación en Streamlit.
        st.warning("⚠️ No se pudo sincronizar el archivo en Google Drive debido a limitaciones de cuota de la Cuenta de Servicio. Los datos calculados se están mostrando en tiempo real.")


# -----------------------------------------------------------------------------
# SCRAPING, LIMPIEZA E INTERPOLACIÓN QC
# -----------------------------------------------------------------------------
def calcular_punto_rocio_magnus(temp, rh):
    """Calcula el punto de rocío (°C) mediante la ecuación de Magnus-Tetens."""
    if rh <= 0:
        return temp
    a, b = 17.27, 237.7
    alpha = ((a * temp) / (b + temp)) + math.log(rh / 100.0)
    return (b * alpha) / (a - alpha)


def limpiar_y_rellenar_datos(df):
    """Filtra picos/outliers físicos y rellena huecos mediante interpolación temporal."""
    if df.empty:
        return df

    df = df.set_index("timestamp").sort_index()

    # 1. Filtros de Rangos Físicos Válidos
    if "AT" in df.columns:
        df.loc[(df["AT"] < 5.0) | (df["AT"] > 48.0), "AT"] = None
    if "RH" in df.columns:
        df.loc[(df["RH"] < 0.0) | (df["RH"] > 100.0), "RH"] = None
    if "BP" in df.columns:
        df.loc[(df["BP"] < 750.0) | (df["BP"] > 1080.0), "BP"] = None

    # 2. Filtro Gradient Check (Spikes)
    for col, umbral in [("AT", 6.0), ("BP", 10.0), ("RH", 40.0)]:
        if col in df.columns:
            diff = df[col].diff().abs()
            df.loc[diff > umbral, col] = None

    # 3. Reindexación regular de tiempo (cada 10 min)
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="10min")
    df = df.reindex(full_idx)

    # 4. Interpolación de variables continuas
    cols_cont = [c for c in ["AT", "RH", "BP"] if c in df.columns]
    df[cols_cont] = df[cols_cont].interpolate(method="time", limit=12).bfill().ffill()

    # 5. Precipitación sin interpolar (faltantes = 0.0)
    cols_lluvia = [c for c in ["PP", "PC"] if c in df.columns]
    df[cols_lluvia] = df[cols_lluvia].fillna(0.0)

    df = df.reset_index().rename(columns={"index": "timestamp"})
    df["fecha"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df


def obtener_serie_tiempo(estacion_id, parametro, fecha_str):
    params = {"estacionid": estacion_id, "parametroid": parametro, "fecha": fecha_str}
    try:
        r = requests.get(BASE_URL, params=params, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return {}

    datos = {}
    for fila in soup.find_all("tr"):
        cols = fila.find_all("td")
        if len(cols) >= 3:
            hora = cols[1].get_text(strip=True)
            valor_txt = cols[2].get_text(strip=True).replace(",", ".")
            if ":" in hora:
                try:
                    datos[hora] = float(valor_txt)
                except ValueError:
                    pass
    return datos


def scraping_reciente(estacion_id):
    hoy = datetime.now()
    registros = []
    for d in range(6, -1, -1):
        fecha_dt = hoy - timedelta(days=d)
        str_fecha = fecha_dt.strftime("%Y-%m-%d")
        for param in PARAMETROS:
            datos = obtener_serie_tiempo(estacion_id, param, str_fecha)
            for hora, val in datos.items():
                try:
                    ts = datetime.strptime(f"{str_fecha} {hora}", "%Y-%m-%d %H:%M")
                except ValueError:
                    continue
                registros.append({"timestamp": ts, "fecha": str_fecha, "param": param, "valor": val})

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    df = df.pivot_table(index=["timestamp", "fecha"], columns="param", values="valor", aggfunc="first").reset_index()
    return df


def obtener_datos_completos(estacion_id, folder_id):
    """Sincroniza el historial almacenado en Drive con las últimas lecturas del SNET."""
    drive_service = obtener_servicio_drive()
    nombre_archivo = f"estacion_{estacion_id}_historico.csv"

    # 1. Leer histórico de Drive
    df_historico, file_id = leer_csv_desde_drive(drive_service, folder_id, nombre_archivo)

    # 2. Hacer web scraping de los últimos 7 días
    df_reciente = scraping_reciente(estacion_id)

    # 3. Unificar y eliminar duplicados por timestamp
    if not df_reciente.empty:
        if not df_historico.empty:
            df_unificado = pd.concat([df_historico, df_reciente], ignore_index=True)
            df_unificado = df_unificado.drop_duplicates(subset=["timestamp"], keep="last")
        else:
            df_unificado = df_reciente
    else:
        df_unificado = df_historico

    if df_unificado.empty:
        return pd.DataFrame()

    # 4. Aplicar depuración QC e interpolación
    df_unificado = limpiar_y_rellenar_datos(df_unificado)

    # 5. Intentar guardar en Google Drive sin detener la app si la cuota falla
    guardar_csv_en_drive(drive_service, folder_id, nombre_archivo, df_unificado, file_id)

    return df_unificado.sort_values("timestamp")


def evaluar_control_calidad_lluvia(row, estacion_id):
    pp = row.get("PP_Calculada", 0.0)
    if pp <= 0:
        return False

    rh, temp, dp, bp = row.get("RH", 0.0), row.get("AT", 0.0), row.get("DP", 0.0), row.get("BP", 1013.25)
    info = REFERENCIA_PRESION_ESTACION.get(estacion_id, {"bp_ref": 1013.25})
    bp_ref = info["bp_ref"]

    if bp > 980.0 and bp_ref < 980.0:
        bp_ref = 1013.25

    depresion_rocio = temp - dp

    if rh >= 93.0 and depresion_rocio <= 1.5:
        return True
    if rh >= 85.0 and depresion_rocio <= 2.5 and bp <= (bp_ref + 3.0):
        return True

    return False


# -----------------------------------------------------------------------------
# INTERFAZ STREAMLIT
# -----------------------------------------------------------------------------
st.title("🌧️ Portal de Control de Calidad Meteorológico")
st.caption("Producto de Uso Interno CCA - DOA - MARN | Módulo QC Continuum")

# Panel lateral
st.sidebar.header("Parámetros de Entrada")
estacion_nombre = st.sidebar.selectbox("Estación Meteorológica", list(ESTACIONES.keys()))
estacion_id = ESTACIONES[estacion_nombre]
aplicar_qc = st.sidebar.checkbox("Activar Algoritmo QC (Filtrar Lluvia Ficticia)", value=True)

FOLDER_ID = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID", "")

if not FOLDER_ID:
    st.error("⚠️ No se ha configurado la variable 'GOOGLE_DRIVE_FOLDER_ID' en los Secrets de Streamlit.")
    st.stop()

with st.spinner("Sincronizando historial con Google Drive y extrayendo datos recientes del SNET..."):
    df = obtener_datos_completos(estacion_id, FOLDER_ID)

if df.empty:
    st.warning("No se encontraron registros para la estación seleccionada.")
    st.stop()

# Filtro interactivo de fechas
min_fecha = df["timestamp"].min().date()
max_fecha = df["timestamp"].max().date()
fechas_sel = st.sidebar.date_input(
    "Rango de fechas a visualizar",
    [max_fecha - timedelta(days=7), max_fecha],
    min_value=min_fecha,
    max_value=max_fecha,
)

if len(fechas_sel) == 2:
    f_inicio, f_fin = fechas_sel
    df = df[(df["timestamp"].dt.date >= f_inicio) & (df["timestamp"].dt.date <= f_fin)].copy()

# Cálculo de precipitación acumulada
if "PC" in df.columns:
    df["PP_Calculada"] = df["PC"].diff().fillna(0).apply(lambda x: x if 0 < x < 30 else 0.0)
elif "PP" in df.columns:
    df["PP_Calculada"] = df["PP"].apply(lambda x: x if 0 < x < 30 else 0.0)
else:
    df["PP_Calculada"] = 0.0

df["DP"] = df.apply(lambda r: calcular_punto_rocio_magnus(r["AT"], r["RH"]), axis=1)
df["Es_Lluvia_Real"] = df.apply(lambda r: evaluar_control_calidad_lluvia(r, estacion_id), axis=1)

if aplicar_qc:
    df["PP_Plot"] = df.apply(lambda r: r["PP_Calculada"] if r["Es_Lluvia_Real"] else 0.0, axis=1)
    df["PP_Ficticia"] = df.apply(lambda r: r["PP_Calculada"] if not r["Es_Lluvia_Real"] else 0.0, axis=1)
else:
    df["PP_Plot"] = df["PP_Calculada"]
    df["PP_Ficticia"] = 0.0

# Botón de descarga local en la barra lateral
csv_datos = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Descargar CSV Procesado",
    data=csv_datos,
    file_name=f"estacion_{estacion_id}_qc.csv",
    mime="text/csv",
)

# Tarjetas métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precipitación Validada QC", f"{df['PP_Plot'].sum():.1f} mm")
col2.metric("Lluvia Ficticia Descartada", f"{df['PP_Ficticia'].sum():.1f} mm")
col3.metric("Rango Térmico", f"{df['AT'].min():.1f}°C / {df['AT'].max():.1f}°C")
col4.metric("Humedad Promedio", f"{df['RH'].mean():.0f}%")

# Gráficos Apilados
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=(
        "Precipitación Validada y Ruido de Sensor (mm / 10 min)",
        "Humedad Relativa (%)",
        "Presión Barométrica (hPa)",
        "Temperatura del Aire (°C) y Punto de Rocío (°C)",
    ),
)

fig.add_trace(go.Bar(x=df["timestamp"], y=df["PP_Plot"], marker_color="#38BDF8", name="Lluvia Real"), row=1, col=1)
if aplicar_qc:
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["PP_Ficticia"], marker_color="#EF4444", name="Lluvia Ficticia"), row=1, col=1)

fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RH"], mode="lines", line=dict(color="#C084FC", width=2), name="Humedad"), row=2, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BP"], mode="lines", line=dict(color="#F87171", width=2), name="Presión"), row=3, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["AT"], mode="lines", line=dict(color="#FB923C", width=2), name="Temp"), row=4, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["DP"], mode="lines", line=dict(color="#38BDF8", width=1.5, dash="dash"), name="P. Rocío"), row=4, col=1)

fig.update_layout(template="plotly_dark", height=850, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)
