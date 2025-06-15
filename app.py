
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib # For loading the scaler
import os

st.set_page_config(layout="wide")

# --- Configuration ---
MODEL_FILENAME = "modelo_lstm_simple.keras"
SCALER_FILENAME = "scaler.gz"
EXAMPLE_DATA_FILENAME = "dataset_imputado_simpleml.csv" # From notebook

# Construct paths relative to the app's directory or a known base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelos")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
EXAMPLE_DATA_PATH = os.path.join(DATA_DIR, EXAMPLE_DATA_FILENAME)

SEQUENCE_LENGTH = 365  # From notebook cell a0a8bb62
MODEL_FORECAST_STEPS = 14 # From notebook cell a0a8bb62
COLUMNS_TO_PREDICT = ['ayacucho', 'caicara', 'ciudad_bolivar', 'palua'] # From notebook

# --- Load Model and Scaler ---
@st.cache_resource
def load_keras_model(path):
    if not os.path.exists(path):
        st.error(f"Error cargando el modelo: Archivo no encontrado en '{path}'.")
        return None
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

@st.cache_resource
def load_joblib_scaler(path):
    if not os.path.exists(path):
        st.error(f"Error cargando el scaler: Archivo no encontrado en '{path}'. Asegúrate de que el archivo exista y haya sido guardado correctamente desde el notebook (ej: joblib.dump(scaler, '{path}')).")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error cargando el scaler: {e}.")
        return None

model = load_keras_model(MODEL_PATH)
scaler = load_joblib_scaler(SCALER_PATH)

# --- Helper Functions ---
def preprocess_data(df_input, scaler_obj):
    df = df_input.copy()
    # Ensure 'fecha' column exists and is datetime
    if 'fecha' not in df.columns:
        st.error("La columna 'fecha' no se encuentra en el CSV.")
        return None, None
    try:
        df['fecha'] = pd.to_datetime(df['fecha'])
    except Exception as e:
        st.error(f"Error al convertir la columna 'fecha' a datetime: {e}")
        return None, None
    
    df = df.sort_values(by='fecha').reset_index(drop=True)

    # Check for required prediction columns
    missing_cols = [col for col in COLUMNS_TO_PREDICT if col not in df.columns]
    if missing_cols:
        st.error(f"Faltan las siguientes columnas requeridas en el CSV: {', '.join(missing_cols)}")
        return None, None

    data_to_scale = df[COLUMNS_TO_PREDICT].copy()
    
    # Handle NaNs before scaling (as in notebook: interpolate, then drop remaining)
    # Note: Your notebook interpolates 'palua' specifically then drops all NaNs.
    # For a general uploader, a more robust strategy might be needed.
    # Here, we'll try to mimic the notebook's outcome: fill then check.
    for col in COLUMNS_TO_PREDICT:
        if data_to_scale[col].isnull().any():
            data_to_scale[col] = data_to_scale[col].interpolate(method="linear", limit_direction="both")
    
    if data_to_scale.isnull().values.any():
        nan_cols = data_to_scale.columns[data_to_scale.isnull().any()].tolist()
        st.warning(f"Se encontraron NaNs en las columnas {nan_cols} después de la interpolación. Se eliminarán las filas con NaNs.")
        df = df.dropna(subset=COLUMNS_TO_PREDICT).reset_index(drop=True)
        data_to_scale = df[COLUMNS_TO_PREDICT].copy() # Re-select after dropping NaNs from df
        if data_to_scale.empty:
            st.error("El DataFrame quedó vacío después de eliminar NaNs.")
            return None, None

    if scaler_obj is None:
        st.error("El scaler no está cargado. No se puede preprocesar.")
        return None, None
        
    try:
        # Ensure no NaNs remain before scaling
        if data_to_scale.isnull().values.any():
             st.error(f"Error: Aún existen NaNs en los datos antes de escalar en columnas: {data_to_scale.columns[data_to_scale.isnull().any()].tolist()}. Por favor, limpie los datos.")
             return None, None
        scaled_data = scaler_obj.transform(data_to_scale)
    except ValueError as e:
        if "Found array with 0 sample(s)" in str(e) or "Expected 2D array, got 1D array instead" in str(e) :
             st.error(f"Error al escalar los datos: El DataFrame parece estar vacío o tener un formato incorrecto. {e}")
        elif "contains NaN" in str(e): # Should be caught above
            st.error(f"Error al escalar los datos: Contiene NaNs. {e}.")
        else:
            st.error(f"Error al escalar los datos: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error inesperado al escalar los datos: {e}")
        return None, None
        
    return df, scaled_data

def create_sequences_for_prediction(scaled_data_input, sequence_length):
    if len(scaled_data_input) < sequence_length:
        st.error(f"No hay suficientes datos históricos ({len(scaled_data_input)} filas) para crear una secuencia de entrada de longitud {sequence_length}.")
        return None
    last_sequence = scaled_data_input[-sequence_length:]
    return last_sequence.reshape((1, sequence_length, scaled_data_input.shape[1]))

def make_predictions_fn(model_obj, input_sequence, scaler_obj):
    if model_obj is None or input_sequence is None or scaler_obj is None:
        return None
    
    raw_predictions = model_obj.predict(input_sequence) # Shape (1, MODEL_FORECAST_STEPS, n_features)
    predictions_scaled = raw_predictions[0] # Shape (MODEL_FORECAST_STEPS, n_features)
    
    predictions_real = scaler_obj.inverse_transform(predictions_scaled)
    return predictions_real

# --- Streamlit UI ---

st.title("📊 Predictor de Niveles de Agua")
st.caption("Sistema de IA para predicción de niveles de agua en ríos (Modelo LSTM Orinoco)")

# Initialize session state
if 'data_df' not in st.session_state:
    st.session_state.data_df = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None

# --- Sidebar ---
with st.sidebar:
    st.header("Cargar Datos")
    st.markdown(f"""
    Sube un archivo CSV con datos históricos de niveles de agua.
    El archivo CSV debe tener la columna `fecha` (formato YYYY/MM/DD o YYYY-MM-DD) 
    y las columnas de niveles: `{', '.join(COLUMNS_TO_PREDICT)}`.
    """)
    uploaded_file = st.file_uploader("Archivo CSV", type="csv", key="file_uploader")
    
    if st.button("Usar Datos de Ejemplo", key="example_data_button"):
        if not os.path.exists(EXAMPLE_DATA_PATH):
            st.error(f"Archivo de datos de ejemplo no encontrado en '{EXAMPLE_DATA_PATH}'.")
        else:
            try:
                df = pd.read_csv(EXAMPLE_DATA_PATH)
                st.session_state.data_df, st.session_state.scaled_data = preprocess_data(df, scaler)
                st.session_state.predictions_df = None # Clear previous predictions
                if st.session_state.data_df is not None:
                    st.success("Datos de ejemplo cargados y procesados.")
            except Exception as e:
                st.error(f"Error al cargar datos de ejemplo: {e}")
                st.session_state.data_df = None
                st.session_state.scaled_data = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data_df, st.session_state.scaled_data = preprocess_data(df, scaler)
            st.session_state.predictions_df = None # Clear previous predictions
            if st.session_state.data_df is not None:
                st.success("Archivo CSV cargado y procesado.")
        except Exception as e:
            st.error(f"Error al procesar el archivo CSV: {e}")
            st.session_state.data_df = None
            st.session_state.scaled_data = None
    
    st.markdown("---")
    st.header("Configuración de Predicción")
    st.markdown(f"El modelo predecirá los próximos **{MODEL_FORECAST_STEPS} días**.")
    # The dropdown from the image is for "Días a Predecir".
    # Our model outputs a fixed MODEL_FORECAST_STEPS.
    # For simplicity, we fix prediction length. A dropdown could select a subset of these.
    # selected_display_days = st.selectbox("Días a mostrar de la predicción:", 
    #                                      options=list(range(1, MODEL_FORECAST_STEPS + 1)), 
    #                                      index=MODEL_FORECAST_STEPS - 1)


    if st.button("Ejecutar Predicción", key="predict_button", disabled=(st.session_state.scaled_data is None or model is None or scaler is None)):
        if st.session_state.scaled_data is not None and model is not None and scaler is not None:
            input_sequence = create_sequences_for_prediction(st.session_state.scaled_data, SEQUENCE_LENGTH)
            if input_sequence is not None:
                predictions_real = make_predictions_fn(model, input_sequence, scaler)
                
                if predictions_real is not None:
                    last_historical_date = st.session_state.data_df['fecha'].iloc[-1]
                    prediction_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=MODEL_FORECAST_STEPS)
                    
                    st.session_state.predictions_df = pd.DataFrame(predictions_real, columns=COLUMNS_TO_PREDICT, index=prediction_dates)
                    st.success(f"Predicción para los próximos {MODEL_FORECAST_STEPS} días generada.")
                else:
                    st.error("No se pudieron generar las predicciones.")
        else:
            if st.session_state.scaled_data is None: st.warning("Carga datos históricos primero.")
            if model is None: st.error("El modelo Keras no está cargado.")
            if scaler is None: st.error("El scaler (preprocesador) no está cargado.")

# --- Main Area: Visualization ---
st.header("Visualización de Datos")
st.caption("Datos históricos y predicciones de niveles de agua")

if st.session_state.data_df is None and st.session_state.predictions_df is None:
    st.info("⬅️ Carga datos históricos o ejecuta una predicción para visualizar.")

tab1, tab2 = st.tabs(["Datos Históricos", "Predicciones"])

with tab1:
    if st.session_state.data_df is not None:
        st.subheader("Gráficos de Niveles Históricos")
        
        selected_cities_hist = st.multiselect("Selecciona ciudades para graficar (Histórico):",
                                          options=COLUMNS_TO_PREDICT,
                                          default=COLUMNS_TO_PREDICT, key="hist_multiselect")
        if selected_cities_hist:
            hist_df_to_plot = st.session_state.data_df.set_index('fecha')[selected_cities_hist]
            st.line_chart(hist_df_to_plot)

            st.subheader("Tabla de Datos Históricos (últimos 10)")
            st.dataframe(st.session_state.data_df.tail(10))
        elif st.session_state.data_df is not None : # Data is loaded but no city selected
             st.info("Selecciona al menos una ciudad para mostrar el gráfico histórico.")
            
    else:
        st.info("Carga datos históricos desde la barra lateral para visualizar en esta pestaña.")

with tab2:
    if st.session_state.predictions_df is not None:
        st.subheader("Gráficos de Predicciones")
        
        if st.session_state.data_df is not None:
            context_days = 3 * MODEL_FORECAST_STEPS 
            historical_context_df = st.session_state.data_df.set_index('fecha')
            
            st.session_state.predictions_df.index = pd.to_datetime(st.session_state.predictions_df.index)

            for city in COLUMNS_TO_PREDICT:
                fig, ax = plt.subplots(figsize=(12, 5))
                
                if city in historical_context_df.columns:
                    # Plot only a relevant portion of historical data for context
                    city_historical_context = historical_context_df[city].iloc[-context_days:]
                    ax.plot(city_historical_context.index, city_historical_context.values, label=f'Histórico - {city}', color='blue', alpha=0.7)
                
                if city in st.session_state.predictions_df.columns:
                    ax.plot(st.session_state.predictions_df.index, st.session_state.predictions_df[city], label=f'Predicción - {city}', color='red', linestyle='--', marker='o', markersize=4)

                ax.set_title(f"Histórico y Predicción para {city.replace('_', ' ').title()}")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Nivel de Agua")
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.subheader("Tabla de Predicciones")
            st.dataframe(st.session_state.predictions_df)
        else: 
            st.line_chart(st.session_state.predictions_df[COLUMNS_TO_PREDICT])
            st.subheader("Tabla de Predicciones")
            st.dataframe(st.session_state.predictions_df)
    else:
        st.info("Ejecuta una predicción desde la barra lateral para visualizar en esta pestaña.")

st.markdown("---")
st.markdown("Modelo LSTM basado en el notebook `model.ipynb`.")
st.markdown(f"**Nota Importante:** El scaler (StandardScaler) debe ser guardado desde el notebook de entrenamiento y estar disponible en `{SCALER_PATH}` para que las predicciones con nuevos datos funcionen correctamente.")


#**To run this app:**

#1.  Save the code above as `app.py` in the root of your `orinoco-lstm` project directory (the same level as your `data` and `modelos` folders).
#2.  Open your terminal, navigate to this directory.
#3.  Run: `streamlit run app.py`

#This will open the web interface in your browser. Remember to have the model and scaler files in the correct `modelos` subfolder and