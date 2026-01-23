import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from contextlib import redirect_stdout

from calculadora_bayesiana import CalculadoraClicksBayesiana
from calculadora_bayesiana_conversiones import CalculadoraConversionesBayesiana


# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="Calculadora A/B",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CSS (manteniendo tu estilo)
# =========================
st.markdown("""
<style>
:root {
--primary-color: #6366f1;
--secondary-color: #8b5cf6;
--success-color: #10b981;
--warning-color: #f59e0b;
--error-color: #ef4444;
--info-color: #3b82f6;
--background-light: #f8fafc;
--background-card: #ffffff;
--text-primary: #1e293b;
--text-secondary: #64748b;
--border-color: #e2e8f0;
}

.main-header {
font-size: 2.5rem;
color: var(--primary-color);
font-weight: 700;
margin-bottom: 2rem;
text-align: center;
}

.sub-header {
font-size: 1.5rem;
color: var(--text-primary);
font-weight: 600;
margin: 1.5rem 0;
}

.success-box {
background: linear-gradient(135deg, #10b981, #059669);
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1);
}

.info-box {
background: linear-gradient(135deg, var(--info-color), var(--primary-color));
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1);
}

.warning-box {
background: linear-gradient(135deg, var(--warning-color), #d97706);
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.1);
}

.error-box {
background: linear-gradient(135deg, var(--error-color), #dc2626);
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
}

.section-spacer { margin: 3rem 0; }
.subsection-spacer { margin: 2rem 0; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }

.stTabs [data-baseweb="tab"] {
height: 50px;
padding-left: 20px;
padding-right: 20px;
background-color: var(--background-light);
border-radius: 8px;
color: var(--text-secondary);
font-weight: 500;
}

.stTabs [aria-selected="true"] {
background-color: #10b981 !important;
color: white !important;
}

.stButton > button {
background: linear-gradient(135deg, #64748b, #475569);
color: white;
border: none;
border-radius: 8px;
padding: 0.6rem 1.5rem;
font-weight: 600;
transition: all 0.3s ease;
}

.stButton > button:hover {
background: linear-gradient(135deg, #475569, #334155);
transform: translateY(-2px);
box-shadow: 0 4px 12px rgba(100, 116, 139, 0.3);
}

.stButton > button[kind="primary"] {
background: linear-gradient(135deg, #10b981, #059669);
}

.stButton > button[kind="primary"]:hover {
background: linear-gradient(135deg, #059669, #047857);
box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.result-card {
background: var(--background-card);
padding: 2rem;
border-radius: 16px;
border: 1px solid var(--border-color);
box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
margin: 1.5rem 0;
}

.choice-card {
background: var(--background-card);
padding: 1.6rem;
border-radius: 16px;
border: 1px solid var(--border-color);
box-shadow: 0 4px 10px rgba(0,0,0,0.06);
height: 100%;
}

.choice-title {
font-size: 1.25rem;
font-weight: 700;
color: var(--text-primary);
margin-bottom: 0.5rem;
}

.choice-text {
color: var(--text-secondary);
line-height: 1.5;
}

.center-wrap {
max-width: 1050px;
margin: 0 auto;
}

</style>
""", unsafe_allow_html=True)


# =========================
# Helpers de estado
# =========================
def reset_wizard():
    st.session_state.wizard_step = 1
    st.session_state.enfoque = None           # "bayesiano" | "frecuentista"
    st.session_state.session_id = None        # True | False
    st.session_state.tipo_valores = None      # "0_1" | "0_inf"
    st.session_state.ruta_ok = False
    st.session_state.selected_model_label = None
    st.session_state.show_app = False
    # Nota: NO tocamos datos de calculadora aquí; se reinicia cuando entras a app


def init_wizard_state():
    if "wizard_step" not in st.session_state:
        reset_wizard()


def set_calculadora_from_selected_model():
    """
    Inicializa la calculadora correcta según el modelo seleccionado por el wizard.
    """
    modelo = st.session_state.get("selected_model_label")
    if modelo == "Conversiones 0/1 (Beta–Binomial)":
        st.session_state.calculadora = CalculadoraConversionesBayesiana()
    else:
        st.session_state.calculadora = CalculadoraClicksBayesiana()

    st.session_state.datos_procesados = False


def check_route_and_set_model():
    """
    Define si la ruta está disponible (solo lo implementado hoy).
    """
    enfoque = st.session_state.get("enfoque")
    session_id = st.session_state.get("session_id")
    tipo_valores = st.session_state.get("tipo_valores")

    # Solo disponible:
    # - Bayesiano
    # - Sin Session ID
    # - valores 0/1 => Beta-Binomial
    # - valores 0-inf => Gamma-Poisson
    if enfoque == "bayesiano" and session_id is False and tipo_valores in ("0_1", "0_inf"):
        st.session_state.ruta_ok = True
        st.session_state.selected_model_label = (
            "Conversiones 0/1 (Beta–Binomial)" if tipo_valores == "0_1"
            else "Clicks (Gamma–Poisson)"
        )
    else:
        st.session_state.ruta_ok = False
        st.session_state.selected_model_label = None


# =========================
# Wizard UI (Figma-like)
# =========================
def render_wizard():
    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)

    # Header
    st.markdown('<h2 class="main-header">VML THE COCKTAIL</h2>', unsafe_allow_html=True)

    step = st.session_state.wizard_step

    # Step 1: elegir enfoque
    if step == 1:
        st.markdown("""
        <div class="result-card">
            <div class="choice-title">¡Bienvenido a la calculadora de tests A/B!</div>
            <div class="choice-text">
                Esta calculadora te ayudará a tomar decisiones basadas en datos eligiendo entre el enfoque bayesiano o frecuentista, los dos modelos estadísticos más comunes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <div class="choice-title">Elige el modelo que deseas utilizar para analizar tu test A/B</div>
            <div class="choice-text">¿No sabes cuál elegir? No te preocupes: te explico cada uno de forma sencilla.</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("""
            <div class="choice-card">
                <div class="choice-title">Modelo Bayesiano</div>
                <div class="choice-text">
                    El enfoque bayesiano interpreta los resultados en términos de probabilidad directa.
                    <br><br>
                    En lugar de preguntarse “¿es este resultado estadísticamente significativo?”, responde preguntas como:
                    “¿cuál es la probabilidad de que la variante B sea mejor que la A?”
                    <ul>
                        <li>No necesitas un tamaño de muestra fijo.</li>
                        <li>Análisis de resultados vasado en probabilidad.</li>
                        <li>Decisión más rápida: puedes parar cuando desees.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Elegir modelo Bayesiano", key="btn_bayesiano", type="primary"):
                st.session_state.enfoque = "bayesiano"
                st.session_state.wizard_step = 2
                st.rerun()
 
        with col2:
            st.markdown("""
            <div class="choice-card">
                <div class="choice-title">Modelo Frecuentista</div>
                <div class="choice-text">
                    El enfoque frecuentista se centra en comprobar si la diferencia observada podría deberse al azar,
                    respondiendo preguntas como: “¿la diferencia A vs B es estadísticamente significativa?” o “¿podemos rechazar la hipótesis nula?”.
                    <ul>
                        <li>Debes calcular previamente la muestra y esperar hasta alcanzarla.</li>
                        <li>Análisis de resultados basado en p-value.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Elegir modelo Frecuentista", key="btn_frecuentista", type="primary"):
                st.session_state.enfoque = "frecuentista"
                st.session_state.wizard_step = 2
                st.rerun()

    # Step 2: Session ID
    elif step == 2:
        enfoque_txt = "Bayesiano" if st.session_state.enfoque == "bayesiano" else "Frecuentista"
        st.markdown(f"""
        <div class="result-card">
            <div class="choice-title">¡Buena elección!</div>
            <div class="choice-text">Has seleccionado analizar tu test A/B con el modelo {enfoque_txt}.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <div class="choice-title">¿Puedes analizar tu test A/B con “Session ID” de cada sesión que ha formado parte del experimento?</div>
            <div class="choice-text">
                Para analizar correctamente un experimento A/B es necesario definir la unidad de análisis.
                En entornos web, el uso del “Session ID” de GA4 permite identificar exposiciones y conversiones a nivel de sesión, evitando duplicidades y sesgos en el cáculo de resultados.
                <br><br>
                ¿Es posible analizar este experimento utilizando el Session ID de GA4?
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            if st.button("Tengo Session ID", key="btn_sid_yes", type="primary"):
                st.session_state.session_id = True
                st.session_state.wizard_step = 3
                st.rerun()
        with c2:
            if st.button("No tengo Session ID", key="btn_sid_no", type="primary"):
                st.session_state.session_id = False
                st.session_state.wizard_step = 3
                st.rerun()

        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)
        if st.button("⬅️ Volver", key="back_2"):
            st.session_state.wizard_step = 1
            st.rerun()

    # Step 3: tipo valores
    elif step == 3:
        sid_txt = "con Session ID" if st.session_state.session_id else "sin Session ID"
        st.markdown(f"""
        <div class="result-card">
            <div class="choice-title">¡Perfecto!</div>
            <div class="choice-text">Analizarás tu test A/B {sid_txt}.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <div class="choice-title">¿Los valores de tu test van de 0 a 1 o van desde 0 a infinito?</div>
            <div class="choice-text">
                <ul>
                    <li><b>Valores entre 0 y 1</b>: de esta manera se analizará mediante la distribución previa Beta, ideal para conversiones (siendo 0 la no conversión y 1 si el usuario ha convertido en la sesión).</li>
                    <li><b>Valores de 0 a infinito</b>: con esta opción se analizará mediante la distrubución previa Gamma-Poisson, es adecuada para conteos de métricas.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            if st.button("Valores entre 0 y 1", key="btn_01", type="primary"):
                st.session_state.tipo_valores = "0_1"
                st.session_state.wizard_step = 4
                st.rerun()
        with c2:
            if st.button("Valores de 0 a infinito", key="btn_0inf", type="primary"):
                st.session_state.tipo_valores = "0_inf"
                st.session_state.wizard_step = 4
                st.rerun()

        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)
        if st.button("⬅️ Volver", key="back_3"):
            st.session_state.wizard_step = 2
            st.rerun()

    # Step 4: botón final Analizar + router
    elif step == 4:
        check_route_and_set_model()

        if st.session_state.ruta_ok:
            extra = (
                "De esta manera, el CSV de tu test A/B deberá contener eventos y sesiones agregados."
                if st.session_state.session_id is False else
                "De esta manera, el CSV deberá contener una columna con los Session ID."
            )
            st.markdown(f"""
            <div class="result-card">
                <div class="choice-title">¡Perfecto!</div>
                <div class="choice-text">
                    Ruta disponible ✅<br>
                    Modelo seleccionado: <b>{st.session_state.selected_model_label}</b><br><br>
                    {extra}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("Analizar test A/B", key="btn_go_app", type="primary"):
                    # Preparamos calculadora y entramos
                    set_calculadora_from_selected_model()
                    st.session_state.show_app = True
                    st.rerun()
        else:
            st.markdown("""
            <div class="warning-box">
                <b>Todavía no disponible</b><br><br>
                Con las opciones seleccionadas todavía no tenemos la implementación visual activa.
                Puedes volver al inicio y elegir una ruta disponible (Bayesiano + sin Session ID + 0/1 o 0–∞).
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("Volver al inicio", key="btn_back_home", type="primary"):
                    reset_wizard()
                    st.rerun()

        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)
        if st.button("⬅️ Volver", key="back_4"):
            st.session_state.wizard_step = 3
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# App actual (tu calculadora)
# =========================
def render_calculadora_actual():
    # Título y descripción
    st.markdown('<h2 class="main-header">Calculadora Bayesiana para Tests A/B</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Esta herramienta te permite analizar los resultados de tus pruebas A/B utilizando estadística bayesiana.
    Sube un archivo CSV con tus datos o ingresa la información manualmente.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Sidebar con info + configuración
    with st.sidebar:
        st.markdown('<p class="sub-header">Modelo seleccionado</p>', unsafe_allow_html=True)
        modelo = st.session_state.get("selected_model_label", "—")
        st.info(f"**{modelo}**")

        if st.button("⬅️ Volver al inicio (Wizard)"):
            reset_wizard()
            st.rerun()

        st.markdown('<p class="sub-header">Configuración</p>', unsafe_allow_html=True)

        umbral_prob = st.slider(
            "Umbral de probabilidad para decisión",
            min_value=0.8,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f",
            key="umbral_prob"
        )

        umbral_mejora = st.slider(
            "Umbral de mejora mínima",
            min_value=0.01,
            max_value=0.20,
            value=0.01,
            step=0.01,
            format="%.2f",
            key="umbral_mejora"
        )

        if st.button("Reiniciar calculadora"):
            set_calculadora_from_selected_model()
            st.success("Calculadora reiniciada correctamente")
            st.rerun()

    # Tabs
    st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Cargar CSV", "✏️ Entrada manual", "📋 Formato CSV"])

    # ---------
    # TAB 1 CSV
    # ---------
    with tab1:
        st.markdown('<p class="sub-header">Cargar datos desde CSV</p>', unsafe_allow_html=True)
        st.info("💡 Si no sabes cómo preparar tu archivo CSV, revisa la pestaña **'Formato CSV'**.")

        uploaded_file = st.file_uploader(
            "Selecciona tu archivo CSV",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                columnas_requeridas = ['Día', 'Conversiones A', 'Visitas A', 'Conversiones B', 'Visitas B']
                columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

                if columnas_faltantes:
                    st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
                    st.info("Revisa requisitos en **'Formato CSV'**.")
                else:
                    st.success("✅ ¡Archivo cargado correctamente!")

                    st.subheader("Vista previa de tus datos:")
                    st.dataframe(df, width="stretch")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Días de datos", len(df))
                    with col2:
                        total_visitas_a = df['Visitas A'].sum()
                        total_conv_a = df['Conversiones A'].sum()
                        tasa_prom_a = total_conv_a / total_visitas_a if total_visitas_a > 0 else 0
                        st.metric("Tasa promedio A", f"{tasa_prom_a:.2%}")
                    with col3:
                        total_visitas_b = df['Visitas B'].sum()
                        total_conv_b = df['Conversiones B'].sum()
                        tasa_prom_b = total_conv_b / total_visitas_b if total_visitas_b > 0 else 0
                        st.metric("Tasa promedio B", f"{tasa_prom_b:.2%}")

                    if st.button("🚀 Procesar datos del CSV", type="primary"):
                        calculadora = st.session_state.calculadora

                        with st.spinner("Por favor ten paciencia mientras se cargan los datos..."):
                            progress_bar = st.progress(0, text="Procesando datos del test A/B...")
                            total_rows = len(df)

                            for i, row in df.iterrows():
                                dia = f"Día {int(row['Día'])}"
                                clicks_a = int(row['Conversiones A'])
                                visitas_a = int(row['Visitas A'])
                                clicks_b = int(row['Conversiones B'])
                                visitas_b = int(row['Visitas B'])

                                calculadora.actualizar_con_datos(clicks_a, visitas_a, clicks_b, visitas_b, dia=dia)

                                current_progress = (i + 1) / total_rows
                                progress_bar.progress(
                                    current_progress,
                                    text=f"Procesando día {i+1} de {total_rows}... ({int(current_progress*100)}%)"
                                )

                            st.session_state.datos_procesados = True
                            st.markdown('<div class="success-box">¡Datos procesados correctamente!</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")

    # --------------
    # TAB 2 Manual
    # --------------
    with tab2:
        st.markdown('<p class="sub-header">Entrada manual de datos</p>', unsafe_allow_html=True)

        with st.form("entrada_manual"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Grupo A")
                clicks_a = st.number_input("Conversiones A", min_value=0, value=0)
                visitas_a = st.number_input("Visitas A", min_value=1, value=100)
                tasa_a = clicks_a / visitas_a if visitas_a > 0 else 0
                st.metric("Tasa de conversión A", f"{tasa_a:.2%}")

            with col2:
                st.subheader("Grupo B")
                clicks_b = st.number_input("Conversiones B", min_value=0, value=0)
                visitas_b = st.number_input("Visitas B", min_value=1, value=100)
                tasa_b = clicks_b / visitas_b if visitas_b > 0 else 0
                st.metric("Tasa de conversión B", f"{tasa_b:.2%}")

            dia = st.text_input("Etiqueta del día (opcional)", value="Día 1")
            submitted = st.form_submit_button("Añadir datos")

            if submitted:
                with st.spinner("Por favor ten paciencia mientras se procesan los datos..."):
                    calculadora = st.session_state.calculadora
                    calculadora.actualizar_con_datos(clicks_a, visitas_a, clicks_b, visitas_b, dia=dia)
                    st.session_state.datos_procesados = True
                    st.markdown(f'<div class="success-box">Datos del {dia} añadidos correctamente</div>', unsafe_allow_html=True)

    # ----------------
    # TAB 3 Formato CSV
    # ----------------
    with tab3:
        st.markdown('<p class="sub-header">Cómo preparar tu archivo CSV</p>', unsafe_allow_html=True)

        st.markdown("""
        ### 📋 Formato requerido
        Tu archivo CSV debe contener **exactamente** estas 5 columnas con estos nombres:
        """)

        requisitos_df = pd.DataFrame({
            'Columna': ['Día', 'Conversiones A', 'Visitas A', 'Conversiones B', 'Visitas B'],
            'Descripción': [
                'Identificador del período (número o texto)',
                'Número de conversiones del grupo A',
                'Número total de visitas del grupo A',
                'Número de conversiones del grupo B',
                'Número total de visitas del grupo B'
            ],
            'Ejemplo': [
                '1, 2, 3... o "Lunes", "Martes"...',
                '13, 29, 28...',
                '188, 254, 207...',
                '21, 14, 22...',
                '181, 176, 173...'
            ]
        })

        st.dataframe(requisitos_df, width="stretch", hide_index=True)

        st.markdown("### 📄 Ejemplo de archivo CSV válido:")
        ejemplo_csv_texto = """Día,Conversiones A,Visitas A,Conversiones B,Visitas B
1,13,188,21,181
2,29,254,14,176
3,28,207,22,173
4,35,312,41,298
5,22,189,28,201"""
        st.code(ejemplo_csv_texto, language="csv")

    # =========================
    # Resultados
    # =========================
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    if st.session_state.get("datos_procesados", False):
        st.markdown("---")
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

        st.markdown('<h2 class="main-header">Resultados del Análisis Bayesiano</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        A continuación se muestran los resultados de tu análisis A/B utilizando estadística bayesiana.
        Explora las pestañas para ver el resumen, historial y gráficos.
        </div>
        """, unsafe_allow_html=True)

        res_tab1, res_tab2, res_tab3 = st.tabs(["📋 Resumen", "📝 Historial detallado", "📈 Gráficos"])

        umbral_prob = st.session_state.get("umbral_prob", 0.95)
        umbral_mejora = st.session_state.get("umbral_mejora", 0.01)

        with res_tab1:
            resultado = st.session_state.calculadora.detectar_ganador(
                umbral_probabilidad=umbral_prob,
                umbral_mejora_minima=umbral_mejora
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Decisión final")
                if resultado.get("ganador") == "A":
                    st.success("🏆 El ganador es: Grupo A")
                elif resultado.get("ganador") == "B":
                    st.success("🏆 El ganador es: Grupo B")
                else:
                    st.info("⚖️ No hay ganador claro todavía")

                st.write(f"**Recomendación:** {resultado.get('decision', '—')}")
                st.write(f"**Razón:** {resultado.get('razon', '—')}")

                dias_con_datos = [
                    paso for paso in st.session_state.calculadora.historial
                    if paso.get('dia') and paso['dia'] != 'A priori'
                ]
                if len(dias_con_datos) < 6:
                    st.warning("⚠️ Has cargado menos de 6 días de datos. La recomendación puede cambiar al añadir más información.")

            with col2:
                if "probabilidad" in resultado:
                    st.metric("Probabilidad", f"{resultado['probabilidad']:.2%}")
                elif "probabilidad_b_mejor" in resultado:
                    st.metric("Probabilidad de que B sea mejor", f"{resultado['probabilidad_b_mejor']:.2%}")

                if "mejora_relativa" in resultado:
                    st.metric("Mejora relativa", f"{resultado['mejora_relativa']:.2%}")

            if len(st.session_state.calculadora.historial) > 0:
                ultimo = st.session_state.calculadora.historial[-1]

                st.subheader("Estado actual")
                colA, colB = st.columns(2)

                with colA:
                    st.write("**Grupo A**")
                    mean_a = ultimo['alpha_a'] / ultimo['beta_a']
                    st.metric("Tasa de conversión esperada", f"{mean_a:.4f}")
                    st.write(f"Parámetros: alpha={ultimo['alpha_a']:.1f}, beta={ultimo['beta_a']:.1f}")

                with colB:
                    st.write("**Grupo B**")
                    mean_b = ultimo['alpha_b'] / ultimo['beta_b']
                    st.metric("Tasa de conversión esperada", f"{mean_b:.4f}")
                    st.write(f"Parámetros: alpha={ultimo['alpha_b']:.1f}, beta={ultimo['beta_b']:.1f}")

        with res_tab2:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                st.session_state.calculadora.mostrar_historial_completo()
            st.code(buffer.getvalue(), language="text")

        with res_tab3:
            if len(st.session_state.calculadora.historial) > 0:
                st.subheader("Gráficos")

                dias_disponibles = [paso["dia"] for paso in st.session_state.calculadora.historial if "dia" in paso]
                if len(dias_disponibles) > 1:
                    dia_seleccionado = st.selectbox(
                        "Selecciona un día para ver sus gráficos:",
                        dias_disponibles[1:],
                        index=len(dias_disponibles) - 2
                    )

                    paso_seleccionado = None
                    for paso in st.session_state.calculadora.historial:
                        if paso.get("dia") == dia_seleccionado:
                            paso_seleccionado = paso
                            break
                else:
                    paso_seleccionado = st.session_state.calculadora.historial[-1]

                if paso_seleccionado is None:
                    st.info("No hay datos suficientes para mostrar gráficos.")
                else:
                    es_gamma = "trace" in paso_seleccionado
                    es_beta = "posterior" in paso_seleccionado and "comparacion" in paso_seleccionado

                    if es_gamma:
                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        tasa_a_samples = paso_seleccionado["trace"].posterior["tasa_clicks_a"].values.flatten()
                        tasa_b_samples = paso_seleccionado["trace"].posterior["tasa_clicks_b"].values.flatten()

                        sns.kdeplot(tasa_a_samples, label="Grupo A", fill=True, ax=ax1)
                        sns.kdeplot(tasa_b_samples, label="Grupo B", fill=True, ax=ax1)
                        ax1.set_title(f"{paso_seleccionado['dia']} - Distribuciones posteriores (Gamma–Poisson)")
                        ax1.set_xlabel("Tasa de clicks por visita")
                        ax1.legend()
                        st.pyplot(fig1)

                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        diff = paso_seleccionado["trace"].posterior["diferencia"].values.flatten()
                        sns.kdeplot(diff, label="Diferencia (B - A)", fill=True, ax=ax2)
                        ax2.axvline(0, color="black", linestyle="--")
                        ax2.set_title(f"{paso_seleccionado['dia']} - Diferencia de tasa de clicks")
                        ax2.set_xlabel("Diferencia en clicks por visita")
                        ax2.legend()
                        st.pyplot(fig2)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"Estadísticas del {paso_seleccionado['dia']}")
                            mean_a = paso_seleccionado["alpha_a"] / paso_seleccionado["beta_a"]
                            mean_b = paso_seleccionado["alpha_b"] / paso_seleccionado["beta_b"]
                            st.metric("Tasa esperada A", f"{mean_a:.4f}")
                            st.metric("Tasa esperada B", f"{mean_b:.4f}")
                        with col2:
                            if "uplift" in paso_seleccionado:
                                uplift = paso_seleccionado["uplift"]
                                st.subheader("Uplift (B vs A)")
                                st.metric("Media", f"{uplift['media']:.2%}")
                                st.metric("IC 95%", f"[{uplift['ic_95'][0]:.2%}, {uplift['ic_95'][1]:.2%}]")

                            prob_b_mejor = np.mean(diff > 0)
                            st.metric("Probabilidad de que B > A", f"{prob_b_mejor:.2%}")

                    elif es_beta:
                        post_a = paso_seleccionado["posterior"]["A"]
                        post_b = paso_seleccionado["posterior"]["B"]
                        comp = paso_seleccionado["comparacion"]

                        muestras_a = post_a["muestras"]
                        muestras_b = post_b["muestras"]
                        diff = comp["diff"]

                        fig1, ax1 = plt.subplots(figsize=(10, 5))
                        sns.kdeplot(muestras_a, label="Grupo A", fill=True, ax=ax1)
                        sns.kdeplot(muestras_b, label="Grupo B", fill=True, ax=ax1)
                        ax1.set_title(f"{paso_seleccionado['dia']} - Distribuciones posteriores (Beta–Binomial)")
                        ax1.set_xlabel("Tasa de conversión")
                        ax1.legend()
                        st.pyplot(fig1)

                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        diff_clean = diff[~np.isnan(diff)]
                        sns.kdeplot(diff_clean, label="Diferencia (B - A)", fill=True, ax=ax2)
                        ax2.axvline(0, color="black", linestyle="--")
                        ax2.set_title(f"{paso_seleccionado['dia']} - Diferencia de tasa de conversión")
                        ax2.set_xlabel("Diferencia en tasa de conversión")
                        ax2.legend()
                        st.pyplot(fig2)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"Estadísticas del {paso_seleccionado['dia']}")
                            st.metric("Tasa esperada A", f"{post_a['media']:.4f}")
                            st.metric("Tasa esperada B", f"{post_b['media']:.4f}")
                            st.write(f"IC95% A: [{post_a['ci'][0]:.4f}, {post_a['ci'][1]:.4f}]")
                            st.write(f"IC95% B: [{post_b['ci'][0]:.4f}, {post_b['ci'][1]:.4f}]")
                        with col2:
                            st.subheader("Comparación B vs A")
                            st.metric("Uplift medio", f"{comp['uplift_media']:.2%}")
                            st.write(f"IC95% uplift: [{comp['uplift_ci'][0]:.2%}, {comp['uplift_ci'][1]:.2%}]")
                            st.metric("Probabilidad de que B > A", f"{comp['prob_b_mejor']:.2%}")
                    else:
                        st.info("No hay información suficiente para mostrar gráficos para este modelo.")

                # Evolución tasas
                if len(st.session_state.calculadora.historial) > 2:
                    st.subheader("Evolución de tasas")

                    dias = []
                    tasas_a = []
                    tasas_b = []

                    for paso in st.session_state.calculadora.historial[1:]:
                        if "dia" not in paso:
                            continue
                        dias.append(paso["dia"])

                        if "trace" in paso:
                            tasa_a = paso["alpha_a"] / paso["beta_a"]
                            tasa_b = paso["alpha_b"] / paso["beta_b"]
                        elif "posterior" in paso:
                            tasa_a = paso["posterior"]["A"]["media"]
                            tasa_b = paso["posterior"]["B"]["media"]
                        else:
                            dias.pop()
                            continue

                        tasas_a.append(tasa_a)
                        tasas_b.append(tasa_b)

                    if dias:
                        fig3, ax3 = plt.subplots(figsize=(10, 5))
                        ax3.plot(dias, tasas_a, 'o-', label="Grupo A")
                        ax3.plot(dias, tasas_b, 'o-', label="Grupo B")
                        ax3.set_title("Evolución de tasas")
                        ax3.set_xlabel("Día")
                        ax3.set_ylabel("Tasa")
                        ax3.legend()
                        ax3.grid(True)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig3)

    # Footer
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px;">
    <p style="margin: 0; color: #555;">Idea y concepto: <strong>Claudia de la Cruz</strong> &nbsp;|&nbsp; Desarrollo: <strong>Pablo González</strong> &nbsp;|&nbsp; Desarrollo visual: <strong>Eduardo Hernández</strong></p>
    </div>
    """, unsafe_allow_html=True)


# =========================
# MAIN
# =========================
init_wizard_state()

if st.session_state.get("show_app", False):
    # Entramos en la calculadora (solo si el wizard eligió ruta disponible)
    render_calculadora_actual()
else:
    # Siempre empezamos en wizard
    render_wizard()
