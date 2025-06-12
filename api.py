import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import pandas as pd
import time
from tabulate import tabulate
from IPython.display import display

# Configuración de la página
st.set_page_config(
    page_title="Simulador Avanzado de Cortadora",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    /* Quitar fondo gris del track */
    div[data-baseweb="slider"] > div:first-child {
        background-color: white !important;
    }

    /* Slider general: sin bordes grises alrededor */
    .stSlider {
        background-color: white !important;
        padding: 0.3rem 0;
    }

    /* Mantener círculo rojo (por defecto de Streamlit) */
    .stSlider [data-baseweb="slider"] span {
        border: none;
    }

    /* Fondo blanco del sidebar */
    .sidebar .sidebar-content {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)




# Clase PIDController idéntica al código madre
class PIDController:
    def __init__(self, Kp, Ki, Kd,
                 setpoint=0.0,
                 output_limits=(None, None),
                 integral_limits=(None, None),
                 d_filter=0.1):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits    # (min, max)
        self.integral_limits = integral_limits
        self.d_filter = d_filter             # 0…1  (1 = sin filtrar)

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_d     = 0.0

    def compute(self, measured_value, dt):
        error = self.setpoint - measured_value

        # Proporcional
        P = self.Kp * error

        # Integral con anti-wind-up
        self.integral += error * dt
        if self.integral_limits[0] is not None:
            self.integral = max(self.integral_limits[0], self.integral)
        if self.integral_limits[1] is not None:
            self.integral = min(self.integral_limits[1], self.integral)
        I = self.Ki * self.integral

        # Derivativo con filtrado exponencial
        raw_d = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_d = self.d_filter * raw_d + (1 - self.d_filter) * self.prev_d
        D = self.Kd * self.prev_d

        # Salida y saturación
        out = P + I + D
        lo, hi = self.output_limits
        if lo is not None:
            out = max(lo, out)
        if hi is not None:
            out = min(hi, out)

        # actualizar históricos
        self.prev_error = error
        return out

# Función principal de simulación
def run_simulation(R, L, v, sigma, l_pasto, tf, Kp, Ki, Kd, seed, omega_fija=None):
    # Parámetros fijos del modelo
    delta_disco = 0.001      # espesor disco [m]
    delta_cuchilla = 0.003  # espesor cuchillas [m]
    w_cuchilla = 0.02       # ancho cuchilla  [m]
    rho_A = 7850            # densidad del acero [kg / m^3]
    lambda_val = 2.9 * 10**(-3)  # Densidad lineal del tallo [kg/m]
    rho = 552               # Densidad volumétrica del pasto [kg / m^3]
    z = 0.25                # contenido fraccionario de materia seca
    w = 23.5                # edad del pasto [semanas]
    
    # Cálculos iniciales
    m_plato = np.pi*R**2*delta_disco*rho_A           # masa del plato [kg]
    m_cuchilla = w_cuchilla*L*delta_cuchilla*rho_A   # masa de las cuchillas [kg]
    I = 0.5 * m_plato * R**2 + 1/6 * m_cuchilla * L**2  # Momento de inercia [kg·m²]
    W_corte = 2*(R+L)                                # Ancho de corte efectivo [m]
    
    # Parámetros del motor
    P = 50 * 745.7                                   # Potencia del motor [W]
    omega_nominal = 540 / 60 * 2 * np.pi             # Velocidad angular nominal del motor [rad/s]
    i = 1.5                                          # Relación de engranaje
    tau_max = P / (omega_nominal*i)                  # Torque máximo para las cuchillas [Nm]
    epsilon = 1e-6                                   # Para evitar división por cero

    # Configuración del PID
    pid = PIDController(Kp, Ki, Kd, output_limits=(0, tau_max), integral_limits=(-60, 60), d_filter=0.1)

    # Parámetros de simulación
    t0 = 0
    dt = 0.01
    n = int((tf - t0) / dt)
    t_values = np.linspace(t0, tf, n)

    # Generación de campo con parámetros variantes
    np.random.seed(seed)
    segment_duration = tf/6
    t_anchors = np.arange(t0, tf + segment_duration, segment_duration)
    
    # Variaciones de a% sobre parámetros base
    a = 0.125
    sigma_anchors  = sigma  * (1 + abs(a * np.random.randn(len(t_anchors))))
    l_pasto_anchors = l_pasto * (1 + abs(a * np.random.randn(len(t_anchors))))
    lambda_anchors = lambda_val * (1 + abs(a * np.random.randn(len(t_anchors))))
    rho_anchors = rho * (1 + abs(a * np.random.randn(len(t_anchors))))
    v_anchors = v * (1 + abs(a * np.random.randn(len(t_anchors))))

    # Interpolación suave
    sigma_interp  = interp1d(t_anchors, sigma_anchors,  kind='quadratic', fill_value="extrapolate")
    l_pasto_interp = interp1d(t_anchors, l_pasto_anchors, kind='quadratic', fill_value="extrapolate")
    lambda_interp = interp1d(t_anchors, lambda_anchors, kind='quadratic', fill_value="extrapolate")
    rho_interp = interp1d(t_anchors, rho_anchors, kind='quadratic', fill_value="extrapolate")
    v_interp = interp1d(t_anchors, v_anchors, kind='quadratic', fill_value="extrapolate")

    sigma_profile  = sigma_interp(t_values)
    l_pasto_profile = l_pasto_interp(t_values)
    lambda_profile = lambda_interp(t_values)
    rho_profile = rho_interp(t_values)
    v_profile = v_interp(t_values)

    # Rocas y clutch
    Area_total_esperada = v * W_corte * tf /3.6 #[m^2]
    densidad_rocas = 0.001              # [rocas / m^2]
    N_impactos = np.random.poisson(densidad_rocas * Area_total_esperada)
    N_impactos = min(N_impactos, 10)
    t_impactos = np.random.choice(t_values, size=N_impactos, replace=False)
    impactos = [(t_i, np.random.uniform(400, 1250)) for t_i in t_impactos]

    alpha = 20
    beta = 2

    def torque_rocas_suave(t_actual):
        torque_total = 0
        for t_impacto, A in impactos:
            if t_actual >= t_impacto:
                dt = t_actual - t_impacto
                pulso = A * (1 - np.exp(-alpha * dt)) * np.exp(-beta * dt)
                torque_total += pulso
        return torque_total

    def clutch_soft_saturation(torque_neto, omega, torque_limite=1000, n=4, omega_min=22):
        if torque_neto >= 0:
            return torque_neto
        escala_omega = max(omega / omega_min, 0.1)**6
        torque_limite_efectivo = torque_limite * escala_omega
        return torque_neto / (1 + (abs(torque_neto)/torque_limite_efectivo)**n)

    # Sistema de Ecuaciones diferenciales
    def f1(omega):
        return omega

    def f2(omega, t):
        omega = max(omega, epsilon)

        sigma = sigma_profile[t]
        l_pasto = l_pasto_profile[t]
        lambda_val = lambda_profile[t]
        rho = rho_profile[t]
        Sigma = sigma * lambda_val * l_pasto
        v = v_profile[t]

        a = 10**(6) * lambda_val / rho
        E_c = (-39.6 + 8.18*a - 253*z + 3.51*w) / 1000
        r = np.sqrt(lambda_val / (rho * np.pi))
        k = 0.5 * np.pi * rho * E_c / lambda_val
        b = (l_pasto + 0.05) * 0.15

        v_min = np.sqrt(2*E_c / (b*lambda_val))
        omega_min = v_min / (R+L)

        masa_tallo = lambda_val * l_pasto
        N_dot = (Sigma * v * W_corte) / masa_tallo

        v_cuchilla = omega * (R+L)

        P_viento = 0.188 * 10**(-3) * omega**3 * W_corte
        g = (np.tanh(omega/(4*v_min)))

        if v_cuchilla > v_min:
            P_corte_pasto = 2 * k * r**2 * N_dot
            P_deflexion_pasto = ((b * lambda_val * v_cuchilla**2 ) * (1 - np.sqrt(1 - (4 * k * r**2) / (b * lambda_val * v_cuchilla**2))) - 2*k*r**2) * N_dot
            P_ac = g * 1000 * 0.0478 * Sigma * v **(2) * (0.749) ** (omega/100) * W_corte
        else:
            P_corte_pasto = 0
            P_deflexion_pasto = 0
            P_ac = g * 1000 * 0.0478 * Sigma * v **(2) * (0.749) ** (omega/100) * W_corte

        def P_total(omega):
            P_ac = 1000 * 0.0478 * Sigma * v**2 * (0.749)**(omega / 100) * W_corte
            P_viento = 0.188e-3 * omega**3 * W_corte
            return P_ac + P_viento

        omega_optimizada = minimize_scalar(P_total, bounds=(1, 300), method='bounded')

        if omega_fija is not None:
            omega_deseada = float(omega_fija)
        else:
            if omega_optimizada.x < (omega_min * 1.5):
                omega_deseada = omega_min * 1.5
            else:
                omega_deseada = omega_optimizada.x

        pid.setpoint = round(omega_deseada, 0)
        torque_maximo = min(P / omega, tau_max)
        pid.output_limits = (0, torque_maximo)
        torque_cuchillas = pid.compute(omega, dt)

        t_actual = t_values[t]
        torque_roca = torque_rocas_suave(t_actual)

        torque_neto = torque_cuchillas - (P_corte_pasto + P_deflexion_pasto + P_ac + P_viento)/omega - torque_roca
        torque_neto_limitado = clutch_soft_saturation(torque_neto, omega)

        return (torque_neto_limitado/ I, torque_cuchillas, P_ac / omega, P_viento / omega, 
                torque_roca, P_corte_pasto / omega, P_deflexion_pasto / omega, 
                omega_optimizada.x, omega_min)

    # Resolución de ecuaciones con RK4
    theta = np.zeros(n)
    omega = np.zeros(n)
    torque_neto = np.zeros(n)
    torque_cuchillas = np.zeros(n)
    torque_pasto = np.zeros(n)
    torque_viento = np.zeros(n)
    Potencia_total = 0
    Area_barrida = 0
    torque_rocas = np.zeros(n)
    torque_corte = np.zeros(n)
    torque_deflexion = np.zeros(n)
    omega_optima = np.zeros(n)
    omega_minima = np.zeros(n)

    for k in range(n - 1):
        # k1
        k1_1 = f1(omega[k])
        k1_2, tau1, tp1, tv1, tr1, tc1, td1, oo1, om1 = f2(omega[k], k)

        # k2
        k2_1 = f1(omega[k] + dt*0.5*k1_2)
        k2_2, tau2, tp2, tv2, tr2, tc2, td2, oo2, om2 = f2(omega[k] + dt*0.5*k1_2, k)

        # k3
        k3_1 = f1(omega[k] + dt*0.5*k2_2)
        k3_2, tau3, tp3, tv3, tr3, tc3, td3, oo3, om3 = f2(omega[k] + dt*0.5*k2_2, k)

        # k4
        k4_1 = f1(omega[k] + dt*k3_2)
        k4_2, tau4, tp4, tv4, tr4, tc4, td4, oo4, om4 = f2(omega[k] + dt*k3_2, k)

        # actualizar
        theta[k+1] = theta[k] + dt*(k1_1 + 2*k2_1 + 2*k3_1 + k4_1)/6
        omega[k+1] = omega[k] + dt*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)/6

        torque_neto[k+1] = I*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)/6
        torque_cuchillas[k+1] = (tau1 + 2*tau2 + 2*tau3 + tau4)/6
        torque_pasto[k+1] = (tp1 + 2*tp2 + 2*tp3 + tp4)/6
        torque_viento[k+1] = (tv1 + 2*tv2 + 2*tv3 + tv4)/6
        torque_rocas[k+1] = (tr1 + 2*tr2 + 2*tr3 + tr4)/6
        torque_corte[k+1] = (tc1 + 2*tc2 + 2*tc3 + tc4)/6
        torque_deflexion[k+1] = (td1 + 2*td2 + 2*td3 + td4)/6
        omega_optima[k+1] = (oo1 + 2*oo2 + 2*oo3 + oo4)/6
        omega_minima[k+1] = (om1 + 2*om2 + 2*om3 + om4)/6

        Potencia_total += omega[k+1] * torque_cuchillas[k+1]
        Area_barrida += v_profile[k+1] * dt * W_corte /3.6

    # Resultados
    df = pd.DataFrame({
        "Parámetro": [
            "Semilla utilizada",
            "Trabajo total (kJ)",
            "Área total barrida (m²)",
            "Trabajo por área (kJ/m²)",
            "Omega óptima promedio",
            "Omega mínima promedio",
            "Blade Usage promedio",
            "Omega promedio real"
        ],
        "Valor": [
            seed,
            f"{Potencia_total * 1e-3 * dt:.2f}",
            f"{Area_barrida:.2f}",
            f"{Potencia_total * 1e-3 * dt / Area_barrida:.2f}",
            f"{np.mean(omega_optima):.2f}",
            f"{np.mean(omega_minima):.2f}",
            f"{np.mean(omega * L / (np.pi * v_profile)):.2f}",
            f"{np.mean(omega):.2f}"
        ]
    })

    # Creación de figuras idénticas al código madre
    figs = []

    # Figura 1: Velocidad angular, Torque y Potencia
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(t_values, omega, label='ω(t)', color='orange')
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('Velocidad angular [rad/s]')
    ax1.set_title('Velocidad angular vs Tiempo')
    ax1.grid(True)
    
    ax2.plot(t_values, torque_cuchillas, label='Tau(t)', color='orange')
    ax2.set_xlabel('tiempo [s])')
    ax2.set_ylabel('Torque de las cuchillas [Nm]')
    ax2.set_title('Torque de las cuchillas vs tiempo')
    ax2.grid(True)
    
    ax3.plot(t_values, torque_cuchillas * omega, label='P(t)', color='purple')
    ax3.set_xlabel('tiempo [s])')
    ax3.set_ylabel('Potencia del motor [W]')
    ax3.set_title('Potencia del motor vs Tiempo')
    ax3.grid(True)
    
    fig1.tight_layout()
    figs.append(fig1)

    # Figura 2: Variaciones del campo de trabajo
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 2))
    
    ax1.plot(t_values, sigma_profile * l_pasto_profile * lambda_profile, label='σ(t)')
    ax1.set_ylabel('σ [kg/m²]')
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_title('Densidad de pasto vs Tiempo')
    ax1.grid(True)
    
    ax2.plot(t_values, v_profile, label='v(t)', color='red')
    ax2.set_ylabel('v [km/h]')
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_title('Velocidad de avance vs Tiempo')
    ax2.grid(True)
    
    figs.append(fig2)

    # Figura 3: Torques detallados
    fig3, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(15, 4))
    
    ax1.plot(t_values, torque_pasto, label='Arrastre de pasto', color='forestgreen')
    ax1.plot(t_values, torque_corte, label='Corte de pasto', color='limegreen')
    ax1.plot(t_values, torque_deflexion, label='Deflexion de pasto', color='olive')
    ax1.set_xlabel('tiempo [s])')
    ax1.set_ylabel('Torque del pasto [Nm]')
    ax1.set_title('Torque del pasto vs tiempo')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t_values, torque_viento, label='TauV(t)', color='blue')
    ax2.set_xlabel('tiempo [s])')
    ax2.set_ylabel('Torque del viento [Nm]')
    ax2.set_title('Torque del viento vs tiempo')
    ax2.grid(True)
    
    ax3.plot(t_values, torque_rocas, label='TauV(t)', color='black')
    ax3.set_xlabel('tiempo [s])')
    ax3.set_ylabel('Torque de las rocas [Nm]')
    ax3.set_title('Torque de las rocas vs Tiempo')
    ax3.grid(True)
    
    ax4.plot(t_values, torque_neto, label='Torque neto', color='green')
    ax4.set_xlabel('Tiempo [s]')
    ax4.set_ylabel('Torque neto [Nm]')
    ax4.set_title('Torque neto vs Tiempo')
    ax4.grid(True)
    
    figs.append(fig3)

    return df, figs

# Interfaz de Streamlit
st.title("🌱 Simulador Avanzado de Cortadora de Pasto")
st.markdown("""
Esta aplicación simula el comportamiento de una cortadora de pasto con control PID, 
teniendo en cuenta múltiples factores físicos y ambientales.
""")

with st.sidebar:
    st.header("⚙️ Parámetros de Configuración")
    
    st.subheader("🔧 Geometría")
    R = st.slider("Radio del plato [m]", 0.0, 1.0, 0.5, 0.01)
    L = st.slider("Largo de la cuchilla [m]", 0.0, 1.0, 0.5, 0.01)
    
    st.subheader("🚜 Operación")
    v = st.slider("Velocidad de avance [km/s]", 0.0, 12.0, 4.0, 0.1)
    sigma = st.slider("Densidad superficial de tallos [tallos/m²]", 1500, 3000, 2210, 10)
    l_pasto = st.slider("Largo del pasto [m]", 0.1, 1.0, 0.2, 0.01)
    tf = st.slider("Tiempo total de simulación [s]", 10, 300, 30, 1)
    
    st.subheader("🎛 Control PID")
    Kp = st.slider("Ganancia Proporcional (Kp)", 0.0, 5.0, 3.5, 0.1)
    Ki = st.slider("Ganancia Integral (Ki)", 0.0, 5.0, 4.0, 0.1)
    Kd = st.slider("Ganancia Derivativa (Kd)", 0.0, 5.0, 0.65, 0.01)
    
    st.subheader("⚡ Configuración Avanzada")
    seed = st.number_input("Semilla aleatoria", value=42, min_value=0, step=1)
    omega_fija = st.text_input("Omega fija [rad/s] (dejar vacío para cálculo automático)", "")
    
    run_button = st.button("▶️ Ejecutar Simulación", type="primary")

if run_button:
    with st.spinner('Simulando... Esto puede tomar unos momentos...'):
        try:
            df_result, figures = run_simulation(
                R, L, v, sigma, l_pasto, tf, Kp, Ki, Kd, seed, 
                omega_fija if omega_fija else None
            )
            
            st.success("Simulación completada exitosamente!")
            
            # Mostrar resultados
            st.subheader("📊 Resultados de la Simulación")
            st.dataframe(df_result)

            # Mostrar gráficas
            st.subheader("📈 Visualización de Resultados")
            
            cols = st.columns(1)
            with cols[0]:
                st.pyplot(figures[0])  # Gráficas de velocidad, torque y potencia
            
            st.pyplot(figures[1])      # Variaciones del campo
            
            cols = st.columns(1)
            with cols[0]:
                st.pyplot(figures[2])  # Torques detallados
                
        except Exception as e:
            st.error(f"Error durante la simulación: {str(e)}")
else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en 'Ejecutar Simulación' para comenzar.")

st.markdown("---")
st.markdown("""
**Notas:**
- Los cálculos incluyen efectos de inercia, resistencia del pasto, viento e impactos con rocas.
- El controlador PID ajusta automáticamente el torque para mantener la velocidad óptima.
- Para forzar una velocidad específica, ingrésala en el campo "Omega fija".
""")