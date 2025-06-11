import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constantes
R = 0.5
L = 0.5
delta_disco = 0.001
delta_cuchilla = 0.003
w_cuchilla = 0.02
rho_A = 7850
m_plato = np.pi * R**2 * delta_disco * rho_A
m_cuchilla = w_cuchilla * L * delta_cuchilla * rho_A
I = 0.5 * m_plato * R**2 + 1/6 * m_cuchilla * L**2


# -------------------------------------------------------------------
# PID simplificado
class PID:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.output_limits = output_limits
        self.setpoint = 0.0
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, value, dt):
        error = self.setpoint - value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return np.clip(output, *self.output_limits)

# -------------------------------------------------------------------
def run_simulation(R, L, v, sigma, l_pasto, tf, Kp, Ki, Kd, seed, omega_manual=None):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    np.random.seed(seed)
    dt = 0.01
    t = np.arange(0, tf, dt)
    n = len(t)
    
    rho = 552
    z = 0.25
    w = 23.5
    lambda_val = 2.9e-3
    P = 50 * 745.7
    omega_nominal = 540 / 60 * 2 * np.pi
    tau_max = P / (omega_nominal * 1.5)
    W_corte = 2 * (R + L)
    rho_A = 7850
    delta_disco = 0.001
    delta_cuchilla = 0.003
    w_cuchilla = 0.02
    epsilon = 1e-6

    m_plato = np.pi * R**2 * delta_disco * rho_A
    m_cuchilla = w_cuchilla * L * delta_cuchilla * rho_A
    I = 0.5 * m_plato * R**2 + (1/6) * m_cuchilla * L**2

    t_anchors = np.linspace(0, tf, 7)
    a = 0.125

    def gen_profile(base): return base * (1 + abs(a * np.random.randn(len(t_anchors))))
    sigma_anchors = gen_profile(sigma)
    l_anchors = gen_profile(l_pasto)
    lambda_anchors = gen_profile(lambda_val)
    rho_anchors = gen_profile(rho)
    v_anchors = gen_profile(v)

    sigma_p = interp1d(t_anchors, sigma_anchors, kind='quadratic')(t)
    l_pasto_p = interp1d(t_anchors, l_anchors, kind='quadratic')(t)
    lambda_p = interp1d(t_anchors, lambda_anchors, kind='quadratic')(t)
    rho_p = interp1d(t_anchors, rho_anchors, kind='quadratic')(t)
    v_p = interp1d(t_anchors, v_anchors, kind='quadratic')(t)

    N = int(v * W_corte * tf * 0.001)
    t_impactos = np.random.choice(t, size=min(N, 10), replace=False)
    impactos = [(ti, np.random.uniform(400, 1250)) for ti in t_impactos]

    def torque_rocas(ti):
        alpha, beta = 20, 2
        return sum(A * (1 - np.exp(-alpha*(t_i - ti))) * np.exp(-beta*(t_i - ti))
                   for t_i, A in impactos if ti >= t_i)

    def clutch_soft(torque_neto, omega, torque_lim=1000, n=4, omega_min=22):
        if torque_neto >= 0: return torque_neto
        escala = max(omega / omega_min, 0.1)**6
        torque_eff = torque_lim * escala
        return torque_neto / (1 + abs(torque_neto / torque_eff)**n)

    class PID:
        def __init__(self, Kp, Ki, Kd):
            self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
            self.setpoint = 0
            self.integral = 0
            self.prev_error = 0

        def compute(self, value, dt):
            error = self.setpoint - value
            self.integral += error * dt
            derivative = (error - self.prev_error) / dt
            self.prev_error = error
            return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    pid = PID(Kp, Ki, Kd)

    omega = np.zeros(n)
    torque = np.zeros(n)
    torque_pasto = np.zeros(n)
    torque_viento = np.zeros(n)
    torque_roca = np.zeros(n)
    potencia = np.zeros(n)

    for k in range(n - 1):
        sig = sigma_p[k]
        lp = l_pasto_p[k]
        lam = lambda_p[k]
        dens = rho_p[k]
        vel = v_p[k]
        Sigma = sig * lam * lp
        v_cuchilla = max(omega[k] * (R + L), epsilon)

        a = 1e6 * lam / dens
        E_c = (-39.6 + 8.18*a - 253*z + 3.51*w) / 1000
        r = np.sqrt(lam / (dens * np.pi))
        kf = 0.5 * np.pi * dens * E_c / lam
        b = (lp + 0.05) * 0.15

        v_min = np.sqrt(2 * E_c / (b * lam))
        omega_min = v_min / (R + L)

        masa_tallo = lam * lp
        N_dot = (Sigma * vel * W_corte) / masa_tallo

        g = np.tanh(omega[k] / (4 * v_min))
        if v_cuchilla > v_min:
            P_ac = g * 1000 * 0.0478 * Sigma * vel**2 * (0.749)**(omega[k] / 100) * W_corte
        else:
            P_ac = 0

        P_viento = 0.188e-3 * omega[k]**3 * W_corte
        potencia[k] = P_ac + P_viento

        def P_total(omega_):
            return 1000 * 0.0478 * Sigma * vel**2 * (0.749)**(omega_ / 100) * W_corte + 0.188e-3 * omega_**3 * W_corte

        if omega_manual:
            omega_deseada = float(omega_manual)
        else:
            result = minimize_scalar(P_total, bounds=(1, 300), method='bounded')
            omega_opt = result.x if result.success else 100
            omega_deseada = max(omega_opt, omega_min * 1.5)

        pid.setpoint = omega_deseada
        tau_max = min(P / max(omega[k], epsilon), 50)
        pid_out = np.clip(pid.compute(omega[k], dt), 0, tau_max)
        torque[k] = pid_out

        tau_pasto = 0.01 * sig * lp
        tau_viento = 0.5 * 1.25 * (vel / 3.6)**2 * 0.02
        tau_roca = torque_rocas(t[k])
        tau_neto = pid_out - tau_pasto - tau_viento - tau_roca
        tau_neto = clutch_soft(tau_neto, omega[k])

        domega = tau_neto / I
        omega[k+1] = omega[k] + domega * dt

        torque_pasto[k] = tau_pasto
        torque_viento[k] = tau_viento
        torque_roca[k] = tau_roca

    trabajo_total = np.trapz(potencia, t) / 1000
    area_total = np.trapz(v_p, t) * W_corte

    df = pd.DataFrame({
        "Parámetro": [
            "Semilla utilizada",
            "Trabajo total (kJ)",
            "Área total barrida (m²)",
            "Trabajo por área (kJ/m²)",
            "Omega promedio real",
            "Omega mínima",
            "Blade Usage promedio",
        ],
        "Valor": [
            seed,
            f"{trabajo_total:.2f}",
            f"{area_total:.2f}",
            f"{trabajo_total / area_total:.2f}",
            f"{np.mean(omega):.2f}",
            f"{np.min(omega):.2f}",
            f"{np.mean(omega * L / (np.pi * v_p)):.2f}",
        ],
    })

    figs = []

    # 1. ω(t)
    fig1, ax1 = plt.subplots()
    ax1.plot(t, omega)
    ax1.set_title("Velocidad angular ω(t)")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("ω [rad/s]")
    ax1.grid(True)
    figs.append(fig1)

    # 2. Torque de cuchillas
    fig2, ax2 = plt.subplots()
    ax2.plot(t, torque)
    ax2.set_title("Torque de las cuchillas")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Torque [Nm]")
    ax2.grid(True)
    figs.append(fig2)

    # 3. Potencia
    fig3, ax3 = plt.subplots()
    ax3.plot(t, potencia)
    ax3.set_title("Potencia del motor")
    ax3.set_xlabel("Tiempo [s]")
    ax3.set_ylabel("Potencia [W]")
    ax3.grid(True)
    figs.append(fig3)

    # 4. Densidad y velocidad
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 3))
    ax4a.plot(t, sigma_p * l_pasto_p * lambda_p)
    ax4a.set_title("Densidad de pasto σ·l·λ")
    ax4a.grid(True)
    ax4b.plot(t, v_p)
    ax4b.set_title("Velocidad de avance v(t)")
    ax4b.grid(True)
    figs.append(fig4)

    # 5. Torques detallados
    fig5, axs = plt.subplots(1, 4, figsize=(15, 4))
    axs[0].plot(t, torque_pasto)
    axs[0].set_title("Torque pasto")
    axs[0].grid(True)
    axs[1].plot(t, torque_viento)
    axs[1].set_title("Torque viento")
    axs[1].grid(True)
    axs[2].plot(t, torque_roca)
    axs[2].set_title("Torque rocas")
    axs[2].grid(True)
    axs[3].plot(t, torque)
    axs[3].set_title("Torque neto")
    axs[3].grid(True)
    figs.append(fig5)

    return df, figs


# -------------------------------------------------------------------
# Interfaz Streamlit
st.set_page_config(page_title="Simulador Cortadora", layout="wide")
st.title("Simulación de cortadora de pasto ✂️")

with st.sidebar:
    st.header("🔧 Parámetros")
    R = st.slider("Radio del plato R [m]", 0.0, 1.0, 0.5)
    L = st.slider("Largo de la cuchilla L [m]", 0.0, 1.0, 0.5)
    v = st.slider("Velocidad de avance v [m/s]", 0.0, 12.0, 4.0)
    sigma = st.slider("Densidad superficial de tallos σ [tallos/m²]", 1500, 3000, 2210)
    l_pasto = st.slider("Largo del pasto l [m]", 0.1, 1.0, 0.2)
    tf = st.slider("Tiempo total de simulación [s]", 10, 300, 30)
    Kp = st.slider("Kp", 0.0, 5.0, 3.5)
    Ki = st.slider("Ki", 0.0, 5.0, 4.0)
    Kd = st.slider("Kd", 0.0, 5.0, 0.65)
    seed = st.number_input("Semilla aleatoria", value=42, step=1)
    omega_manual = st.text_input("Omega fija (dejar vacío para automático)", value="")

    run_sim = st.button("🚀 Correr simulación")

if run_sim:
    with st.spinner("Simulando..."):
        df, figs = run_simulation(R, L, v, sigma, l_pasto, tf, Kp, Ki, Kd, seed, omega_manual if omega_manual else None)

    st.subheader("📊 Resultados")
    st.dataframe(df)

    st.subheader("📈 Gráficas")
    for fig in figs:
        st.pyplot(fig)

    st.success("Simulación finalizada")
else:
    st.info("Ajusta los parámetros y presiona 'Correr simulación'.")

st.markdown("---\n*App creada en 🇲🇽 *")


