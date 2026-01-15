#######################################################################
######### Funciones auxiliares para estudio de procesos ARMA ##########
#######################################################################

#Módulos requeridos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_process import arma_impulse_response, arma_acf, ArmaProcess, arma_acovf, arma2ar, arma2ma
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import shapiro
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot 
from statsmodels.stats.diagnostic import het_arch
from typing import Optional, Union



#####################################################
### Función que simula el lanzamiento de un dado  ###
#####################################################
def sim_dado(num_lanzamientos=1000,num_realizaciones=1,semilla=None, graficar=True, resultados=False,tam_fig=(14,4)):
    #Valor semilla para controlar la generación de números pesudoaleatorios si se desea
    if semilla is not None:
        np.random.seed(semilla) 

    #Simulamos el número de realizaciones solicitadas
    realizaciones = [np.random.randint(1, 7, size=num_lanzamientos) for i in range(num_realizaciones)]
    
    #Gráfico de la realización si se desea
    if graficar:
        plt.figure(figsize=tam_fig)
        for i in range(num_realizaciones):
            plt.plot(range(1, num_lanzamientos + 1), realizaciones[i])
        plt.title(f"Simulación de un proceso estocástico: lanzamiento de un dado (número de realizaciones = {num_realizaciones})")
        plt.xlabel("t: días")
        plt.ylabel("Resultado del dado")
        plt.grid()
        plt.show()
        
    if resultados:
        return realizaciones[0]




################################################################
### Función que obtiene el promedio (acumulado) y lo grafica ###
################################################################
def prom_acum(serie,
              resultados=False,
              mu=None,
              tam_fig=(14,5)):
    serie_pa=np.cumsum(serie)/np.arange(1, len(serie) + 1)
    plt.figure(figsize=tam_fig)
    plt.plot(range(1, len(serie_pa) + 1), serie_pa, linewidth=2, color="green",label="Promedio acumulado")
    plt.title("Promedio acumulado")
    plt.xlabel("Tiempo (t)")
    plt.ylabel("Promedio acumulado hasta el tiempo t")
    if mu is not None:
        plt.axhline(mu,color="red",label="Media teórica")
        plt.legend()
    plt.xticks()
    plt.grid()
    plt.show()
    if resultados:
        return serie_pa



####################################################################
### Función para graficar la IRF teórica de un proceso ARMA(p,q) ###
####################################################################
def irf_teorica_arma(pol_ma=[1],
                     pol_ar=[1],
                     orden_max=5,
                     graficar=True,
                     resultados=False,
                     tam_fig=(8,3)
                     ):
    
    irf_arma_est=arma_impulse_response(ar=np.array(pol_ar), 
                                     ma=np.array(pol_ma), 
                                     leads=orden_max+1)
    #Dibujamos la IRF teórica del proceso
    if graficar:
        plt.figure(figsize=tam_fig)
        plt.stem(range(len(irf_arma_est)),irf_arma_est)
        plt.title(f"Función impulso-respuesta (IRF) teórica de un proceso ARMA({len(pol_ar)-1},{len(pol_ma)-1})")
        plt.xlabel("")
        plt.ylabel("")
        for i, val in enumerate(irf_arma_est):
            if i>0:
                plt.text(i, val + 0.05*val, f"{val:.2f}", ha='center', va='bottom' if val >= 0 else 'top')
        plt.grid()
        plt.show()
    if resultados:
        return irf_arma_est


###############################################################
### Función para graficar ACF teórica de procesos ARMA(p,q) ###
###############################################################
def acf_teorica_arma(pol_ma=[1],
                     pol_ar=[1],
                     orden_max=5,
                     graficar=True,
                     resultados=False,
                     tam_fig=(8,3)
                     ):
    
    acf_teorica_arma_est=arma_acf(ar=np.array(pol_ar), 
                                ma=np.array(pol_ma), 
                                lags=orden_max+1)

    #Dibujamos la ACF teórica del proceso
    if graficar:
        plt.figure(figsize=tam_fig)
        plt.stem(range(len(acf_teorica_arma_est)),acf_teorica_arma_est)
        plt.title(f"Correlograma teórico ACF de un proceso ARMA({len(pol_ar)-1},{len(pol_ma)-1})")
        plt.xlabel("Rezagos (j)")
        plt.ylabel("ACF(j) teórica")
        for i, val in enumerate(acf_teorica_arma_est):
            if i>0:
                plt.text(i, val + 0.05*val, f"{val:.2f}", ha='center', va='bottom' if val >= 0 else 'top')
        plt.grid()
        plt.show()

    if resultados:
        return acf_teorica_arma_est
    
######################################################
### Definimos función que simula proceso ARMA(p,q) ###
######################################################
def simulacion_arma(cte=0,
           pol_ar=[1],
           pol_ma=[1],
           sigma2=1,
           semilla=None,
           num_obs=100, 
           graficar=True, 
           graficar_teoricos=True, 
           resultados=False,
           tam_fig=(8,3)
           ):
    
    #Calculamos resultados de interés
    muf=cte/np.sum(pol_ar)
    gamma0f=arma_acovf(ar=np.array(pol_ar),
                       ma=np.array(pol_ma),
                       nobs=1,
                       sigma2=sigma2)[0]
    desv_estf=gamma0f**0.5
    LI95f = muf - 1.96*desv_estf
    LS95f = muf + 1.96*desv_estf
    
    #Simulación del proceso MA paso a paso
    if semilla is not None:
        np.random.seed(semilla)
    
    proceso_arma_est = ArmaProcess(ar=np.array(pol_ar), 
                                 ma=np.array(pol_ma)) 
    simulacionpp_arma_est=proceso_arma_est.generate_sample(nsample=num_obs, scale=sigma2**0.5)
    simulacionpp_arma_est += muf #Agregamos la media
    
    if graficar:
        plt.figure(figsize=tam_fig)
        plt.plot(range(1,len(simulacionpp_arma_est)+1),simulacionpp_arma_est,linewidth=1)
        plt.title(f"Simulación de proceso ARMA({len(pol_ar)-1},{len(pol_ma)-1})")
        plt.xlabel("Tiempo (t)")
        plt.ylabel("Y(t)")
        if graficar_teoricos:
            plt.axhline(muf, color="green", linestyle="-", label=f"Media (teórica) = {muf:.2f}")
            plt.axhline(LI95f, color="red", linestyle="--", label=f"IC 95% inferior (teórico) = {LI95f:.2f}")
            plt.axhline(LS95f, color="red", linestyle="--", label=f"IC 95% superior (teórico) = {LS95f:.2f}")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))  
        plt.grid()
        plt.show()
        
    if resultados:
        return simulacionpp_arma_est



##############################################################################
### Función que estima la PACF teórica usando el algoritmo Levinson-Durbin ###
####################### (Enders, 2015 , cap.2) ###############################
##############################################################################
def pacf_levinson_durbin(rho, nlags, tol=1e-12):
    if nlags>=len(rho):
        return print(f"Insuficientes términos en la ACF teórica (nlags máximo puede ser {len(rho)-1})")
    """
    Levinson–Durbin usando convención phi[k,j] (k=orden, j=rezago), como en Enders (2015, cap.2).
    """
    phi = np.zeros((nlags + 1, nlags + 1))
    pacf = np.zeros(nlags + 1)

    pacf[0] = 1.0
    if nlags >= 1:
        phi[1, 1] = rho[1]
        pacf[1] = rho[1]

    for k in range(2, nlags + 1):
        num = rho[k] - np.dot(phi[k-1, 1:k], rho[1:k][::-1])
        den = 1.0 - np.dot(phi[k-1, 1:k], rho[1:k])
        if abs(den) < tol:
            den = np.sign(den) * tol if den != 0 else tol

        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]

        pacf[k] = phi[k, k]

    return pacf


######################################################
### Función para obtener PACF de proceso ARMA(p,q) ###
######################################################
def pacf_teorica_arma(pol_ma=[1],
                    pol_ar=[1],
                    orden_max=5,
                    graficar=True,
                    resultados=False,
                    tam_fig=(8, 3)
                    ):
    
    rhof = acf_teorica_arma(pol_ar=pol_ar,
                            pol_ma=pol_ma,
                            orden_max=orden_max,
                            graficar=False,
                            resultados=True)
    pacf_teorica_arma_est = pacf_levinson_durbin(rhof, nlags=orden_max)
    
    #Graficamos la PACF teórica del proceso
    if graficar:
        plt.figure(figsize=tam_fig)
        plt.stem(range(len(pacf_teorica_arma_est)),pacf_teorica_arma_est)
        plt.title(f"Correlograma teórico PACF de un proceso ARMA({len(pol_ar)-1},{len(pol_ma)-1})")
        plt.xlabel("Rezagos (j)")
        plt.ylabel("PACF(j) teórica")
        for i, val in enumerate(pacf_teorica_arma_est):
            if i>0:
                plt.text(i, val + 0.05*val, f"{val:.2f}", ha='center', va='bottom' if val >= 0 else 'top')
        plt.grid()
        plt.show()

    if resultados:
        return pacf_teorica_arma_est
    

############################################################
### Función para reescribir proceso ARMA(p,q) en AR(inf) ###
############################################################
def trans_arma_arinf(mu=0,
                    pol_ar=[1],
                    pol_ma=[1],
                    orden_max=10, 
                    resultados=False):
    print("\n=== Representación AR(∞) ===")
    
    #Genero constante solo si no tengo componente AR
    if len(pol_ar) == 1:
        constante_ar=mu/(np.sum(pol_ma))
        print(f"Constante = {constante_ar:.4f}")
    
    coef_arinf=arma2ar(ar=np.array(pol_ar), 
                       ma=np.array(pol_ma), 
                       lags=orden_max+1)
    len(coef_arinf)
    for i,val in enumerate(coef_arinf):
        if i>0:
            print(f"Coeficiente AR({i}) = {-val:.6f}")
    
    if resultados:
        return {"constante":constante_ar,"coeficientes":coef_arinf}
    

############################################################
### Función para reescribir proceso ARMA(p,q) en MA(inf) ###
############################################################
def trans_arma_mainf(cte=0,
                     pol_ar=[1],
                     pol_ma=[1],
                     orden_max=10, 
                     resultados=False):
    print("\n=== Representación MA(∞) ===")
    
    #Genero constante solo si no tengo componente MA
    if len(pol_ma) ==1:
        constante_ma=cte/(np.sum(pol_ar))
        print(f"Constante = {constante_ma:.4f}")
    
    coef_mainf=arma2ma(ar=np.array(pol_ar), 
                       ma=np.array(pol_ma), 
                       lags=orden_max+1)
    len(coef_mainf)
    for i,val in enumerate(coef_mainf):
        if i>0:
            print(f"Coeficiente MA({i}) = {val:.6f}")
    
    if resultados:
        return {"constante":constante_ma,"coeficientes":coef_mainf}
    



    
###########################################################    
### Función que simula un proceso de caminata aleatoria ###
###########################################################
def simulacion_caminata_aleatoria(valor_inic=0, 
                                  deriva=0, 
                                  sigma2=1, 
                                  num_obs=100, 
                                  semilla=None, 
                                  graficar=True, 
                                  resultados=False,
                                  tam_fig=(14, 5),
                                  num_realizaciones=1):

    if semilla is not None:
        np.random.seed(semilla)

    # Guardar múltiples simulaciones
    sims = []

    for _ in range(num_realizaciones):
        sim = valor_inic + np.cumsum(
            np.random.normal(loc=deriva, scale=sigma2**0.5, size=num_obs)
        )
        sims.append(sim)

    sims = np.array(sims)  # shape: (num_realizaciones, num_obs)

    # Graficar
    if graficar:
        plt.figure(figsize=tam_fig)
        
        for i in range(num_realizaciones):
            plt.plot(range(1, num_obs + 1), sims[i], linewidth=1)

        titulo = "Simulación de un proceso de caminata aleatoria "
        titulo += "sin deriva" if deriva == 0 else "con deriva"

        if num_realizaciones > 1:
            titulo += f" ({num_realizaciones} realizaciones)"

        plt.title(titulo)
        plt.xlabel("Tiempo (t)")
        plt.ylabel("Y(t)")
        plt.grid()
        plt.show()

    if resultados:
        return sims


##############################################################
#### Función que simula un proceso ARCH(1) de forma manual ###
##############################################################
def simulacion_arch1(mu=0,  
                     alpha0=0.2, 
                     alpha1=0.8, 
                     num_obs=100, 
                     semilla=None, 
                     graficar=True, 
                     resultados=False,
                     tam_fig=(8,3),
                     num_realizaciones=1
                     ):
    
    if semilla is not None:
        np.random.seed(semilla)
    
    # Lista para guardar las distintas realizaciones de y
    lista_y = []

    for _ in range(num_realizaciones):
        z = np.random.normal(0, 1, size=num_obs)
        y = np.zeros(num_obs)
        sigma2 = np.zeros(num_obs)

        # Varianza inicial (no cambia la lógica original)
        sigma2[0] = alpha0 / (1 - alpha1) if alpha1 < 1 else 1
        y[0] = mu + np.sqrt(sigma2[0]) * z[0]

        for t in range(1, num_obs):
            sigma2[t] = alpha0 + alpha1 * y[t-1]**2
            y[t] = mu + np.sqrt(sigma2[t]) * z[t]

        lista_y.append(y)

    # Convertimos a array de shape (num_realizaciones, num_obs)
    ys = np.array(lista_y)

    # Graficamos el proceso con heterocedasticidad condicional
    if graficar:
        plt.figure(figsize=tam_fig)
        for i in range(num_realizaciones):
            plt.plot(range(1, num_obs + 1), ys[i], linewidth=1)

        titulo = "Simulación de un proceso con heterocedasticidad condicional"
        if num_realizaciones > 1:
            titulo += f" ({num_realizaciones} realizaciones)"

        plt.title(titulo)
        plt.xlabel("Tiempo (t)")
        plt.ylabel("Y(t)")
        plt.xticks()
        plt.grid()
        plt.show()

    if resultados:
        # Para no romper código existente, si hay una sola realización devolvemos un 1D
        if num_realizaciones == 1:
            return ys[0]
        else:
            return ys
    

##############################################################
### Función que evalúa la existencia de heterocedasticidad ###
##############################################################
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_goldfeldquandt, het_white

def pruebas_heterocedasticidad(
    y,
    X=None,
    *,
    add_constant=True,
    gq_split=None,
    gq_drop=None,
    gq_alternative="two-sided",
    white_include_f=True,
    dropna=True
):
    """
    Devuelve un DataFrame con resultados principales de:
      - Goldfeld–Quandt (F y p-valor)
      - White (LM y opcionalmente F)

    Parámetros
    ----------
    y : array-like (n,)
        Serie a testear (p.ej., residuos o residuos estandarizados).
    X : array-like (n,) o (n,k), opcional
        Regresores/variable(s) para ordenar (GQ) y para White.
        Para series temporales, una elección típica es X = t (índice temporal).
        Si X is None, se usa el índice temporal.
    add_constant : bool
        Si True, agrega constante a X (recomendado para White).
    gq_split : int, opcional
        Punto de corte para GQ. Si None, statsmodels usa split=nobs//2.
        Para replicar “primer tercio vs último tercio”: gq_split = n//3.
    gq_drop : float o int, opcional
        Observaciones a descartar en el centro (ver statsmodels).
        Para replicar “primer tercio vs último tercio”: gq_drop = n//3 (o 1/3).
    gq_alternative : {"increasing","decreasing","two-sided"}
        Alternativa del test GQ.
    white_include_f : bool
        Si True, incluye también el estadístico F y p-valor del test de White.
    dropna : bool
        Si True, elimina NaN/inf en y (y filas correspondientes en X).

    Returns
    -------
    pandas.DataFrame
        Tabla con estadísticos, p-valores y metadatos clave.
    """

    y = np.asarray(y).squeeze()
    if y.ndim != 1:
        raise ValueError("y debe ser un vector 1D.")

    n = len(y)

    # Si X no se proporciona, usar índice temporal
    if X is None:
        X = np.arange(n)

    X = np.asarray(X)

    # Asegurar 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("X debe ser 1D o 2D.")

    if len(X) != n:
        raise ValueError("X y y deben tener el mismo número de observaciones.")

    # Limpiar NaN/inf si es necesario
    if dropna:
        mask = np.isfinite(y)
        # también exigir que X sea finita por fila
        mask &= np.all(np.isfinite(X), axis=1)
        y = y[mask]
        X = X[mask, :]
        n = len(y)

    # Agregar constante si corresponde (muy recomendable para White)
    X_use = sm.add_constant(X, has_constant="add") if add_constant else X

    # ---------- Goldfeld–Quandt ----------
    gq_F, gq_p, gq_order = het_goldfeldquandt(
        y, X_use,
        split=gq_split,
        drop=gq_drop,
        alternative=gq_alternative,
        store=False
    )

    # ---------- White ----------
    # White requiere exog con constante (idealmente) y usa términos cruzados internamente
    w_lm, w_lm_p, w_f, w_f_p = het_white(y, X_use)

    rows = []

    rows.append({
        "Prueba": "Goldfeld–Quandt",
        "Estadístico": gq_F,
        "Valor p": gq_p,
        "Detalles": f"alternative={gq_alternative}, split={gq_split}, drop={gq_drop}, add_constant={add_constant}"
    })

    rows.append({
        "Prueba": "White (LM)",
        "Estadístico": w_lm,
        "Valor p": w_lm_p,
        "Detalles": f"LM version, add_constant={add_constant}"
    })

    if white_include_f:
        rows.append({
            "Prueba": "White (F)",
            "Estadístico": w_f,
            "Valor p": w_f_p,
            "Detalles": f"F version, add_constant={add_constant}"
        })

    out = pd.DataFrame(rows)

    # Formato amigable
    out["Estadístico"] = out["Estadístico"].astype(float)
    out["Valor p"] = out["Valor p"].astype(float)

    return out


#################################################################
#### Función que obtiene raíces de polinomios de retardo ARMA ###
#################################################################
def _fmt_complex(z, dec=3):
    a = np.round(z.real, dec)
    b = np.round(z.imag, dec)
    if abs(b) < 10**(-dec):
        return f"{a:.{dec}f}"
    sign = "+" if b >= 0 else "-"
    return f"{a:.{dec}f}{sign}{abs(b):.{dec}f}i"

def raices_arma(pol_ar=[1], 
                    pol_ma=[1], 
                    unit_circle=True, 
                    raices_inv=False, 
                    decimals=3, 
                    annotate=True,
                    resultados=False):
    ar = np.asarray(pol_ar, float)
    ma = np.asarray(pol_ma, float)

    # np.roots espera coeficientes en orden de grado decreciente ⇒ invertimos
    r_ar = np.roots(ar[::-1]) if ar.size > 1 else np.array([])
    r_ma = np.roots(ma[::-1]) if ma.size > 1 else np.array([])

    if raices_inv:
        if r_ar.size: r_ar = 1.0 / r_ar
        if r_ma.size: r_ma = 1.0 / r_ma

    title=f"Raíces ARMA({len(pol_ar)-1},{len(pol_ma)-1}) y círculo unitario"

    fig, ax = plt.subplots(figsize=(6, 6))

    # Puntos
    if r_ar.size:
        ax.scatter(r_ar.real, r_ar.imag, marker="x", s=80, label="Raíces AR")
    if r_ma.size:
        ax.scatter(r_ma.real, r_ma.imag, marker="o", facecolors="red", s=80, label="Raíces MA")

    # Anotaciones
    if annotate:
        for i, z in enumerate(r_ar, 1):
            ax.annotate(f"AR{i}: {_fmt_complex(z, decimals)} \n Mod={abs(z):.{decimals}f}",
                        (z.real, z.imag), textcoords="offset points", xytext=(6, 6),
                        ha="left", va="bottom", color="green")
        for i, z in enumerate(r_ma, 1):
            ax.annotate(f"MA{i}: {_fmt_complex(z, decimals)} \n Mod={abs(z):.{decimals}f}",
                        (z.real, z.imag), textcoords="offset points", xytext=(6, -10),
                        ha="left", va="top", color="red")

    # Círculo unitario
    if unit_circle:
        t = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), linestyle="--", linewidth=1, label="Círculo unitario")

    # Ejes y estilo
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginaria")
    ax.set_title(title + (" (raíces inversas)" if raices_inv else ""))
    fig.subplots_adjust(bottom=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),ncol=3, frameon=False)
    plt.show()

    if resultados:
        return r_ar, np.abs(r_ar), r_ma, np.abs(r_ma)


##################################################################
#### Función para aplicar pruebas de raíz unitaria a una serie ###
##################################################################
# pip install statsmodels arch pandas
def pruebas_estacionariedad(y,
                                 adf_regression='c',   # 'c' (const), 'ct' (const+tend), 'nc' (none)
                                 adf_autolag='AIC',
                                 pp_trend='c',         # 'n','c','ct'
                                 kpss_regression='c',  # 'c' (nivel) o 'ct' (tendencia)
                                 kpss_nlags='auto',
                                 alpha=0.05):
    """
    Aplica ADF (statsmodels), Phillips–Perron (arch) y KPSS (statsmodels) a la serie y.
    Maneja automáticamente el InterpolationWarning de KPSS y decide con valores críticos.
    Retorna un DataFrame con estadísticos, p-valores, y decisión a nivel alpha.
    """
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]

    resultados = []

    # --- ADF (H0: raíz unitaria) ---
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
        y, regression=adf_regression, autolag=adf_autolag
    )
    resultados.append({
        'prueba': 'ADF (statsmodels)',
        'estadístico': adf_stat,
        'p_valor': adf_p,
        'lags/bw': adf_lags,
        'nobs': adf_nobs,
        'crit_1%': adf_crit.get('1%'),
        'crit_5%': adf_crit.get('5%'),
        'crit_10%': adf_crit.get('10%'),
        'H0': 'raíz unitaria',
        'decisión': 'Rechazar H0 (serie estacionaria)' if adf_p < alpha else 'No rechazar H0 (serie no estacionaria)',
        'nota': ''
    })

    # --- Phillips–Perron (H0: raíz unitaria) [arch] ---
    pp_res = PhillipsPerron(y, trend=pp_trend)  # kernel/bandwidth por defecto
    pp_stat = getattr(pp_res, 'stat', getattr(pp_res, 'statistic', None))
    pp_p = pp_res.pvalue
    pp_bw = getattr(pp_res, 'bandwidth', getattr(pp_res, 'lags', None))
    pp_nobs = getattr(pp_res, 'nobs', len(y))
    resultados.append({
        'prueba': 'PP (arch)',
        'estadístico': pp_stat,
        'p_valor': pp_p,
        'lags/bw': pp_bw,
        'nobs': pp_nobs,
        'crit_1%': np.nan, 'crit_5%': np.nan, 'crit_10%': np.nan,
        'H0': 'raíz unitaria',
        'decisión': 'Rechazar H0 (serie estacionaria)' if pp_p < alpha else 'No rechazar H0 (serie no estacionaria)',
        'nota': ''
    })

    # --- KPSS (H0: estacionaria) ---
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", InterpolationWarning)
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
            y, regression=kpss_regression, nlags=kpss_nlags
        )
    kpss_warn = any(issubclass(w.category, InterpolationWarning) for w in wlist) \
                or (kpss_p is None or (isinstance(kpss_p, float) and not np.isfinite(kpss_p)))

    # decide por p-valor o por críticos si hubo warning/truncamiento
    crit_key = {0.10: '10%', 0.05: '5%', 0.01: '1%'}.get(round(alpha, 2), '5%')
    crit_val = float(kpss_crit.get(crit_key))
    if kpss_warn:
        kpss_decision = 'Rechazar H0 (serie no estacionaria)' if kpss_stat > crit_val else 'No rechazar H0 (serie estacionaria)'
        kpss_nota = f"KPSS: p-valor fuera de tabla → decisión por críticos ({crit_key})."
    else:
        kpss_decision = 'Rechazar H0 (serie no estacionaria)' if kpss_p < alpha else 'No rechazar H0 (serie estacionaria)'
        kpss_nota = ''

    resultados.append({
        'prueba': 'KPSS (statsmodels)',
        'estadístico': kpss_stat,
        'p_valor': kpss_p,
        'lags/bw': kpss_lags,
        'nobs': len(y),
        'crit_1%': kpss_crit.get('1%'),
        'crit_5%': kpss_crit.get('5%'),
        'crit_10%': kpss_crit.get('10%'),
        'H0': 'estacionaria (c/ct según opción)',
        'decisión': kpss_decision,
        'nota': kpss_nota
    })

    cols = ['prueba','estadístico','p_valor','lags/bw','nobs',
            'crit_1%','crit_5%','crit_10%','H0','decisión','nota']
    return pd.DataFrame(resultados)[cols]


################################################################################
### Función para generar criterios de información para distintos p,q en ARMA ###
################################################################################
def tabla_arma_ic(y,
                          p_max=4,
                          q_max=4,
                          trend='n',                # 'n','c','t','ct'
                          exog=None,
                          enforce_stationarity=False,
                          enforce_invertibility=False,
                          cov_type='opg',           # 'opg'|'oim'|'robust'
                          concentrate_scale=False,
                          sort_by=None,             # None|'AIC'|'BIC'|'HQIC'
                          ascending=True,           # menor es mejor → True
                          dropna=True):
    """
    Devuelve AIC, BIC y HQIC para SARIMAX(p,0,q) sin estacionalidad,
    para 0<=p<=p_max y 0<=q<=q_max (excluye p=q=0).

    sort_by: None o uno de {'AIC','BIC','HQIC'} para ordenar la tabla.
    ascending: True (menor es mejor).
    """
    y = np.asarray(y, float)
    rows = []

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = SARIMAX(y,
                                  exog=exog,
                                  order=(p, 0, q),
                                  seasonal_order=(0, 0, 0, 0),
                                  trend=trend,
                                  enforce_stationarity=enforce_stationarity,
                                  enforce_invertibility=enforce_invertibility,
                                  concentrate_scale=concentrate_scale)
                    res = mod.fit(disp=False, cov_type=cov_type)
                rows.append({'p': p, 'q': q, 'AIC': res.aic, 'BIC': res.bic, 'HQIC': res.hqic})
            except Exception:
                rows.append({'p': p, 'q': q, 'AIC': np.nan, 'BIC': np.nan, 'HQIC': np.nan})

    df = pd.DataFrame(rows)
    if dropna:
        df = df.dropna(subset=['AIC','BIC','HQIC'])

    if sort_by is not None:
        key = sort_by.upper()
        if key not in {'AIC','BIC','HQIC'}:
            raise ValueError("sort_by debe ser None, 'AIC', 'BIC' o 'HQIC'.")
        df = df.sort_values(key, ascending=ascending, na_position='last').reset_index(drop=True)

    return df


##################################################################
### Función que genera gráfica de predicción (y complementos) ####
##################################################################
# ----------------- Utilidades -----------------
def _extract_intercept(results) -> Optional[float]:
    """Obtiene el intercepto ('const' o 'intercept') si existe."""
    try:
        names = list(results.param_names)
        params = results.params
        for key in ["const", "intercept"]:
            if key in names:
                # Acceso por posición para evitar FutureWarning
                return float(params.iloc[names.index(key)])
        return None
    except Exception:
        return None

def _unconditional_mean(results) -> Optional[float]:
    """
    Media teórica bajo φ(L) y_t = c + θ(L) ε_t: mu = c / φ(1),
    con φ(1) = 1 - sum(arparams expandidos).
    """
    try:
        c = _extract_intercept(results)
        if c is None:
            return None
        arparams = np.asarray(getattr(results, "arparams", np.array([])), dtype=float)
        phi1 = 1.0 - arparams.sum() if arparams.size else 1.0
        if np.isclose(phi1, 0.0):
            return None
        mu = c / phi1
        return float(mu) if np.isfinite(mu) else None
    except Exception:
        return None


def _extract_conf_int(forecast_res, steps: int, alpha: float):
    """
    Devuelve (lower, upper) de longitud == steps, robusto a distintas formas de conf_int().
    """
    ci = forecast_res.conf_int(alpha=alpha)

    # Caso DataFrame (común)
    if isinstance(ci, pd.DataFrame):
        r, c = ci.shape
        if r == steps and c >= 2:
            return (ci.iloc[:, 0].to_numpy(dtype=float),
                    ci.iloc[:, 1].to_numpy(dtype=float))
        if r == 2 and c == steps:
            return (ci.iloc[0, :].to_numpy(dtype=float),
                    ci.iloc[1, :].to_numpy(dtype=float))
        cols_lower = [col for col in ci.columns if "lower" in str(col).lower()]
        cols_upper = [col for col in ci.columns if "upper" in str(col).lower()]
        if cols_lower and cols_upper and r == steps:
            return (ci[cols_lower[0]].to_numpy(dtype=float),
                    ci[cols_upper[0]].to_numpy(dtype=float))

    # Caso ndarray
    arr = np.asarray(ci)
    if arr.ndim == 2:
        if arr.shape == (steps, 2):
            return arr[:, 0].astype(float), arr[:, 1].astype(float)
        if arr.shape == (2, steps):
            return arr[0, :].astype(float), arr[1, :].astype(float)

    # Último recurso: summary_frame
    try:
        sf = forecast_res.summary_frame(alpha=alpha)
        for lo_name in ["mean_ci_lower", "lower y", "lower"]:
            for hi_name in ["mean_ci_upper", "upper y", "upper"]:
                if lo_name in sf.columns and hi_name in sf.columns:
                    lo = sf[lo_name].to_numpy(dtype=float)
                    hi = sf[hi_name].to_numpy(dtype=float)
                    if lo.size == steps and hi.size == steps:
                        return lo, hi
        lo_cols = [c for c in sf.columns if "lower" in str(c).lower()]
        hi_cols = [c for c in sf.columns if "upper" in str(c).lower()]
        if lo_cols and hi_cols:
            lo = sf[lo_cols[0]].to_numpy(dtype=float)
            hi = sf[hi_cols[0]].to_numpy(dtype=float)
            if lo.size == steps and hi.size == steps:
                return lo, hi
    except Exception:
        pass

    # Si no se pudo, NaN para evitar crash (se omiten bandas).
    return np.full(steps, np.nan), np.full(steps, np.nan)

def _make_forecast_index(endog_index, steps: int):
    """
    Genera el índice del horizonte:
      - Si endog tiene DatetimeIndex/PeriodIndex, respeta su frecuencia.
      - Si no, devuelve un Range/np.arange contiguo.
    """
    # Series sin índice explícito (np.ndarray/list) -> rango entero
    if not isinstance(endog_index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex, pd.Index)):
        return np.arange(steps)

    # DatetimeIndex
    if isinstance(endog_index, pd.DatetimeIndex):
        freq = endog_index.freq or endog_index.inferred_freq
        if freq is None:
            # intentar delta del último tramo
            if len(endog_index) >= 2:
                delta = endog_index[-1] - endog_index[-2]
                if isinstance(delta, pd.Timedelta) and delta != pd.Timedelta(0):
                    start = endog_index[-1] + delta
                    return pd.date_range(start=start, periods=steps, freq=delta)
            # fallback diario
            return pd.date_range(start=endog_index[-1], periods=steps+1, freq="D")[1:]
        else:
            offset = pd.tseries.frequencies.to_offset(freq)
            start = endog_index[-1] + offset
            return pd.date_range(start=start, periods=steps, freq=freq)

    # PeriodIndex
    if isinstance(endog_index, pd.PeriodIndex):
        freq = endog_index.freq
        if freq is None:
            # intentar convertir a datetime e inferir
            dt = endog_index.to_timestamp()
            return _make_forecast_index(dt, steps)
        # siguiente período es +1
        start = endog_index[-1] + 1
        return pd.period_range(start=start, periods=steps, freq=freq)

    # RangeIndex o Index genérico numérico => enteros contiguos
    #n = len(endog_index)
    try:
        start = int(endog_index[-1]) + 1
        return pd.RangeIndex(start, start + steps)
    except Exception:
        # índice no numérico: usar 0..steps-1
        return np.arange(steps)

def _slice_index(endog_index, start_hist, n):
    """Subconjunto del índice de endog entre start_hist y n (maneja numpy/pandas)."""
    if isinstance(endog_index, (pd.Index, pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex)):
        return endog_index[start_hist:n]
    else:
        return np.arange(start_hist, n)

# ----------------- Función principal -----------------
def prediccion_grafico(
    results,
    endog: Union[pd.Series, pd.DataFrame, np.ndarray, list],
    steps: int,
    alpha: float = 0.05,
    last_n: Optional[int] = 100,
    connect_last_to_first_forecast: bool = True,
    ax: Optional[plt.Axes] = None,
    resultados=False,
    media_est=None,
    lim_inf_est=None,
    lim_sup_est=None
) -> pd.DataFrame:
    """
    Pronostica y grafica. Devuelve DataFrame con ['mean','lower','upper'].
    - Si `endog` tiene Datetime/PeriodIndex, el DF de salida ejes usan fechas.
    - Media teórica sólo si d=D=0 y sin exógenas.
    """
    # Asegurar 1D y capturar índice
    if isinstance(endog, pd.DataFrame):
        if endog.shape[1] == 0:
            raise ValueError("El DataFrame `endog` no tiene columnas.")
        endog_values = endog.iloc[:, 0].to_numpy(dtype=float).ravel()
        endog_index = endog.index
    elif isinstance(endog, pd.Series):
        endog_values = endog.to_numpy(dtype=float).ravel()
        endog_index = endog.index
    else:
        endog_values = np.asarray(endog, dtype=float).ravel()
        endog_index = pd.RangeIndex(len(endog_values))

    if steps <= 0:
        raise ValueError("`steps` debe ser un entero positivo.")
    if endog_values.size == 0:
        raise ValueError("`endog` está vacío.")

    n = endog_values.size

    # Pronóstico
    forecast_res = results.get_forecast(steps=steps)
    mean_forecast = np.asarray(forecast_res.predicted_mean, dtype=float).ravel()
    lower, upper = _extract_conf_int(forecast_res, steps=steps, alpha=alpha)

    # Verificación de dimensiones
    if not (mean_forecast.size == lower.size == upper.size == steps):
        raise ValueError(
            f"ConfInt mal dimensionado: mean={mean_forecast.size}, lower={lower.size}, upper={upper.size}, steps={steps}"
        )

    # Índices para gráfico y retorno
    start_hist = 0 if last_n is None else int(max(0, n - int(last_n)))
    x_hist = _slice_index(endog_index, start_hist, n)
    y_hist = endog_values[start_hist:]

    x_fore = _make_forecast_index(endog_index, steps)

    # Decidir si mostrar media teórica
    mu = None
    try:
        p, d, q = getattr(results.model, "order", (None, None, None))
        P, D, Q, s = getattr(results.model, "seasonal_order", (0, 0, 0, 0))
        k_exog = getattr(results.model, "k_exog", 0)
        if (d == 0) and (D == 0) and (k_exog == 0):
            if _unconditional_mean(results) is not None:
                mu = _unconditional_mean(results)
            else:
                mu=media_est
    except Exception:
        pass

    # Graficar
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x_hist, y_hist, label="Observado")

    # Curva de pronóstico y color
    (forecast_line,) = ax.plot(x_fore, mean_forecast, label="Pronóstico")
    forecast_color = forecast_line.get_color()

    # Conectar último observado con primer pronóstico
    if connect_last_to_first_forecast and steps > 0:
        ax.plot([x_hist[-1], x_fore[0]], [endog_values[-1], mean_forecast[0]], color=forecast_color)

    # Bandas de IC
    finite_mask = np.isfinite(lower) & np.isfinite(upper)
    if finite_mask.all():
        ax.fill_between(x_fore, lower, upper, alpha=0.2, label=f"IC {int(round((1-alpha)*100))}%", color=forecast_color)
    elif finite_mask.any():
        ax.fill_between(pd.Index(x_fore)[finite_mask], lower[finite_mask], upper[finite_mask],
                        alpha=0.2, label=f"IC {int(round((1-alpha)*100))}%", color=forecast_color)

    # Media teórica (si aplica)
    if mu is not None:
        ax.axhline(mu, linestyle="--", linewidth=1.5, label=f"Media teórica ({mu:.3g})")
        
    if lim_inf_est is not None:
        ax.axhline(lim_inf_est, 
                   linestyle="--", 
                   color="red",
                   linewidth=1.5, 
                   label=f"Límite inferior teórico {int(round((1-alpha)*100))}% ({lim_inf_est:.3g})")

    if lim_sup_est is not None:
        ax.axhline(lim_sup_est, 
                   linestyle="--", 
                   color="red",
                   linewidth=1.5, 
                   label=f"Límite superior teórico {int(round((1-alpha)*100))}% ({lim_sup_est:.3g})")
        
    ax.set_xlabel("Tiempo" if isinstance(endog_index, (pd.DatetimeIndex, pd.PeriodIndex)) else "Índice")
    ax.set_ylabel("Nivel de la serie")
    ax.set_title("Pronóstico SARIMAX con IC y media teórica")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.tight_layout()

    # DataFrame resultado con índice del horizonte (fechas si aplica)
    df = pd.DataFrame({"mean": mean_forecast, "lower": lower, "upper": upper}, index=x_fore)
    # Nombre del índice
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df.index.name = "horizon_time"
    else:
        df.index.name = "horizon_index"
        
    if resultados:
        return df


##########################################
### Función que resume pruebas JB y SW ###
##########################################
def pruebas_normalidad(v_hat, alpha=0.05, redondeo=4):
    v = np.asarray(v_hat).ravel()
    v = v[~np.isnan(v)]
    n = v.size

    # Jarque–Bera (statsmodels devuelve jb, p, skew, kurt)
    jb, p_jb, skew, kurt = jarque_bera(v)
    # Shapiro–Wilk
    w, p_sw = shapiro(v)

    rows = [
        {"Prueba": "Jarque–Bera", "Estadístico": jb, "p-valor": p_jb,
         "Decisión": "No se rechaza normalidad" if p_jb > alpha else "Se rechaza normalidad",
         "Sesgo": skew, "Curtosis": kurt,
         "Comentario": ""},
        {"Prueba": "Shapiro–Wilk", "Estadístico": w, "p-valor": p_sw,
         "Decisión": "No se rechaza normalidad" if p_sw > alpha else "Se rechaza normalidad",
         "Sesgo": np.nan, "Curtosis": np.nan,
         "Comentario": "Recomendado hasta ~5000 obs" if n > 5000 else ""},
    ]
    df = pd.DataFrame(rows).set_index("Prueba")
    df.attrs["n"] = n
    df.attrs["alpha"] = alpha

    def _color_p(val):
        if pd.isna(val):
            return ""
        # verde si no se rechaza, rojo si se rechaza
        return "background-color: #e8f5e9" if val > alpha else "background-color: #ffebee"

    def _color_decision(val):
        if val == "Se rechaza":
            return "color: #b71c1c; font-weight: 700"
        if val == "No se rechaza":
            return "color: #1b5e20; font-weight: 700"
        return ""

    return df


#####################################################
### Función de diagnóstico de residuos ARMA-GARCH ###
#####################################################
def diagnostico_residuos_est(z_hat,
                             graficar_z_hat=False,
                             ljung_box_z_hat=False,
                             num_rezagos=10,
                             correlograma_z_hat=False,
                             histograma_z_hat=False,
                             qq_z_hat=False,
                             jb_sw_z_hat=False,
                             alphan=0.05,
                             arch_z_hat=False,
                             ljung_box_z_hat2=False,
                             correlograma_z_hat2=False,
                             tam_fig=(8,3)
                             ):
    
    #Graficamos los residuos estandarizados
    if graficar_z_hat:
        print("Gráfica de z_hat")
        plt.figure(figsize=tam_fig) 
        plt.plot(z_hat, color='blue', linewidth=1) 
        plt.title("Gráfica de residuos estandarizados", fontsize=14)
        plt.ylabel("Residuos")
        plt.axhline(0, color="green", linestyle="-", label=f"Media (teórica) = 0")
        plt.axhline(-1.96, color="red", linestyle="--", label=f"IC 95% inferior (teórico) = -1.96")
        plt.axhline(1.96, color="red", linestyle="--", label=f"IC 95% superior (teórico) = +1.96")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))  
        plt.xlabel("")
        plt.grid()
        plt.show()
        
    #Diagnosticamos que los residuos estandarizados no presenten autocorrelación (prueba LB)
    if ljung_box_z_hat:
        print("\nLjung–Box en z_hat")
        #print(acorr_ljungbox(z_hat, lags=10, return_df=True))
        return acorr_ljungbox(z_hat, lags=num_rezagos, return_df=True)
    
    #Diagnóstico de ausencia de autocorrelación en residuos estandarizados con ACF y PACF
    if correlograma_z_hat:
        print("\nCorrelogramas de z_hat")
        fig, axes = plt.subplots(2,1, figsize=tam_fig) 
        plot_acf(z_hat, alpha=0.05, ax=axes[0]) 
        axes[0].set_title("ACF muestral de residuos estandarizados")
        plot_pacf(z_hat, alpha=0.05, ax=axes[1],method="ywm") 
        axes[1].set_title("PACF muestral de residuos estandarizados")
        [ax.set_ylim(-0.25, 1.1) for ax in axes]
        [ax.grid() for ax in axes]
        plt.tight_layout()
        plt.show()

    #Diagnóstico de normalidad de residuos estandarizados (histograma vs N(0,1))
    if histograma_z_hat:
        print("\nHistograma de z_hat")
        plt.figure()
        plt.hist(z_hat, bins=40, density=True, alpha=0.7, edgecolor="black")
        x = np.linspace(z_hat.min(), z_hat.max(), 300)
        pdf = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
        plt.plot(x, pdf)
        plt.title("Residuos estandarizados: histograma vs Normal(0,1)")
        plt.tight_layout()
    
    #Diagnóstico de normalidad de residuos estandarizados (gráfico QQ)
    if qq_z_hat:
        print("\nQQ-plot de z_hat")
        qqplot(z_hat, line="45", fit=True)   # línea 45° y ajuste
        plt.title("QQ-plot: residuos estandarizados vs Normal(0,1)")
        plt.xlabel("Cuantiles teóricos N(0,1)")
        plt.ylabel("Cuantiles muestrales")
        plt.tight_layout()
        plt.show()
    
    #Diagnostico de normalidad pruebas JB y SW
    if jb_sw_z_hat:
        print("\nPruebas de normalidad sobre z_hat")
        v_hat=z_hat
        v = np.asarray(v_hat).ravel()
        v = v[~np.isnan(v)]
        n = v.size

        # Jarque–Bera (statsmodels devuelve jb, p, skew, kurt)
        jb, p_jb, skew, kurt = jarque_bera(v)
        # Shapiro–Wilk
        w, p_sw = shapiro(v)

        rows = [
            {"Prueba": "Jarque–Bera", "Estadístico": jb, "p-valor": p_jb,
             "Decisión": "No se rechaza normalidad" if p_jb > alphan else "Se rechaza normalidad",
             "Sesgo": skew, "Curtosis": kurt,
             "Comentario": ""},
            {"Prueba": "Shapiro–Wilk", "Estadístico": w, "p-valor": p_sw,
             "Decisión": "No se rechaza normalidad" if p_sw > alphan else "Se rechaza normalidad",
             "Sesgo": np.nan, "Curtosis": np.nan,
             "Comentario": "Recomendado hasta ~5000 obs" if n > 5000 else ""},
        ]
        df = pd.DataFrame(rows).set_index("Prueba")
        df.attrs["n"] = n
        df.attrs["alpha"] = alphan

        def _color_p(val):
            if pd.isna(val):
                return ""
            # verde si no se rechaza, rojo si se rechaza
            return "background-color: #e8f5e9" if val > alphan else "background-color: #ffebee"

        def _color_decision(val):
            if val == "Se rechaza":
                return "color: #b71c1c; font-weight: 700"
            if val == "No se rechaza":
                return "color: #1b5e20; font-weight: 700"
            return ""

        return df

    ##Diagnóstico de ausencia de ARCH en residuos estandarizados (prueba ARCH)
    if arch_z_hat:
        #lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(z_hat, nlags=10)
        #print("\nLM de Engle sobre residuos estandarizados")
        #print({"LM": lm_stat, "LM_pvalue": lm_pvalue, "F": f_stat, "F_pvalue": f_pvalue})
        redondeo=4
        z = np.asarray(z_hat).ravel()
        z = z[~np.isnan(z)]
        n = z.size
    
        lm_stat, lm_p, f_stat, f_p = het_arch(z, nlags=num_rezagos)
    
        df = pd.DataFrame({
            "Estadístico": [lm_stat, f_stat],
            "p-valor":     [lm_p,     f_p],
            "Decisión": [
                "No se rechaza H0 (no existe ARCH)" if lm_p > alphan else "Se rechaza H0 (existe ARCH)",
                "No se rechaza H0 (no existe ARCH)" if f_p > alphan else "Se rechaza H0 (existe ARCH)"
            ],
            "nlags": [num_rezagos, num_rezagos],
            "n":     [n, n]
        }, index=["LM (Engle)", "F (Engle)"])
    
        # Redondeo simple de columnas numéricas
        df[["Estadístico", "p-valor"]] = df[["Estadístico", "p-valor"]].astype(float).round(redondeo)
        return df



    #Diagnóstico de ausencia de ARCH en residuos estandarizados (prueba LB en z_hat^2)
    if ljung_box_z_hat2:
        print("Ljung–Box en (z_hat)^2")
        return acorr_ljungbox(z_hat**2, lags=num_rezagos, return_df=True)


    #Diagnóstico de ausencia de ARCH (ACF y PACF de z_hat^2)
    if correlograma_z_hat2:
        fig, axes = plt.subplots(2,1, figsize=tam_fig) 
        plot_acf(z_hat**2, alpha=0.05, ax=axes[0]) 
        axes[0].set_title("ACF muestral de residuos estandarizados al cuadrado")
        plot_pacf(z_hat**2, alpha=0.05, ax=axes[1],method="ywm") 
        axes[1].set_title("PACF muestral de residuos estandarizados al cuadrado")
        [ax.set_ylim(-0.25, 1.1) for ax in axes]
        [ax.grid() for ax in axes]
        plt.tight_layout()
        plt.show()


##########################################
###### Función para Prueba ARCH ##########
##########################################
def prueba_arch(z_hat, nlags=10, alpha=0.05, redondeo=4):
    z = np.asarray(z_hat).ravel()
    z = z[~np.isnan(z)]
    n = z.size

    lm_stat, lm_p, f_stat, f_p = het_arch(z, nlags=nlags)

    df = pd.DataFrame({
        "Estadístico": [lm_stat, f_stat],
        "p-valor":     [lm_p,     f_p],
        "Decisión": [
            "No se rechaza H0 (no existe ARCH)" if lm_p > alpha else "Se rechaza H0 (existe ARCH)",
            "No se rechaza H0 (no existe ARCH)" if f_p > alpha else "Se rechaza H0 (existe ARCH)"
        ],
        "nlags": [nlags, nlags],
        "n":     [n, n]
    }, index=["LM (Engle)", "F (Engle)"])

    # Redondeo simple de columnas numéricas
    df[["Estadístico", "p-valor"]] = df[["Estadístico", "p-valor"]].astype(float).round(redondeo)
    return df














