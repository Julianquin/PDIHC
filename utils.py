import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.forecasting.theta import ThetaModel
import matplotlib.pyplot as plt



# =====================
#  Preparación de Datos
# =====================

def generar_datos(n_series=3, n_points=120, seed=42, start_date="2023-01-01"):
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=n_points, freq="D")

    # Generación de datos
    data = []
    for i in range(1, n_series + 1):
        key = f"SERIE_{i}"
        
        # Componentes básicos
        trend = np.linspace(100, 120, n_points)  # Tendencia lineal
        weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Estacionalidad semanal
        monthly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / 30)  # Estacionalidad mensual
        noise = np.random.normal(0, 3, n_points)  # Ruido aleatorio
        
        # Valores reales y predicciones
        y_real = trend + weekly_seasonality + monthly_seasonality + noise
        y_pred = y_real + np.random.normal(0, 7, n_points)  # Agregar ruido adicional a las predicciones
        
        # Etiquetas de futuro
        future = [0] * int(n_points * 0.7) + [1] * int(n_points * 0.3)
        
        # Construir el DataFrame
        data.extend({
            "KEY": key,
            "FECHA": date,
            "Y": real if not fut else np.nan,
            "YHATFIN": pred,
            "FUTURE": fut,
            "YHAT_L": np.nan,
            "YHAT_U": np.nan,
        } for date, real, pred, fut in zip(dates, y_real, y_pred, future))

    df = pd.DataFrame(data)
    return df

def assign_data_sets(df, date_col, future_col, calib_ratio=0.6):
    """
    Asigna etiquetas de conjunto (TRAIN, CALIBRATION, TEST) a un DataFrame.

    - TEST: Datos con FUTURE = 1.
    - CALIBRATION: Porcentaje especificado del subconjunto FUTURE = 0.
    - TRAIN: Resto del subconjunto FUTURE = 0.

    Parámetros:
    - df: DataFrame de entrada.
    - date_col: Nombre de la columna con fechas para ordenar.
    - future_col: Nombre de la columna que indica si es futuro (1) o pasado (0).
    - calib_ratio: Porcentaje de datos históricos asignados al conjunto de calibración.

    Devuelve:
    - DataFrame con una nueva columna `SET` indicando el conjunto asignado.
    """
    # Ordenar los datos por fecha
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # Crear la columna SET inicializando como TEST
    df["SET"] = "TEST"

    # Identificar índices de FUTURE = 0 (datos históricos)
    historical_indices = df[df[future_col] == 0].index

    # Calcular tamaños para TRAIN y CALIBRATION
    n_historical = len(historical_indices)
    n_calib = int(n_historical * calib_ratio)

    # Asignar TRAIN y CALIBRATION en datos históricos
    df.loc[historical_indices[:n_historical - n_calib], "SET"] = "TRAIN"
    df.loc[historical_indices[n_historical - n_calib:], "SET"] = "CALIBRATION"

    return df

def split_data_by_set(group, set_col):
    """
    Divide un grupo en conjuntos de entrenamiento, calibración y prueba.
    """
    train = group[group[set_col] == "TRAIN"]
    calib = group[group[set_col] == "CALIBRATION"]
    test = group[group[set_col] == "TEST"]
    return train, calib, test

# ==============================
#  Cálculo de Intervalos - PDI
# ==============================

def quantile_integrator_log_scorecaster(
    scores, 
    alpha, 
    lr, 
    T_burnin, 
    Csat, 
    KI, 
    ahead, 
    seasonal_period, 
    time_index, 
    integrate=True, 
    proportional_lr=True, 
    scorecast=True
):
    """
    Calcula intervalos dinámicos de predicción utilizando PDI.

    Parámetros:
    - scores (np.ndarray): Errores absolutos entre valores reales y predicciones.
    - alpha (float): Nivel de significancia (1 - alpha es la cobertura deseada).
    - lr (float): Tasa de aprendizaje para ajustar el término proporcional.
    - T_burnin (int): Período de calentamiento antes de aplicar integral y derivativo.
    - Csat (float): Constante de saturación para el término integral.
    - KI (float): Escala del componente integral.
    - ahead (int): Horizonte de predicción (pasos hacia el futuro).
    - seasonal_period (int): Período de estacionalidad para el modelo derivativo.
    - time_index (pd.DatetimeIndex): Índice temporal de los datos.
    - integrate (bool): Si se debe aplicar el componente integral.
    - proportional_lr (bool): Si se debe ajustar dinámicamente la tasa de aprendizaje.
    - scorecast (bool): Si se debe incluir el componente derivativo.

    Retorna:
    - qs (np.ndarray): Intervalos ajustados dinámicamente.
    """

    T_test = scores.shape[0]  # Número total de pasos temporales
    # Inicialización de variables
    qs = np.zeros((T_test,))  # Cuantiles ajustados
    qts = np.zeros((T_test,))  # Componente proporcional ajustado
    integrators = np.zeros((T_test,))  # Componente integral ajustado
    scorecasts = np.zeros((T_test,))  # Componente derivativo ajustado
    covereds = np.zeros((T_test,))  # Indicador de cobertura

    # Iterar a través de los pasos temporales
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1  # Tiempo de predicción actual
        if t_pred < 0:
            continue  # Saltar si no hay datos suficientes para predecir

        # ================================
        # Tasa de aprendizaje dinámica (P)
        # ================================
        t_lr_min = max(t - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t].max() - scores[t_lr_min:t].min()) if proportional_lr and t > 0 else lr

        # ======================================
        # Componente Proporcional y Gradiente (P)
        # ======================================
        covereds[t] = qs[t] >= scores[t]  # Determinar si el cuantil cubre el error
        grad = alpha if covereds[t_pred] else -(1 - alpha)  # Gradiente basado en cobertura

        # ============================
        # Componente Integral (I)
        # ============================
        integrator_arg = (1 - covereds)[:t_pred].sum() - (t_pred) * alpha
        integrator = (
            np.tan(integrator_arg * np.log(t_pred + 1) / (Csat * (t_pred + 1))) if integrate else 0
        )  # Corrección acumulativa

        # ================================
        # Componente Derivativo (D)
        # ================================
        if scorecast and t_pred > T_burnin and t + ahead < T_test:
            score_series = pd.Series(scores[:t_pred], index=time_index[:t_pred])
            model = ThetaModel(score_series, period=seasonal_period)  # Modelo derivativo
            fitted_model = model.fit()
            scorecasts[t + ahead] = fitted_model.forecast(ahead).iloc[-1]  # Predicción futura del error

        # ================================
        # Actualización del Cuantil
        # ================================
        if t < T_test - 1:
            # Actualización del término proporcional
            qts[t + 1] = qts[t] - lr_t * grad
            # Actualización del término integral
            integrators[t + 1] = integrator if integrate else 0
            # Cálculo del cuantil ajustado
            qs[t + 1] = qts[t + 1] + integrators[t + 1]
            if scorecast:
                qs[t + 1] += scorecasts[t + 1]  # Agregar predicción futura

    return qs


def quantile_integrator_log_scorecaster_with_diagnostics(
    scores, 
    alpha, 
    lr, 
    T_burnin, 
    Csat, 
    KI, 
    ahead, 
    seasonal_period, 
    time_index, 
    integrate=True, 
    proportional_lr=True, 
    scorecast=True,
    coverage_threshold=0.9  # Umbral para alertar cobertura baja
):
    """
    Calcula intervalos dinámicos de predicción con diagnóstico detallado.

    Retorna:
    - qs (np.ndarray): Intervalos ajustados dinámicamente.
    - logs (dict): Logs detallados de cada componente.
    """
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))

    # Logs para análisis posterior
    logs = {"proportional": [], "integral": [], "derivative": [], "coverage": []}

    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue

        # ================================
        # Tasa de aprendizaje dinámica (P)
        # ================================
        t_lr_min = max(t - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t].max() - scores[t_lr_min:t].min()) if proportional_lr and t > 0 else lr
        # lr_t = lr
        # ======================================
        # Componente Proporcional y Gradiente (P)
        # ======================================
        covereds[t] = qs[t] >= scores[t]  # Determinar si el cuantil cubre el error
        grad = alpha if covereds[t_pred] else -(1 - alpha)
        proportional_update = -lr_t * grad
        logs["proportional"].append(proportional_update)

        # ============================
        # Componente Integral (I)
        # ============================
        integrator_arg = (1 - covereds)[:t_pred].sum() - (t_pred) * alpha
        integrator = (
            np.tan(integrator_arg * np.log(t_pred + 1) / (Csat * (t_pred + 1))) if integrate else 0
        )
        integrators[t] = integrator
        logs["integral"].append(integrator)

        # ================================
        # Componente Derivativo (D)
        # ================================
        if scorecast and t_pred > T_burnin and t + ahead < T_test:
            score_series = pd.Series(scores[:t_pred], index=time_index[:t_pred])
            model = ThetaModel(score_series, period=seasonal_period)
            fitted_model = model.fit()
            scorecasts[t + ahead] = fitted_model.forecast(ahead).iloc[-1]
        logs["derivative"].append(scorecasts[t])

        # Registrar cobertura
        logs["coverage"].append(covereds[t])

        # Alertar si la cobertura es baja
        if covereds[t] < coverage_threshold:
            print(f"⚠️ Cobertura baja en el paso {t}: {covereds[t]}")

        # ================================
        # Actualización del Cuantil
        # ================================
        if t < T_test - 1:
            qts[t + 1] = qts[t] + proportional_update
            integrators[t + 1] = integrator if integrate else 0
            qs[t + 1] = qts[t + 1] + integrators[t + 1]
            if scorecast:
                qs[t + 1] += scorecasts[t + 1]

    return qs, logs


def apply_pdi_with_calibration(
    df, 
    key_col, 
    date_col, 
    value_col, 
    pred_col, 
    lower_col, 
    upper_col, 
    alpha, 
    lr, 
    T_burnin, 
    Csat, 
    KI, 
    ahead, 
    seasonal_period, 
    set_col
):
    """
    Aplica el método PDI con calibración a un DataFrame que contiene múltiples series.

    Esta función genera intervalos dinámicos de predicción (PDI) para cada serie identificada por `key_col`.
    Se utilizan conjuntos de datos separados en entrenamiento (TRAIN), calibración (CALIBRATION), 
    y prueba (TEST) según lo indicado en la columna `set_col`.

    Parámetros:
    - df (pd.DataFrame): DataFrame que contiene los datos.
    - key_col (str): Columna que identifica las diferentes series.
    - date_col (str): Columna que contiene las fechas.
    - value_col (str): Columna con los valores reales.
    - pred_col (str): Columna con las predicciones.
    - lower_col (str): Nombre de la columna donde se guardará el límite inferior del intervalo.
    - upper_col (str): Nombre de la columna donde se guardará el límite superior del intervalo.
    - alpha (float): Nivel de significancia (1 - alpha es la cobertura deseada).
    - lr (float): Tasa de aprendizaje para ajustar el término proporcional.
    - T_burnin (int): Período de calentamiento antes de aplicar integral y derivativo.
    - Csat (float): Constante de saturación para el término integral.
    - KI (float): Escala del componente integral.
    - ahead (int): Horizonte de predicción (pasos hacia el futuro).
    - seasonal_period (int): Período de estacionalidad para el modelo derivativo.
    - set_col (str): Columna que indica el conjunto al que pertenece cada dato (TRAIN, CALIBRATION, TEST).

    Retorna:
    - pd.DataFrame: DataFrame con los intervalos de predicción (`lower_col`, `upper_col`) generados.

    """
    # Lista para almacenar los resultados procesados por serie
    results = []

    # Iterar por cada serie identificada por key_col
    for key, group in df.groupby(key_col):
        # Ordenar los datos por la columna de fechas
        group = group.sort_values(by=date_col)

        # Dividir los datos en conjuntos de entrenamiento, calibración y prueba
        train, calib, test = split_data_by_set(group, set_col=set_col)

        # ==============================
        # Datos de Entrenamiento
        # ==============================
        y_train = train[value_col].dropna().values  # Valores reales
        y_pred_train = train[pred_col].dropna().values  # Predicciones
        scores_train = np.abs(y_train - y_pred_train)  # Cálculo de errores absolutos
        time_index_train = pd.to_datetime(train[date_col])  # Índice temporal para entrenamiento

        # Calcular intervalos para datos de entrenamiento usando PDI
        qs_calib = quantile_integrator_log_scorecaster(
            scores=scores_train,
            alpha=alpha,
            lr=lr,
            T_burnin=T_burnin,
            Csat=Csat,
            KI=KI,
            ahead=ahead,
            seasonal_period=seasonal_period,
            time_index=time_index_train
        )

        # Generar intervalos para el conjunto de entrenamiento
        train[lower_col] = train[pred_col] - qs_calib
        train[upper_col] = train[pred_col] + qs_calib

        # ==============================
        # Datos de Calibración y Prueba
        # ==============================
        last_calib_quantile = qs_calib[-1]  # Último cuantil ajustado del entrenamiento
        # Calibración
        calib[lower_col] = calib[pred_col] - last_calib_quantile
        calib[upper_col] = calib[pred_col] + last_calib_quantile
        # Prueba
        test[lower_col] = test[pred_col] - last_calib_quantile
        test[upper_col] = test[pred_col] + last_calib_quantile

        # Agregar los conjuntos procesados a la lista de resultados
        results.append(pd.concat([train, calib, test]))

    # Combinar los resultados procesados para todas las series
    return pd.concat(results)


def apply_pdi_with_calibration_with_diagnostics(
    df, 
    key_col, 
    date_col, 
    value_col, 
    pred_col, 
    lower_col, 
    upper_col, 
    alpha, 
    lr, 
    T_burnin, 
    Csat, 
    KI, 
    ahead, 
    seasonal_period, 
    set_col
):
    """
    Aplica el método PDI con calibración a un DataFrame que contiene múltiples series.

    Esta función genera intervalos dinámicos de predicción (PDI) para cada serie identificada por `key_col`.
    Se utilizan conjuntos de datos separados en entrenamiento (TRAIN), calibración (CALIBRATION), 
    y prueba (TEST) según lo indicado en la columna `set_col`.

    Parámetros:
    - df (pd.DataFrame): DataFrame que contiene los datos.
    - key_col (str): Columna que identifica las diferentes series.
    - date_col (str): Columna que contiene las fechas.
    - value_col (str): Columna con los valores reales.
    - pred_col (str): Columna con las predicciones.
    - lower_col (str): Nombre de la columna donde se guardará el límite inferior del intervalo.
    - upper_col (str): Nombre de la columna donde se guardará el límite superior del intervalo.
    - alpha (float): Nivel de significancia (1 - alpha es la cobertura deseada).
    - lr (float): Tasa de aprendizaje para ajustar el término proporcional.
    - T_burnin (int): Período de calentamiento antes de aplicar integral y derivativo.
    - Csat (float): Constante de saturación para el término integral.
    - KI (float): Escala del componente integral.
    - ahead (int): Horizonte de predicción (pasos hacia el futuro).
    - seasonal_period (int): Período de estacionalidad para el modelo derivativo.
    - set_col (str): Columna que indica el conjunto al que pertenece cada dato (TRAIN, CALIBRATION, TEST).

    Retorna:
    - pd.DataFrame: DataFrame con los intervalos de predicción (`lower_col`, `upper_col`) generados.

    """
    # Lista para almacenar los resultados procesados por serie
    results = []

    # Iterar por cada serie identificada por key_col
    for key, group in df.groupby(key_col):
        # Ordenar los datos por la columna de fechas
        group = group.sort_values(by=date_col)

        # Dividir los datos en conjuntos de entrenamiento, calibración y prueba
        train, calib, test = split_data_by_set(group, set_col=set_col)

        # ==============================
        # Datos de Entrenamiento
        # ==============================
        y_train = train[value_col].dropna().values  # Valores reales
        y_pred_train = train[pred_col].dropna().values  # Predicciones
        scores_train = np.abs(y_train - y_pred_train)  # Cálculo de errores absolutos
        time_index_train = pd.to_datetime(train[date_col])  # Índice temporal para entrenamiento

        # Calcular intervalos para datos de entrenamiento usando PDI


        qs_calib, logs = quantile_integrator_log_scorecaster_with_diagnostics(
            scores=scores_train,
            alpha=alpha,
            lr=lr,
            T_burnin=T_burnin,
            Csat=Csat,
            KI=KI,
            ahead=ahead,
            seasonal_period=seasonal_period,
            time_index=time_index_train
        )

        # Generar intervalos para el conjunto de entrenamiento
        train[lower_col] = train[pred_col] - qs_calib
        train[upper_col] = train[pred_col] + qs_calib

        # ==============================
        # Datos de Calibración y Prueba
        # ==============================
        last_calib_quantile = qs_calib[-1]  # Último cuantil ajustado del entrenamiento
        # Calibración
        calib[lower_col] = calib[pred_col] - last_calib_quantile
        calib[upper_col] = calib[pred_col] + last_calib_quantile
        # Prueba
        test[lower_col] = test[pred_col] - last_calib_quantile
        test[upper_col] = test[pred_col] + last_calib_quantile

        # Agregar los conjuntos procesados a la lista de resultados
        results.append(pd.concat([train, calib, test]))

    # Combinar los resultados procesados para todas las series
    return pd.concat(results), logs

# =========================
#  Evaluación de Resultados
# =========================

# =============================
#  Funciones de Evaluación
# =============================
def calculate_marginal_coverage(df, value_col, lower_col, upper_col):
    """
    Calcula la cobertura marginal de los intervalos.
    """
    coverage = (df[value_col] >= df[lower_col]) & (df[value_col] <= df[upper_col])
    return coverage.mean()

def calculate_coverage_deviation(df, value_col, lower_col, upper_col, alpha):
    """
    Calcula el sesgo de cobertura (Coverage Deviation).
    """
    empirical_coverage = calculate_marginal_coverage(df, value_col, lower_col, upper_col)
    return empirical_coverage - (1 - alpha)

def calculate_average_region_size(df, lower_col, upper_col):
    """
    Calcula el tamaño promedio de los intervalos.
    """
    return (df[upper_col] - df[lower_col]).mean()

def calculate_winkler_score(df, value_col, lower_col, upper_col, alpha):
    """
    Calcula el Winkler Score para evaluar los intervalos.
    """
    inside_interval = (df[value_col] >= df[lower_col]) & (df[value_col] <= df[upper_col])
    lower_penalty = ((df[lower_col] - df[value_col]) * ~inside_interval * (df[value_col] < df[lower_col])).sum()
    upper_penalty = ((df[value_col] - df[upper_col]) * ~inside_interval * (df[value_col] > df[upper_col])).sum()
    interval_width = (df[upper_col] - df[lower_col]).sum()
    return (interval_width + (2 / alpha) * (lower_penalty + upper_penalty)) / len(df)

def calculate_pinball_loss(df, value_col, quantile_col, quantile_level):
    """
    Calcula la Pinball Loss para un cuantil dado.
    """
    error = df[value_col] - df[quantile_col]
    loss = np.where(
        error < 0, 
        (1 - quantile_level) * -error, 
        quantile_level * error
    )
    return loss.mean()

def calculate_metrics(df, value_col, lower_col, upper_col, alpha, condition_col=None):
    """
    Calcula todas las métricas principales (Marginal Coverage, Coverage Deviation, Average Region Size, Winkler Score)
    y opcionalmente por grupos definidos por `condition_col`.
    """
    metrics = []

    if condition_col:
        grouped = df.groupby(condition_col)
        for group_name, group in grouped:
            group_metrics = {
                "Group": group_name,
                "Marginal Coverage": calculate_marginal_coverage(group, value_col, lower_col, upper_col),
                "Coverage Deviation": calculate_coverage_deviation(group, value_col, lower_col, upper_col, alpha),
                "Average Region Size": calculate_average_region_size(group, lower_col, upper_col),
                "Winkler Score": calculate_winkler_score(group, value_col, lower_col, upper_col, alpha)
            }
            metrics.append(group_metrics)
    else:
        metrics.append({
            "Group": "Global",
            "Marginal Coverage": calculate_marginal_coverage(df, value_col, lower_col, upper_col),
            "Coverage Deviation": calculate_coverage_deviation(df, value_col, lower_col, upper_col, alpha),
            "Average Region Size": calculate_average_region_size(df, lower_col, upper_col),
            "Winkler Score": calculate_winkler_score(df, value_col, lower_col, upper_col, alpha)
        })

    return pd.DataFrame(metrics)

# ====================
#  Visualización
# ====================

def plot_series_results_with_sets(
    df, key, key_col="KEY", date_col="FECHA", value_col="Y",
    pred_col="YHATFIN", lower_col="YHAT_L", upper_col="YHAT_U", set_col="SET"
):
    """
    Visualiza resultados por serie con conjuntos de datos marcados.
    """
    series_df = df[df[key_col] == key].sort_values(by=date_col)
    train_df = series_df[series_df[set_col] == "TRAIN"]
    calib_df = series_df[series_df[set_col] == "CALIBRATION"]
    test_df = series_df[series_df[set_col] == "TEST"]

    plt.figure(figsize=(12, 6))
    plt.plot(series_df[date_col], series_df[value_col], "-", label="Valores Reales", color="blue")
    plt.plot(series_df[date_col], series_df[pred_col], "-", label="Predicción", color="orange")
    plt.fill_between(series_df[date_col], series_df[lower_col], series_df[upper_col], color="lightblue", alpha=0.3, label="Intervalo")
    plt.axvspan(train_df[date_col].min(), train_df[date_col].max(), color="green", alpha=0.1, label="Entrenamiento")
    plt.axvspan(calib_df[date_col].min(), calib_df[date_col].max(), color="yellow", alpha=0.1, label="Calibración")
    plt.axvspan(test_df[date_col].min(), test_df[date_col].max(), color="red", alpha=0.1, label="Prueba")
    plt.title(f"Resultados para la Serie: {key}")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_logs(logs):
    """
    Genera gráficos de los logs para analizar el comportamiento de los componentes.
    """
    time_steps = range(len(logs["proportional"]))

    # Gráfico de Componentes
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, logs["proportional"], label="Proporcional")
    plt.plot(time_steps, logs["integral"], label="Integral")
    plt.plot(time_steps, logs["derivative"], label="Derivativo")
    plt.legend()
    plt.title("Contribución de Componentes en el Tiempo")
    plt.xlabel("Pasos Temporales")
    plt.ylabel("Valor")
    plt.show()

    # Gráfico de Cobertura
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, logs["coverage"], label="Cobertura Marginal")
    plt.axhline(y=0.9, color="r", linestyle="--", label="Umbral de Cobertura (90%)")
    plt.legend()
    plt.title("Cobertura Marginal en el Tiempo")
    plt.xlabel("Pasos Temporales")
    plt.ylabel("Cobertura")
    plt.show()


def save_logs_to_csv(logs, filename="component_logs.csv"):
    """
    Guarda los logs en un archivo CSV para análisis posterior.
    """
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(filename, index=False)
    print(f"Logs guardados en {filename}")
