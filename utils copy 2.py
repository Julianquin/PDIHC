import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.forecasting.theta import ThetaModel
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

from prophet import Prophet
import lightgbm as lgb
from sklearn.linear_model import Ridge

# =====================
#  Baseline Methods
# =====================


def trailing_window(
    scores,
    alpha,
    lr,  # Argumento dummy
    weight_length,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if min(weight_length, t_pred) < np.ceil(1 / alpha):
            qs[t] = np.inf  # Cambiar np.infty por np.inf
        else:
            qs[t] = np.quantile(scores[max(t_pred - weight_length, 0):t_pred], 1 - alpha, method='higher')
    results = {"method": "Trail", "q": qs}
    return results

def aci_clipped(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.inf
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.inf
        if qs[t] == np.inf:
            qs[t] = clip_value
    results = { "method": "ACI (clipped)", "q" : qs, "alpha" : alphas}
    return results

def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.inf
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.inf
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}
    return results

# =====================
#  Preparación de Datos
# =====================

def generar_datos(n_series=3, n_points=120, seed=42, start_date="2023-01-01", noise_std=3,fore_std=7):
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
        noise = np.random.normal(0, noise_std, n_points)  # Ruido aleatorio
        
        # Valores reales y predicciones
        y_real = trend + weekly_seasonality + monthly_seasonality + noise
        y_pred = y_real + np.random.normal(0, fore_std, n_points)  # Agregar ruido adicional a las predicciones
        
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
#  Funciones Auxiliares - PDI
# ==============================

def mytan(x):
    if x >= np.pi/2:
        return np.inf
    elif x <= -np.pi/2:
        return -np.inf
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))

def calculate_dynamic_learning_rate(
    scores, 
    t_lr_min, 
    t, 
    lr, 
    alpha, 
    covereds, 
    option
):
    """
    Calcula la tasa de aprendizaje dinámica basada en varias opciones.

    Parámetros:
    - scores (np.ndarray): Errores absolutos.
    - t_lr_min (int): Índice mínimo para el cálculo de `lr_t`.
    - t (int): Índice actual.
    - lr (float): Tasa de aprendizaje base.
    - alpha (float): Nivel de significancia.
    - covereds (np.ndarray): Indicadores de cobertura.
    - option (str): Método de cálculo ('proportional_range', 'simple', 'smoothed', 'iqr', etc.).

    Retorna:
    - lr_t (float): Tasa de aprendizaje dinámica.
    """
    if t == 0 or len(scores[t_lr_min:t]) == 0:
        return lr

    range_scores = scores[t_lr_min:t].max() - scores[t_lr_min:t].min()

    if option == "proportional_range":
        return lr * range_scores 
    
    
    elif option == "smoothed":
        smoothed_range = pd.Series([range_scores]).rolling(window=3, min_periods=1).mean().iloc[-1]
        return lr * smoothed_range
    
    elif option == "dynamic_limited":
        if len(scores[t_lr_min:t]) >= 2:  # Asegura suficientes datos para cálculos
            std_dev = np.std(scores[t_lr_min:t]) + 1e-8  # Evita división por cero
            lr_min_dynamic = 0.01 * lr  # Escalado mínimo fijo
            lr_max_dynamic = lr * (1 + range_scores / std_dev)  # Escalado dinámico máximo
            return np.clip(lr * range_scores, lr_min_dynamic, lr_max_dynamic)
        else:
            return lr  

    elif option == "iqr":
        if len(scores[t_lr_min:t]) >= 2:
            iqr_scores = np.percentile(scores[t_lr_min:t], 75) - np.percentile(scores[t_lr_min:t], 25)
            return lr * iqr_scores
        else:
            return lr
    
    elif option == "gradient_history":
        grad_history = np.cumsum([alpha if covered else -(1 - alpha) for covered in covereds[t_lr_min:t]])
        return lr * range_scores / (1 + np.abs(grad_history[-1])) if len(grad_history) > 0 else lr
    
    elif option == "scaling_factor":
        scaling_factor = np.sqrt(t + 1)
        return lr * range_scores / scaling_factor
    else:
        raise ValueError(f"Opción desconocida para la tasa de aprendizaje: {option}")

def create_features(scores, time_index, exogenous_vars=None, ahead=30):
    """
    Genera características para modelos basados en regresión a partir de una serie temporal.

    Parámetros:
    - scores (np.ndarray): Serie temporal de valores.
    - time_index (pd.DatetimeIndex): Índices de tiempo correspondientes a los valores.
    - exogenous_vars (pd.DataFrame): Variables exógenas correspondientes al tiempo (opcional).
    - ahead (int): Horizonte de predicción en días.

    Retorna:
    - pd.DataFrame: DataFrame con las características generadas.
    """
    # Convertir a DataFrame
    df = pd.DataFrame({
        'y': scores,
        'time': time_index
    })

    # Crear características temporales
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_of_month'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Rolling statistics (ventanas móviles)
    rolling_windows = [7, 14, 30]
    # for window in rolling_windows:
    #     df[f'rolling_mean_{window}'] = df['y'].shift(ahead).rolling(window=window).mean()
    #     df[f'rolling_std_{window}'] = df['y'].shift(ahead).rolling(window=window).std()

    # Incluir variables exógenas si están disponibles
    if exogenous_vars is not None:
        for col in exogenous_vars.columns:
            df[col] = exogenous_vars[col].values

    # Eliminar filas con valores NaN debido a lags y rolling stats
    df = df.dropna().reset_index(drop=True)

    # Eliminar columna de tiempo si no es necesaria
    df = df.drop(columns=['time'], errors='ignore')

    return df

def generic_forecaster(model_type, scores, time_index, seasonal_period, ahead, exogenous_vars=None):
    """
    Función genérica para realizar predicciones usando diferentes modelos.

    Parámetros:
    - model_type (str): Tipo de modelo a usar ('theta', 'prophet', 'ridge', 'lightgbm').
    - scores (np.ndarray): Valores históricos de la serie temporal.
    - time_index (pd.DatetimeIndex): Índices de tiempo correspondientes a los valores.
    - seasonal_period (int): Período estacional (usado en Theta y Prophet).
    - ahead (int): Horizonte de predicción.
    - exogenous_vars (pd.DataFrame): Variables exógenas correspondientes al tiempo (opcional).

    Retorna:
    - float: Predicción para el horizonte especificado.
    """
    if model_type == "theta":
        score_series = pd.Series(scores, index=pd.date_range(start=time_index[0], periods=len(scores), freq='D'))
        model = ThetaModel(score_series, period=seasonal_period)
        fitted_model = model.fit()
        return fitted_model.forecast(ahead).iloc[-1]
    
    elif model_type == "prophet":
        df = pd.DataFrame({'ds': time_index, 'y': scores})
        if exogenous_vars is not None:
            for col in exogenous_vars.columns:
                df[col] = exogenous_vars[col]
        model = Prophet()
        if exogenous_vars is not None:
            for col in exogenous_vars.columns:
                model.add_regressor(col)
        model.fit(df)
        future = model.make_future_dataframe(periods=ahead)
        if exogenous_vars is not None:
            for col in exogenous_vars.columns:
                future[col] = exogenous_vars[col].iloc[-ahead:]
        forecast = model.predict(future)
        return forecast.iloc[-1]['yhat']

    elif model_type == "ridge":
        # Suponiendo que se han generado características para Ridge
        from sklearn.linear_model import Ridge
        features = create_features(scores, time_index, exogenous_vars)
        model = Ridge()
        X_train, y_train = features[:-ahead], scores[:-ahead]
        model.fit(X_train, y_train)
        return model.predict(features[-ahead:])[0]

    elif model_type == "lightgbm":
        import lightgbm as lgb
        features = create_features(scores, time_index, exogenous_vars)
        X_train, y_train = features[:-ahead], scores[:-ahead]
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {'objective': 'regression', 'boosting_type': 'gbdt', 'verbosity': -1}
        model = lgb.train(params, train_data, num_boost_round=100)
        return model.predict(features[-ahead:])[0]

    else:
        raise ValueError(f"Modelo '{model_type}' no soportado.")


# ==============================
#  Cálculo de Intervalos - PDI
# ==============================

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
    lr_option,
    integrate,  
    scorecast, 
    model_type
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
        lr_t = calculate_dynamic_learning_rate(
            scores=scores,
            t_lr_min=t_lr_min,
            t=t,
            lr=lr,
            alpha=alpha,
            covereds=covereds,
            option=lr_option
        )


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
        # integrator = (np.tan(integrator_arg * np.log(t_pred + 1) / (Csat * (t_pred + 1))) if integrate else 0 )
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI) if integrate else 0
        integrators[t] = integrator
        logs["integral"].append(integrator)

        # ================================
        # Componente Derivativo (D)
        # ================================

        if scorecast and t_pred > T_burnin and t + ahead < T_test:
            model_filename = f"./model_cache/scorecaster_{t_pred}.pkl"
            scorecasts[t + ahead] = generic_forecaster(
                model_type=model_type,  # Cambia a "theta" "prophet", "ridge", o "lightgbm" según lo que quieras probar
                scores=scores[:t_pred],
                time_index=pd.date_range(start=time_index.iloc[0], periods=len(scores[:t_pred]), freq='D'),
                seasonal_period=seasonal_period,
                ahead=ahead,
                exogenous_vars=None  # Cambia esto si tienes variables exógenas
            )


        logs["derivative"].append(scorecasts[t])

        # Registrar cobertura
        logs["coverage"].append(covereds[t])

        # Alertar si la cobertura es baja
        # if covereds[t] < coverage_threshold:
        #     print(f"⚠️ Cobertura baja en el paso {t}: {covereds[t]}")

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
    set_col,
    lr_option, 
    integrate,
    scorecast,
    model_type
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
    all_logs = {}  # Diccionario para almacenar los logs por serie

    # Iterar por cada serie identificada por key_col
    for key, group in tqdm(df.groupby(key_col)):
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


        if len(scores_train) == 0:
            print(f"⚠️ Conjunto de entrenamiento vacío para {key}.")
            continue

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
            time_index=time_index_train,
            lr_option=lr_option,
            integrate=integrate,
            scorecast=scorecast,
            model_type=model_type,
        )
        
        if len(qs_calib) == 0:
            print(f"⚠️ No se generaron cuantiles para {key}.")
            continue

        all_logs[key] = logs

        # Generar intervalos para el conjunto de entrenamiento
        train.loc[:,lower_col] = train[pred_col] - qs_calib
        train.loc[:,upper_col] = train[pred_col] + qs_calib

        # ==============================
        # Datos de Calibración y Prueba
        # ==============================
        last_calib_quantile = qs_calib[-1]  # Último cuantil ajustado del entrenamiento
        # Calibración
        calib.loc[:,lower_col] = calib[pred_col] - last_calib_quantile
        calib.loc[:,upper_col] = calib[pred_col] + last_calib_quantile
        # Prueba
        test.loc[:,lower_col] = test[pred_col] - last_calib_quantile
        test.loc[:,upper_col] = test[pred_col] + last_calib_quantile

        # Agregar los conjuntos procesados a la lista de resultados
        results.append(pd.concat([train, calib, test]))

    # Combinar los resultados procesados para todas las series
    return pd.concat(results), all_logs

# =========================
#  Evaluación de Resultados
# =========================

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
    for key, series_logs in logs.items():
        time_steps = range(len(series_logs["proportional"]))

        # Gráfico de Componentes
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, series_logs["proportional"], label="Proporcional")
        plt.plot(time_steps, series_logs["integral"], label="Integral")
        plt.plot(time_steps, series_logs["derivative"], label="Derivativo")
        plt.legend()
        plt.title(f"Contribución de Componentes - {key}")
        plt.xlabel("Pasos Temporales")
        plt.ylabel("Valor")
        plt.show()

        # Gráfico de Cobertura
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, series_logs["coverage"], label="Cobertura Marginal")
        plt.title(f"Cobertura Marginal - {key}")
        plt.xlabel("Pasos Temporales")
        plt.ylabel("Cobertura")
        plt.show()



def plot_heatmap(logs, component):
    all_series = []
    max_length = max(len(series_logs[component]) for series_logs in logs.values())
    
    for series_logs in logs.values():
        series_data = series_logs[component]
        padded_data = np.pad(series_data, (0, max_length - len(series_data)), mode="constant", constant_values=np.nan)
        all_series.append(padded_data)
    
    heatmap_data = np.array(all_series)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="viridis", cbar=True)
    plt.title(f"Heatmap de {component.capitalize()}")
    plt.xlabel("Pasos Temporales")
    plt.ylabel("Series")
    plt.show()


def aggregate_logs_average(logs):
    aggregated_logs = {
        "proportional": [],
        "integral": [],
        "derivative": [],
        "coverage": []
    }
    max_length = max(len(series_logs["proportional"]) for series_logs in logs.values())
    
    for t in range(max_length):
        proportional_vals = [series_logs["proportional"][t] for series_logs in logs.values() if t < len(series_logs["proportional"])]
        integral_vals = [series_logs["integral"][t] for series_logs in logs.values() if t < len(series_logs["integral"])]
        derivative_vals = [series_logs["derivative"][t] for series_logs in logs.values() if t < len(series_logs["derivative"])]
        coverage_vals = [series_logs["coverage"][t] for series_logs in logs.values() if t < len(series_logs["coverage"])]
        
        aggregated_logs["proportional"].append(np.mean(proportional_vals))
        aggregated_logs["integral"].append(np.mean(integral_vals))
        aggregated_logs["derivative"].append(np.mean(derivative_vals))
        aggregated_logs["coverage"].append(np.mean(coverage_vals))
    
    return aggregated_logs

def plot_logs_agg(logs):
    """
    Genera gráficos de los logs para analizar el comportamiento agregado de los componentes y la cobertura.
    
    Parámetros:
    - logs (dict): Diccionario de componentes con listas de valores promedio por paso temporal.
    """
    time_steps = range(len(next(iter(logs.values()))))  # Número de pasos temporales

    # Gráfico de Contribuciones (Proporcional, Integral, Derivativa)
    plt.figure(figsize=(12, 6))
    for component in ["proportional", "integral", "derivative"]:
        if component in logs:  # Asegurarse de que el componente existe en los logs
            plt.plot(time_steps, logs[component], label=component.capitalize())
    plt.legend()
    plt.title("Contribuciones Promedio de los Componentes en el Tiempo")
    plt.xlabel("Pasos Temporales")
    plt.ylabel("Valor Promedio")
    plt.show()

    # Gráfico de Cobertura
    if "coverage" in logs:  # Asegurarse de que la cobertura existe en los logs
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, logs["coverage"], label="Cobertura")
        plt.axhline(y=0.95, color="r", linestyle="--", label="Umbral de Cobertura (95%)")
        plt.legend()
        plt.title("Cobertura Promedio en el Tiempo")
        plt.xlabel("Pasos Temporales")
        plt.ylabel("Cobertura Promedio")
        plt.show()


def save_logs_to_csv(logs, filename="component_logs.csv"):
    """
    Guarda los logs en un archivo CSV para análisis posterior.
    """
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(filename, index=False)
    print(f"Logs guardados en {filename}")

def save_model(model, filename):
    """
    Guarda un modelo entrenado en un archivo.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """
    Carga un modelo previamente entrenado desde un archivo.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)



