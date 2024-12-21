import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.forecasting.theta import ThetaModel
import matplotlib.pyplot as plt

# =====================
#  Preparación de Datos
# =====================
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
    scores, alpha, lr, T_burnin, Csat, KI, ahead, seasonal_period, time_index, integrate=True, proportional_lr=True, scorecast=True
):
    """
    Calcula intervalos dinámicos de predicción utilizando PDI.
    """
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))

    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue
        t_lr_min = max(t - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t].max() - scores[t_lr_min:t].min()) if proportional_lr and t > 0 else lr
        covereds[t] = qs[t] >= scores[t]
        grad = alpha if covereds[t_pred] else -(1 - alpha)
        integrator_arg = (1 - covereds)[:t_pred].sum() - (t_pred) * alpha
        integrator = np.tan(integrator_arg * np.log(t_pred + 1) / (Csat * (t_pred + 1))) if integrate else 0
        if scorecast and t_pred > T_burnin and t + ahead < T_test:
            score_series = pd.Series(scores[:t_pred], index=time_index[:t_pred])
            model = ThetaModel(score_series, period=seasonal_period)
            fitted_model = model.fit()
            scorecasts[t + ahead] = fitted_model.forecast(ahead).iloc[-1]
        if t < T_test - 1:
            qts[t + 1] = qts[t] - lr_t * grad
            integrators[t + 1] = integrator if integrate else 0
            qs[t + 1] = qts[t + 1] + integrators[t + 1]
            if scorecast:
                qs[t + 1] += scorecasts[t + 1]
    return qs


def apply_pdi_with_calibration(df, key_col, date_col, value_col, pred_col, lower_col, upper_col,
                               alpha, lr, T_burnin, Csat, KI, ahead, seasonal_period, set_col):
    """
    Aplica PDI a cada serie en el DataFrame.
    """
    results = []
    for key, group in df.groupby(key_col):
        group = group.sort_values(by=date_col)
        train, calib, test = split_data_by_set(group, set_col=set_col)

        y_train = train[value_col].dropna().values
        y_pred_train = train[pred_col].dropna().values
        scores_train = np.abs(y_train - y_pred_train)
        time_index_train = pd.to_datetime(train[date_col])

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

        # Generar intervalos para calibración y prueba
        last_calib_quantile = qs_calib[-1]
        calib[lower_col] = calib[pred_col] - last_calib_quantile
        calib[upper_col] = calib[pred_col] + last_calib_quantile
        test[lower_col] = test[pred_col] - last_calib_quantile
        test[upper_col] = test[pred_col] + last_calib_quantile


        results.append(pd.concat([train, calib, test]))

    return pd.concat(results)


# =========================
#  Evaluación de Resultados
# =========================
def calculate_metrics(df, value_col, lower_col, upper_col, condition_col=None):
    """
    Calcula Marginal Coverage, Average Region Size y Conditional Coverage.

    Parámetros:
    - df (pd.DataFrame): DataFrame con los datos.
    - value_col (str): Columna con los valores reales.
    - lower_col (str): Columna con los límites inferiores de los intervalos.
    - upper_col (str): Columna con los límites superiores de los intervalos.
    - condition_col (str, opcional): Columna para agrupar (por ejemplo, 'KEY').

    Retorna:
    - pd.DataFrame: DataFrame con las métricas calculadas.
    """
    if condition_col:
        # Calcular métricas por grupo (por KEY u otra condición)
        metrics = (
            df.groupby(condition_col)
            .apply(lambda group: pd.Series({
                "Marginal Coverage": ((group[value_col] >= group[lower_col]) & 
                                      (group[value_col] <= group[upper_col])).mean(),
                "Average Region Size": (group[upper_col] - group[lower_col]).mean(),
                "Conditional Coverage": ((group[value_col] >= group[lower_col]) & 
                                         (group[value_col] <= group[upper_col])).mean()
            }))
            .reset_index()
        )
    else:
        # Calcular métricas globales
        metrics = pd.DataFrame([{
            "Marginal Coverage": ((df[value_col] >= df[lower_col]) & 
                                  (df[value_col] <= df[upper_col])).mean(),
            "Average Region Size": (df[upper_col] - df[lower_col]).mean(),
            "Conditional Coverage": ((df[value_col] >= df[lower_col]) & 
                                     (df[value_col] <= df[upper_col])).mean()
        }])

    return metrics


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
