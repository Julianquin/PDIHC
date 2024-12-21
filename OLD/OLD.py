def apply_pdi_with_dynamic_calibration(
    df, key_col, date_col, value_col, pred_col, lower_col, upper_col,
    alpha, lr, T_burnin, Csat, KI, ahead, seasonal_period, set_col
):
    """
    Aplica PDI a cada serie en el DataFrame con recalibración dinámica en el conjunto de calibración.

    Parámetros:
    - df (pd.DataFrame): DataFrame con datos de entrada.
    - key_col (str): Columna que identifica cada serie.
    - date_col (str): Columna con las fechas.
    - value_col (str): Columna con valores reales.
    - pred_col (str): Columna con las predicciones.
    - lower_col (str): Columna para los límites inferiores de los intervalos.
    - upper_col (str): Columna para los límites superiores de los intervalos.
    - alpha (float): Nivel de significancia para PDI.
    - lr (float): Tasa de aprendizaje para el ajuste.
    - T_burnin (int): Periodo de calentamiento.
    - Csat (float): Parámetro de saturación del componente integral.
    - KI (float): Escala del componente integral.
    - ahead (int): Horizonte de predicción.
    - seasonal_period (int): Periodo de estacionalidad para el modelo Theta.
    - set_col (str): Columna que indica el conjunto (TRAIN, CALIBRATION, TEST).

    Retorna:
    - pd.DataFrame: DataFrame con intervalos ajustados para cada conjunto.
    """
    results = []

    for key, group in df.groupby(key_col):
        group = group.sort_values(by=date_col)

        # Dividir en conjuntos de entrenamiento, calibración y prueba
        train, calib, test = split_data_by_set(group, set_col=set_col)

        # Calcular intervalos para el conjunto de entrenamiento
        y_train = train[value_col].dropna().values
        y_pred_train = train[pred_col].dropna().values
        scores_train = np.abs(y_train - y_pred_train)
        time_index_train = pd.to_datetime(train[date_col])

        qs_train = quantile_integrator_log_scorecaster(
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
        train[lower_col] = train[pred_col] - qs_train
        train[upper_col] = train[pred_col] + qs_train

        # Recalibración dinámica en el conjunto de calibración
        if not calib.empty:
            y_calib = calib[value_col].dropna().values
            y_pred_calib = calib[pred_col].dropna().values
            scores_calib = np.abs(y_calib - y_pred_calib)
            time_index_calib = pd.to_datetime(calib[date_col])

            qs_calib = quantile_integrator_log_scorecaster(
                scores=scores_calib,
                alpha=alpha,
                lr=lr,
                T_burnin=T_burnin,
                Csat=Csat,
                KI=KI,
                ahead=ahead,
                seasonal_period=seasonal_period,
                time_index=time_index_calib
            )

            # Generar intervalos para el conjunto de calibración
            calib[lower_col] = calib[pred_col] - qs_calib
            calib[upper_col] = calib[pred_col] + qs_calib

            # Usar el último cuantil ajustado de calibración para extender al conjunto de prueba
            last_calib_quantile = qs_calib[-1]
        else:
            # Si el conjunto de calibración está vacío, usar el último del entrenamiento
            last_calib_quantile = qs_train[-1] if len(qs_train) > 0 else 0

        # Generar intervalos para el conjunto de prueba
        test[lower_col] = test[pred_col] - last_calib_quantile
        test[upper_col] = test[pred_col] + last_calib_quantile

        # Concatenar los resultados
        results.append(pd.concat([train, calib, test]))

    return pd.concat(results)

df_pdi2 = apply_pdi_with_dynamic_calibration(
    df=df,
    key_col="KEY",
    date_col="FECHA",
    value_col="Y",
    pred_col="YHATFIN",
    lower_col="YHAT_L",
    upper_col="YHAT_U",
    alpha=alpha,
    lr=lr,
    T_burnin=T_burnin,
    Csat=Csat,
    KI=KI,
    ahead=ahead,
    seasonal_period=seasonal_period,
    set_col="SET"
)

# Dividir el DataFrame en TRAIN, CALIBRATION y TEST
train_df = df_pdi2[df_pdi["SET"] == "TRAIN"]
calib_df = df_pdi2[df_pdi["SET"] == "CALIBRATION"]
test_df = df_pdi2[df_pdi["SET"] == "TEST"]

# Calcular métricas por serie (KEY) para el conjunto de calibración
train_metrics_by_key = calculate_metrics(
    train_df,
    value_col="Y",
    lower_col="YHAT_L",
    upper_col="YHAT_U",
    condition_col="KEY"
)

# Mostrar resultados
print("Métricas por KEY (Train):")
print(train_metrics_by_key)

# Calcular métricas por serie (KEY) para el conjunto de calibración
calib_metrics_by_key = calculate_metrics(
    calib_df,
    value_col="Y",
    lower_col="YHAT_L",
    upper_col="YHAT_U",
    condition_col="KEY"
)

# Mostrar resultados
print("Métricas por KEY (Calibración):")
print(calib_metrics_by_key)