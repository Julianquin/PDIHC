# PIDHC - PID Haute Couture


## **1. Objetivo**

Este proyecto tiene como objetivo principal implementar, adaptar y evaluar el método **PID (Proportional-Integral-Derivative)** para generar intervalos de predicción dinámicos en series temporales. El método está diseñado para cumplir con una cobertura específica $1-\alpha$, ajustándose en línea con base en tres componentes principales: proporcional, integral y derivativo.

Además, se busca integrar este enfoque en un marco práctico que permita:
- Trabajar con un `DataFrame` que contenga múltiples series temporales identificadas por una clave (`KEY`).
- Diferenciar entre datos históricos y futuros, generando intervalos para ambos escenarios.
- Implementar técnicas de evaluación para garantizar la cobertura y estabilidad de los intervalos generados.


## **2. Desarrollo**

### **A. Preparación de Datos**
1. **Separación en Conjuntos**:
   - Se implementó la función `assign_data_sets` para dividir las series en tres conjuntos:
     - **TRAIN**: Conjunto de entrenamiento, utilizado para ajustar los intervalos dinámicos.
     - **CALIBRATION**: Conjunto de calibración, utilizado para validar y estabilizar los intervalos ajustados.
     - **TEST**: Conjunto de prueba, destinado a datos futuros donde no se tienen valores reales (`Y`).

   - La asignación respeta las siguientes reglas:
     - Datos con `FUTURE = 1` pertenecen al conjunto de prueba (`TEST`).
     - Datos con `FUTURE = 0` se dividen en TRAIN y CALIBRATION según un porcentaje configurable (`calib_ratio`).

2. **Función `split_data_by_set`**:
   - Permite dividir los datos de una serie específica en los conjuntos mencionados anteriormente.

### **B. Implementación del Método PID**

#### **1. Método Principal: `quantile_integrator_log_scorecaster`**
- Este método ajusta los intervalos dinámicamente utilizando los tres componentes PID:
  - **Proporcional (P)**:
    - Ajusta el intervalo en función del error reciente.
    - Utiliza la tasa de aprendizaje dinámica `lr_t` calculada según la variación reciente en los errores.
  - **Integral (I)**:
    - Corrige desviaciones acumulativas en la cobertura observada respecto a la deseada.
    - Implementa un integrador no lineal basado en la función tangente.
  - **Derivativo (D)**:
    - Anticipa cambios futuros en los errores utilizando el modelo `ThetaModel`.

#### **2. Aplicación del Método a Datos Reales**
- La función `apply_PID_with_calibration` adapta el método PID a un `DataFrame` que contiene múltiples series.
  - Para cada serie:
    - Ajusta los intervalos dinámicamente en el conjunto de entrenamiento (`TRAIN`).
    - Genera intervalos consistentes para los conjuntos de calibración (`CALIBRATION`) y prueba (`TEST`).
  - Los intervalos generados para calibración y prueba se basan en el último cuantil ajustado en el conjunto de entrenamiento.

### **C. Evaluación de Resultados**

#### **1. Métricas Calculadas**
- **Cobertura Marginal (`Marginal Coverage`)**:
  - Porcentaje de valores reales que caen dentro del intervalo de predicción.
- **Tamaño Promedio del Intervalo (`Average Region Size`)**:
  - Longitud promedio de los intervalos generados.
- **Cobertura Condicional (`Conditional Coverage`)**:
  - Cobertura evaluada para cada serie (`KEY`).

#### **2. Función de Evaluación: `calculate_metrics`**
- Centraliza el cálculo de las tres métricas mencionadas.
- Permite calcular métricas globales (a nivel del DataFrame) o por serie (a nivel de `KEY`).

#### **3. Visualización de Resultados**
- La función `plot_series_results_with_sets` genera gráficos individuales para cada serie, mostrando:
  - Valores reales y predicciones.
  - Intervalos de predicción generados.
  - Periodos marcados para los conjuntos TRAIN, CALIBRATION y TEST.



## **3. Código Final**

### **A. Preparación de Datos**
- `assign_data_sets`: Asigna etiquetas de conjunto (TRAIN, CALIBRATION, TEST).
- `split_data_by_set`: Divide un grupo en los conjuntos mencionados.

### **B. Cálculo de Intervalos**
- `quantile_integrator_log_scorecaster`: Implementa el método PID para ajustar intervalos dinámicos.
- `apply_PID_with_calibration`: Aplica PID a un DataFrame con múltiples series.

### **C. Evaluación y Visualización**
- `calculate_metrics`: Calcula métricas de evaluación (Marginal Coverage, Average Region Size, Conditional Coverage).
- `plot_series_results_with_sets`: Visualiza resultados individuales por serie.


## **4. Parámetros del Modelo**

Los parámetros principales del modelo PID incluyen:

1. **Cobertura y Aprendizaje**:
   - `alpha`: Nivel de significancia deseado, que determina la cobertura esperada ($1 - \alpha$).
   - `lr`: Tasa de aprendizaje para ajustar dinámicamente la componente proporcional.

2. **Estabilidad y Predicción**:
   - `T_burnin`: Período inicial para estabilizar el modelo antes de aplicar componentes integral o derivativo.
   - `Csat`: Constante de saturación para evitar ajustes excesivos en la componente integral.
   - `KI`: Escala del componente integral.
   - `ahead`: Horizonte de predicción (número de pasos futuros a predecir).
   - `seasonal_period`: Período de estacionalidad para ajustar predicciones con el modelo `ThetaModel`.

3. **División de Datos**:
   - `calib_ratio`: Proporción de datos históricos asignados al conjunto de calibración.


## **5. Resultados Actuales**

1. **Cálculo de Intervalos**:
   - Intervalos dinámicos generados consistentemente para datos históricos y futuros.
   - Integración completa de los componentes PID (Proporcional, Integral y Derivativo).

2. **Evaluación**:
   - Métricas calculadas para cada serie:
     - Cobertura Marginal
     - Tamaño Promedio del Intervalo
     - Cobertura Condicional

3. **Visualización**:
   - Gráficos que destacan los conjuntos TRAIN, CALIBRATION y TEST, mostrando la alineación con los valores reales y predicciones.


## **6. Próximos Pasos**
1. **Optimización del Código**:
   - Revisar la eficiencia en la generación de intervalos para grandes volúmenes de datos.
   - Paralelización para múltiples series.

2. **Extensión del Modelo**:
   - Incorporar variables exógenas en el modelo derivativo.
   - Experimentar con otros métodos de scorecasting.

3. **Validación Adicional**:
   - Evaluar el rendimiento en datos reales y comparar con enfoques alternativos.




PLOTEAR $q_{t}$ de los scores
