# PDIHC
PDI Haute Couture

### **1. Objetivo**
Discutir y entender la implementación del método **PDI (Proportional-Integral-Derivative)** para generar intervalos de predicción dinámicos en series temporales. Además, adaptar este método para trabajar con un `DataFrame` que contiene múltiples series identificadas por una clave (`KEY`), incluyendo la consideración de datos futuros sin valores reales.

---

### **2. Desarrollo**

#### **A. Implementación Inicial de PDI**
- Implementamos el método `quantile_integrator_log_scorecaster`, el cual:
  - Ajusta los intervalos dinámicamente en función de tres componentes:
    - **Proporcional (P)**: Ajusta el intervalo con base en el error reciente.
    - **Integral (I)**: Corrige desviaciones acumulativas de cobertura.
    - **Derivativo (D)**: Anticipa cambios futuros usando un modelo `ThetaModel`.
  - Considera parámetros clave como la tasa de aprendizaje dinámica (`lr`), el período de calentamiento (`T_burnin`), y el horizonte de predicción (`ahead`).

#### **B. Problema Inicial**
- Los datos históricos se procesaron correctamente, pero no se generaban intervalos para datos futuros donde no existían valores reales (`Y`), identificados con la columna `FUTURE`.

#### **C. Modificaciones al Método**
1. **Incorporación de Datos Futuros**:
   - Se ajustó el método para diferenciar datos históricos y futuros usando la columna `FUTURE`.
   - Para datos futuros:
     - Se utilizó el último intervalo ajustado como base.
     - Se agregó un componente derivativo predicho usando `ThetaModel`.

2. **Manejo de Casos Extremos**:
   - Se añadió una validación para manejar casos donde los datos históricos sean insuficientes o faltantes.
   - En estos casos, se usa la media de los errores históricos para estimar intervalos futuros.

3. **Validaciones de Índices y Máscaras**:
   - Se corrigieron problemas relacionados con el desajuste entre índices y máscaras booleanas (`is_future`).

#### **D. Función Final**
La función final implementada fue `apply_pdi_to_dataframe_with_future`, la cual:
- Recibe un `DataFrame` con múltiples series (`KEY`), valores históricos (`Y`), predicciones (`YHATFIN`), y un indicador de futuro (`FUTURE`).
- Genera bandas de predicción ajustadas (`YHAT_L` y `YHAT_U`) tanto para datos históricos como futuros.

#### **E. Resultados**
1. **Cálculo de Intervalos**:
   - Se logró calcular intervalos dinámicos de predicción para cada serie y cada período, respetando las especificaciones de PDI.

2. **Visualización**:
   - Se discutieron opciones para visualizar los resultados, incluyendo el uso de `Plotly` para una representación interactiva de los intervalos.

---

### **3. Parámetros del Modelo**
Los parámetros clave del modelo PDI fueron discutidos y explicados detalladamente:

- **`alpha`**: Nivel de significancia (determina la cobertura deseada del intervalo).
- **`lr`**: Tasa de aprendizaje para ajustar la componente proporcional.
- **`T_burnin`**: Período inicial donde no se aplica ajuste integral ni derivativo.
- **`Csat`**: Controla la saturación del componente integral para evitar explosiones en el ajuste.
- **`KI`**: Escala del componente integral.
- **`ahead`**: Horizonte de predicción.
- **`seasonal_period`**: Período de estacionalidad para desestacionalizar datos en el modelo derivativo.

---

### **4. Código Final**

#### **Método PDI:**
El método `quantile_integrator_log_scorecaster` ajusta los intervalos dinámicamente según la formulación PID descrita.

#### **Aplicación del Método:**
La función `apply_pdi_to_dataframe_with_future` adapta PDI a un `DataFrame` con múltiples series y datos futuros.

---

### **5. Principales Problemas Resueltos**
1. **Errores en el Índice Temporal (`time_index`)**:
   - Se corrigió la alineación entre los datos históricos y los índices temporales.
2. **Incorporación de Datos Futuros**:
   - Ahora se generan intervalos consistentes para predicciones futuras.
3. **Problemas con la Máscara `is_future`**:
   - Se validaron los tipos y la alineación de las máscaras booleanas.

---

### **6. Próximos Pasos**
1. **Optimización del Código**:
   - Revisar la eficiencia en el cálculo de intervalos para grandes volúmenes de datos.
2. **Validación Adicional**:
   - Evaluar la cobertura y el rendimiento de los intervalos ajustados en datos reales.
3. **Extensión del Modelo**:
   - Considerar incluir variables exógenas en el modelo derivativo.

---

Este resumen cubre los aspectos clave de la implementación, los problemas encontrados y las soluciones aplicadas. Si necesitas más detalles en algún punto específico, ¡házmelo saber! 🚀