# 🚀 Modelo Brain Max - Guía para Google Colab

## 📋 Instrucciones para Google Colab

### 1. Preparación del Entorno

1. **Abrir Google Colab**: Ve a [colab.research.google.com](https://colab.research.google.com)
2. **Crear un nuevo notebook** o usar uno existente
3. **Subir tu archivo CSV**: 
   - Ve al panel izquierdo y haz clic en el ícono de archivos 📁
   - Haz clic en "Subir" y selecciona tu archivo `EURUSD_M15.csv`
   - El archivo se guardará automáticamente en `/content/`

### 2. Instalación y Configuración

```python
# Ejecutar en una celda de Colab
!pip install yfinance xgboost lightgbm optuna pandas numpy scikit-learn
```

### 3. Subir el Script

```python
# Opción 1: Subir el archivo directamente
# Ve al panel de archivos y sube Modelo_Brain_Max.py

# Opción 2: Descargar desde GitHub (si está disponible)
!wget https://raw.githubusercontent.com/tu-repo/Modelo_Brain_Max.py
```

### 4. Ejecutar el Script

```python
# Ejecutar el script principal
!python Modelo_Brain_Max.py
```

### 5. Opciones Disponibles

Cuando ejecutes el script, verás un menú con las siguientes opciones:

- **Opción 1**: Entrenamiento completo (15-20 minutos)
- **Opción 2**: Análisis completo con métricas detalladas  
- **Opción 3**: Probar disponibilidad de datos (diagnóstico)
- **Opción 4**: Optimización específica para 85%+ accuracy (EURUSD)
- **Opción 5**: MULTI-ESTILOS Y MULTI-PARES
- **Opción 6**: 🎯 ENTRENAMIENTO INTERACTIVO (Recomendado)
- **Opción 7**: 🔍 PROBAR ACCESO A ARCHIVOS (diagnóstico)

### 6. Entrenamiento Interactivo (Opción 6)

1. **Selecciona la opción 6** cuando aparezca el menú
2. **Elige el par de divisas** (ej: EURUSD)
3. **Selecciona el estilo de trading**:
   - 1. Scalping (operaciones muy cortas)
   - 2. Day Trading (operaciones diarias) ⭐ Recomendado
   - 3. Swing Trading (operaciones de varios días)
   - 4. Position Trading (operaciones de largo plazo)
4. **Ingresa el número de meses** (1-24, recomendado: 3-12)
5. **Selecciona la fuente de datos**:
   - 1. Yahoo Finance (datos en tiempo real)
   - 2. Dataset de Kaggle (archivos CSV) ⭐ Para Colab

### 7. Solución de Problemas

#### ❌ No se detectan archivos CSV

Si el script no encuentra tu archivo CSV:

1. **Verifica que el archivo esté en `/content/`**:
   ```python
   import os
   print("Archivos en /content:", os.listdir("/content"))
   ```

2. **Usa la opción 7** para diagnóstico:
   - Ejecuta el script y selecciona opción 7
   - Te mostrará exactamente qué archivos encuentra

3. **Verifica el nombre del archivo**:
   - El script busca archivos que terminen en `.csv`
   - Prioriza archivos con nombres como `EURUSD_M15.csv`

#### ❌ Error de dependencias

Si hay errores de importación:

```python
# Instalar todas las dependencias necesarias
!pip install yfinance xgboost lightgbm optuna pandas numpy scikit-learn matplotlib seaborn
```

#### ❌ Error de memoria

Si Colab se queda sin memoria:

1. **Reinicia el runtime**: Runtime → Restart runtime
2. **Usa menos meses** de entrenamiento (1-3 meses)
3. **Cierra otros notebooks** que no estés usando

### 8. Estructura de Archivos Esperada

El script espera archivos CSV con las siguientes columnas:
- `Open`, `High`, `Low`, `Close`, `Volume`
- O columnas con nombres similares (el script las mapea automáticamente)

### 9. Resultados

Después del entrenamiento, encontrarás:
- **Modelos guardados** en `/content/models_PAIR_STYLE/`
- **Métricas de rendimiento** en la consola
- **Análisis detallado** del modelo

### 10. Consejos Adicionales

- **Usa GPU** si está disponible: Runtime → Change runtime type → GPU
- **Guarda tus resultados**: Los archivos se guardan automáticamente en `/content/`
- **Descarga los modelos**: Puedes descargar los modelos entrenados desde el panel de archivos

---

## 🆘 ¿Necesitas Ayuda?

Si tienes problemas:

1. **Ejecuta la opción 7** para diagnóstico
2. **Verifica que tu archivo CSV esté en `/content/`**
3. **Asegúrate de que el archivo tenga el formato correcto**
4. **Revisa los mensajes de error en la consola**

¡El script está optimizado para funcionar perfectamente en Google Colab! 🚀 