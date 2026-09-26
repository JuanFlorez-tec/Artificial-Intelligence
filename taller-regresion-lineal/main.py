import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("=" * 60)
    print(" PASO 1: Carga e Inspección Inicial de Datos ")
    print("=" * 60)
    
    # Cargar el dataset especificando el separador ';'
    try:
        data = pd.read_csv('student-mat.csv', sep=';')
        print("Dataset cargado exitosamente.")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'student-mat.csv'. Verifique la ubicación.")
        return

    print(f"\nDimensiones iniciales del dataset: {data.shape[0]} filas, {data.shape[1]} columnas.\n")
    print("--- Primeras 5 filas ---")
    print(data.head())
    
    print("\n--- Tipos de Datos por Columna ---")
    print(data.dtypes)

    print("\n" + "=" * 60)
    print(" PASO 2: Transformación de Variables Categóricas ")
    print("=" * 60)
    
    # Mapear la variable categórica famsize de ('LE3', 'GT3') a (0, 1)
    mapa_tamanho_familia = {'LE3': 0, 'GT3': 1}
    data_modificada = data.copy()
    data_modificada['famsize'] = data_modificada['famsize'].map(mapa_tamanho_familia)
    
    print("Mapeo realizado en columna 'famsize' ('LE3': 0, 'GT3': 1).")
    print(data_modificada[['famsize']].head())

    print("\n" + "=" * 60)
    print(" PASO 3: Análisis Estadístico y Limpieza de Datos ")
    print("=" * 60)
    
    print("--- Resumen Estadístico de Variables Numéricas ---")
    print(data_modificada.describe())
    
    print("\n--- Verificación de Valores Nulos ---")
    nulos = data_modificada.isnull().sum()
    print(nulos[nulos > 0] if nulos.sum() > 0 else "No existen valores nulos en el dataset.")

    # Detección y filtrado de datos atípicos (outliers) mediante IQR
    Q1 = data_modificada.quantile(0.25, numeric_only=True)
    Q3 = data_modificada.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    
    # Filtrar el DataFrame excluyendo registros con atípicos en variables numéricas
    num_cols = data_modificada.select_dtypes(include='number').columns
    condicion_atipicos = ((data_modificada[num_cols] < (Q1 - 1.5 * IQR)) | 
                          (data_modificada[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    data_sin_atipicos = data_modificada[~condicion_atipicos].copy()
    print(f"\nDimensiones tras filtrar atípicos (IQR): {data_sin_atipicos.shape[0]} filas.")

    print("\n" + "=" * 60)
    print(" PASO 4: Matriz de Correlación ")
    print("=" * 60)
    
    matriz_correlacion = data_sin_atipicos.corr(numeric_only=True)
    print("Top 5 correlaciones con la nota final (G3):")
    print(matriz_correlacion['G3'].sort_values(ascending=False).head(6))

    # Guardar heatmap como imagen
    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Matriz de Correlación - Rendimiento de Estudiantes")
    plt.tight_layout()
    plt.savefig('matriz_correlacion.png')
    plt.close()
    print("\nGráfico 'matriz_correlacion.png' guardado con éxito.")

    print("\n" + "=" * 60)
    print(" PASO 5: Construcción y Entrenamiento del Modelo de Regresión Lineal ")
    print("=" * 60)
    
    # Para predecir la nota final G3, utilizaremos las notas previas (G1, G2) y tiempo de estudio
    caracteristicas = ['G1', 'G2', 'studytime', 'failures', 'absences']
    variable_objetivo = 'G3'

    X = data_sin_atipicos[caracteristicas]
    y = data_sin_atipicos[variable_objetivo]

    # División en conjuntos de Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instanciar e integrar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    print("Modelo entrenado exitosamente.")
    print(f"Intercepto ($w_0$): {modelo.intercept_:.4f}")
    print("Coeficientes ($w_i$):")
    for col, coef in zip(caracteristicas, modelo.coef_):
        print(f"  - {col}: {coef:.4f}")

    print("\n" + "=" * 60)
    print(" PASO 6: Evaluación del Modelo ")
    print("=" * 60)
    
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print(f"Coeficiente de Determinación ($R^2$): {r2:.4f}")

    # Gráfica de Valores Reales vs Predichos
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='indigo', alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Notas Reales (G3)")
    plt.ylabel("Notas Predichas (G3)")
    plt.title("Predicción de Regresión Lineal: Reales vs Predichos")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('predicciones_vs_reales.png')
    plt.close()
    print("\nGráfico 'predicciones_vs_reales.png' guardado con éxito.")
    print("\nProceso finalizado.")

if __name__ == "__main__":
    main()