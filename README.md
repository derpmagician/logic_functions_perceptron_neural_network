# Implementación de Compuertas Lógicas con Redes Neuronales

Este proyecto implementa un sistema de redes neuronales que simula compuertas lógicas (AND, OR, NOT, XOR) utilizando tanto perceptrones simples como redes neuronales multicapa. Cuenta con una interfaz gráfica interactiva para visualizar los límites de decisión y los resultados del entrenamiento.

## Características

- Implementación de compuertas lógicas básicas:
  - Compuerta AND (usando perceptrón simple)
  - Compuerta OR (usando perceptrón simple)
  - Compuerta NOT (usando perceptrón simple)
  - Compuerta XOR (usando red neuronal multicapa)
- Interfaz gráfica interactiva con Tkinter
- Visualización en tiempo real de los límites de decisión
- Monitoreo del progreso de entrenamiento
- Optimización avanzada usando el optimizador Adam

## Requisitos

- Python 3.x
- NumPy
- Matplotlib
- Tkinter

## Instalación

1. Clona este repositorio o descarga el código fuente
2. Instala las dependencias requeridas:
```bash
pip install numpy matplotlib
```

## Uso

Ejecuta la aplicación principal:
```bash
python logic_functions_perceptron_neural_network.py
```

Aparecerá la interfaz gráfica con botones para cada compuerta lógica. Haz clic en cualquier compuerta para:
- Entrenar la red neuronal correspondiente
- Ver la visualización del límite de decisión
- Ver los resultados de entrada-salida

## Detalles de Implementación

### Perceptrón Simple
- Utilizado para las compuertas AND, OR y NOT
- Arquitectura de una sola capa con función de activación escalonada
- Incluye término de sesgo para mejor posicionamiento del límite de decisión

### Red Neuronal Multicapa (XOR)
- Arquitectura de dos capas con capa oculta
- Función de activación sigmoide
- Optimizador Adam para entrenamiento eficiente
- Retropropagación para actualización de pesos

### Visualización
- Límites de decisión 2D para las compuertas AND, OR y XOR
- Gráfico de función 1D para la compuerta NOT
- Visualización en tiempo real de los resultados del entrenamiento

## Parámetros de Entrenamiento

- Perceptrón Simple:
  - Tasa de aprendizaje: 0.1
  - Épocas: 10

- Red Neuronal XOR:
  - Tasa de aprendizaje: 0.001
  - Tamaño de capa oculta: 2
  - Épocas: 10000
  - Optimizador: Adam (β1=0.9, β2=0.999)

## Estructura del Proyecto

- `logic_functions_perceptron_neural_network.py`: Archivo principal de la aplicación
  - Implementaciones de redes neuronales
  - Interfaz gráfica
  - Funciones de visualización
  - Algoritmos de entrenamiento

## Licencia

Este proyecto es de código abierto y está disponible para fines educativos y de investigación.