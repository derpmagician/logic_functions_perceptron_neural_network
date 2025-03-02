# Implementación de Compuertas Lógicas con Redes Neuronales

Este proyecto implementa un sistema de redes neuronales que simula compuertas lógicas utilizando tanto perceptrones simples como redes neuronales multicapa. Cuenta con una interfaz gráfica interactiva para visualizar los límites de decisión y los resultados del entrenamiento.

## Tabla de Contenidos
- [Características](#características)
- [Compuertas Lógicas Implementadas](#compuertas-lógicas-implementadas)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Detalles de Implementación](#detalles-de-implementación)
- [Parámetros de Entrenamiento](#parámetros-de-entrenamiento)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Visualizaciones](#visualizaciones)
- [Licencia](#licencia)

## Características

- Implementación de 7 compuertas lógicas fundamentales
- Interfaz gráfica interactiva con Tkinter
- Visualización en tiempo real de los límites de decisión
- Monitoreo del progreso de entrenamiento
- Optimización avanzada usando el optimizador Adam
- Visualización de resultados y predicciones

## Compuertas Lógicas Implementadas

### Implementadas con Perceptrón Simple (problemas linealmente separables)
| Compuerta | Descripción | Tabla de Verdad |
|-----------|-------------|-----------------|
| **AND** | Devuelve 1 solo si ambas entradas son 1 | (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1 |
| **OR** | Devuelve 1 si al menos una entrada es 1 | (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→1 |
| **NOT** | Invierte la entrada | (0)→1, (1)→0 |
| **NAND** | Negación de AND | (0,0)→1, (0,1)→1, (1,0)→1, (1,1)→0 |
| **NOR** | Negación de OR | (0,0)→1, (0,1)→0, (1,0)→0, (1,1)→0 |

### Implementadas con Red Neuronal Multicapa (problemas no linealmente separables)
| Compuerta | Descripción | Tabla de Verdad |
|-----------|-------------|-----------------|
| **XOR** | Devuelve 1 si las entradas son diferentes | (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 |
| **XNOR** | Negación de XOR | (0,0)→1, (0,1)→0, (1,0)→0, (1,1)→1 |

## Requisitos

- Python 3.x
- NumPy
- Matplotlib
- Tkinter (incluido en la mayoría de instalaciones de Python)

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

Aparecerá la interfaz gráfica con botones para cada compuerta lógica organizados en dos filas:
- **Primera fila**: AND, OR, NOT, XOR (compuertas básicas)
- **Segunda fila**: NAND, NOR, XNOR (compuertas adicionales)

Al hacer clic en cualquier compuerta:
1. Se entrena la red neuronal correspondiente
2. Se visualiza el límite de decisión (o función para NOT)
3. Se muestran los resultados de entrada-salida con los valores esperados

## Detalles de Implementación

### Perceptrón Simple
- **Arquitectura**: Una sola capa con función de activación escalonada
- **Aplicación**: Compuertas AND, OR, NOT, NAND y NOR
- **Características**:
  - Incluye término de sesgo (bias) para mejor posicionamiento del límite de decisión
  - Algoritmo de entrenamiento basado en la regla del perceptrón
  - Convergencia garantizada para problemas linealmente separables

### Red Neuronal Multicapa
- **Arquitectura**: Dos capas con capa oculta de 2 neuronas
- **Aplicación**: Compuertas XOR y XNOR
- **Características**:
  - Función de activación sigmoide
  - Algoritmo de retropropagación (backpropagation)
  - Optimizador Adam para convergencia más rápida y estable

### Visualización
- **Compuertas 2D**: Límites de decisión para AND, OR, XOR, NAND, NOR y XNOR
- **Compuerta NOT**: Gráfico de función 1D
- **Resultados**: Tabla de valores de entrada, salida y valores esperados

## Parámetros de Entrenamiento

### Perceptrón Simple
- **Tasa de aprendizaje**: 0.1
- **Épocas**: 10
- **Inicialización de pesos**: Ceros

### Red Neuronal Multicapa
- **Tasa de aprendizaje**: 0.001
- **Tamaño de capa oculta**: 2
- **Épocas**: 10000
- **Optimizador**: Adam (β1=0.9, β2=0.999, ε=1e-8)
- **Inicialización de pesos**: Aleatoria con semilla fija

## Estructura del Proyecto

- `logic_functions_perceptron_neural_network.py`: Archivo principal que contiene:
  - Implementaciones de redes neuronales
  - Interfaz gráfica con Tkinter
  - Funciones de visualización con Matplotlib
  - Algoritmos de entrenamiento y optimización
- `README.md`: Documentación del proyecto

## Visualizaciones

El proyecto genera visualizaciones interactivas que muestran:

1. **Límites de decisión**: Representación gráfica de cómo la red neuronal separa el espacio de entrada
2. **Puntos de entrenamiento**: Visualización de los datos de entrenamiento y sus etiquetas
3. **Resultados numéricos**: Tabla con los valores de entrada, salida predicha y valor esperado

## Licencia

Este proyecto es de código abierto y está disponible para fines educativos y de investigación.