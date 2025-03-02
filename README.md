# Implementación de Compuertas Lógicas con Redes Neuronales

Este proyecto implementa un sistema de redes neuronales que simula compuertas lógicas utilizando tanto perceptrones simples como redes neuronales multicapa. Cuenta con una interfaz gráfica interactiva para visualizar los límites de decisión y los resultados del entrenamiento.

## Tabla de Contenidos
- [Características](#características)
- [Compuertas Lógicas Implementadas](#compuertas-lógicas-implementadas)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Detalles de Implementación](#detalles-de-implementación)
- [Parámetros de Entrenamiento](#parámetros-de-entrenamiento)
- [Optimizadores Disponibles](#optimizadores-disponibles)
- [Visualizaciones](#visualizaciones)
- [Extensibilidad](#extensibilidad)
- [Licencia](#licencia)

## Características

- Implementación de 7 compuertas lógicas fundamentales
- Arquitectura modular y orientada a objetos
- Múltiples optimizadores para entrenamiento de redes neuronales
- Interfaz gráfica interactiva con Tkinter
- Visualización en tiempo real de los límites de decisión
- Monitoreo del progreso de entrenamiento
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

## Estructura del Proyecto

```
neurona/
├── puertas_logicas/
│   ├── __init__.py           # Hace que la carpeta sea un paquete Python
│   ├── common.py             # Funciones comunes para todas las puertas lógicas
│   ├── perceptron_gates.py   # Implementación de puertas basadas en perceptrón simple
│   ├── neural_network_gates.py # Implementación de puertas basadas en redes neuronales
│   └── optimizers.py         # Implementación de optimizadores
└── logic_functions_perceptron_neural_network.py  # Archivo principal con la interfaz gráfica
```

### Descripción de los Módulos

- **common.py**: Contiene funciones de activación y utilidades compartidas por todas las puertas lógicas.
- **perceptron_gates.py**: Implementa la clase base `PerceptronGate` y las clases específicas para cada puerta lógica basada en perceptrón simple.
- **neural_network_gates.py**: Implementa la clase base `NeuralNetworkGate` y las clases específicas para puertas lógicas que requieren redes multicapa.
- **optimizers.py**: Contiene diferentes algoritmos de optimización para el entrenamiento de redes neuronales.

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

### Arquitectura Orientada a Objetos

El proyecto utiliza una jerarquía de clases para representar las puertas lógicas:

```
LogicGate (clase base abstracta)
├── PerceptronGate (para problemas linealmente separables)
│   ├── AND
│   ├── OR
│   ├── NOT
│   ├── NAND
│   └── NOR
└── NeuralNetworkGate (para problemas no linealmente separables)
    ├── XOR
    └── XNOR
```

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
  - Optimizadores avanzados para convergencia más rápida y estable

## Parámetros de Entrenamiento

### Perceptrón Simple
- **Tasa de aprendizaje**: 0.1
- **Épocas**: 10
- **Inicialización de pesos**: Ceros

### Red Neuronal Multicapa
- **Tasa de aprendizaje**: 0.001
- **Tamaño de capa oculta**: 2
- **Épocas**: 10000
- **Optimizador**: Adam (por defecto)
- **Inicialización de pesos**: Aleatoria con semilla fija

## Optimizadores Disponibles

El proyecto implementa varios optimizadores para el entrenamiento de redes neuronales:

1. **SGD (Stochastic Gradient Descent)**
   - Implementación básica del descenso de gradiente estocástico
   - Parámetros: tasa de aprendizaje

2. **Adam**
   - Optimizador adaptativo que combina las ventajas de RMSprop y Momentum
   - Parámetros: tasa de aprendizaje, beta1, beta2, epsilon
   - Valores por defecto: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8

3. **RMSprop**
   - Optimizador que adapta la tasa de aprendizaje para cada parámetro
   - Parámetros: tasa de aprendizaje, decay_rate, epsilon
   - Valores por defecto: learning_rate=0.001, decay_rate=0.9, epsilon=1e-8

## Visualizaciones

El proyecto genera visualizaciones interactivas que muestran:

1. **Límites de decisión**: Representación gráfica de cómo la red neuronal separa el espacio de entrada
2. **Puntos de entrenamiento**: Visualización de los datos de entrenamiento y sus etiquetas
3. **Resultados numéricos**: Tabla con los valores de entrada, salida predicha y valor esperado

## Extensibilidad

El diseño modular y orientado a objetos del proyecto facilita su extensión:

1. **Añadir nuevas puertas lógicas**:
   - Para problemas linealmente separables: Crear una nueva clase que herede de `PerceptronGate`
   - Para problemas no linealmente separables: Crear una nueva clase que herede de `NeuralNetworkGate`

2. **Implementar nuevos optimizadores**:
   - Crear una nueva clase que herede de la clase base `Optimizer`
   - Implementar el método `update(weights, gradients)`

3. **Modificar la arquitectura de la red**:
   - Ajustar el parámetro `hidden_size` en las clases de redes neuronales
   - Extender la clase `NeuralNetworkGate` para soportar más capas ocultas

## Licencia

Este proyecto es de código abierto y está disponible para fines educativos y de investigación.