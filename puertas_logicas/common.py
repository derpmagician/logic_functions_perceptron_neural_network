import numpy as np

def sigmoid(z):
    """Función sigmoide para la red neuronal."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivada de la función sigmoide."""
    return sigmoid(z) * (1 - sigmoid(z))

def activation_fn_step(z):
    """Función escalón para el perceptrón simple."""
    return 1 if z >= 0 else 0

def predict_perceptron(x, weights):
    """
    Realiza una predicción con el perceptrón simple.
    - x: Entrada (vector sin sesgo).
    - weights: Pesos (incluyendo el sesgo).
    """
    x = np.insert(x, 0, 1)  # Agregar el término de sesgo (bias)
    z = np.dot(weights, x)  # Suma ponderada
    return activation_fn_step(z)

def train_perceptron(X, y, learning_rate=0.1, epochs=10, debug=False):
    """
    Entrena un perceptrón simple.
    - X: Datos de entrada.
    - y: Etiquetas esperadas.
    - learning_rate: Tasa de aprendizaje.
    - epochs: Número de épocas.
    - debug: Si es True, muestra detalles del entrenamiento.
    """
    weights = np.zeros(len(X[0]) + 1)  # Inicializar pesos (incluyendo bias)
    for epoch in range(epochs):
        total_error = 0
        for i in range(y.shape[0]):
            prediction = predict_perceptron(X[i], weights)
            error = y[i] - prediction
            total_error += abs(error)
            weights += learning_rate * error * np.insert(X[i], 0, 1)
        if debug:
            print(f"Época {epoch + 1}, Error total: {total_error}")
    return weights 