import numpy as np
from .common import sigmoid, sigmoid_derivative

class NeuralNetworkGate:
    """Clase base para puertas lógicas implementadas con redes neuronales multicapa."""
    def __init__(self, name, hidden_size=2):
        self.name = name
        self.hidden_size = hidden_size
        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.trained = False
    
    def train(self, X, y, optimizer, epochs=10000, debug=False):
        """
        Entrena la red neuronal para la puerta lógica.
        - X: Datos de entrada.
        - y: Etiquetas esperadas.
        - optimizer: Instancia de un optimizador.
        - epochs: Número de épocas.
        - debug: Si es True, muestra detalles del entrenamiento.
        """
        input_size = X.shape[1]
        output_size = 1
        np.random.seed(42)
        
        # Inicialización de pesos
        self.weights_input_hidden = np.random.rand(input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, output_size)
        
        # Convertir pesos a un diccionario para compatibilidad con optimizadores
        weights = {
            "input_hidden": self.weights_input_hidden,
            "hidden_output": self.weights_hidden_output
        }

        for epoch in range(epochs):
            # Propagación hacia adelante
            hidden_layer_input = np.dot(X, weights["input_hidden"])
            hidden_layer_output = sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, weights["hidden_output"])
            output_layer_output = sigmoid(output_layer_input)

            # Retropropagación
            output_error = y - output_layer_output
            output_delta = output_error * sigmoid_derivative(output_layer_input)
            hidden_error = output_delta.dot(weights["hidden_output"].T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_input)

            # Calcular gradientes
            gradients = {
                "input_hidden": X.T.dot(hidden_delta),
                "hidden_output": hidden_layer_output.T.dot(output_delta)
            }

            # Actualizar pesos usando el optimizador
            optimizer.update(weights, gradients)

            # Mostrar detalles si debug está activado
            if debug and epoch % 1000 == 0:
                error = np.mean(np.abs(output_error))
                print(f"Época {epoch}, Error: {error:.4f}")

        self.weights_input_hidden = weights["input_hidden"]
        self.weights_hidden_output = weights["hidden_output"]
        self.trained = True
        return self.weights_input_hidden, self.weights_hidden_output
    
    def predict(self, x):
        """
        Realiza una predicción con la red neuronal entrenada.
        - x: Entrada.
        """
        if not self.trained:
            raise Exception(f"La puerta {self.name} no ha sido entrenada aún.")
        
        # Propagación hacia adelante
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)
        
        return output_layer_output[0]
    
    def get_name(self):
        """Devuelve el nombre de la puerta lógica."""
        return self.name

class XOR(NeuralNetworkGate):
    """Implementación de la puerta lógica XOR."""
    def __init__(self, hidden_size=2):
        super().__init__("XOR", hidden_size)
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

class XNOR(NeuralNetworkGate):
    """Implementación de la puerta lógica XNOR (NOT XOR)."""
    def __init__(self, hidden_size=2):
        super().__init__("XNOR", hidden_size)
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[1], [0], [0], [1]]) 