import numpy as np
from .common import train_perceptron, predict_perceptron

class LogicGate:
    """Clase base para todas las puertas lógicas."""
    def __init__(self, name):
        self.name = name
        self.weights = None
        self.trained = False
    
    def train(self, X, y, learning_rate=0.1, epochs=10, debug=False):
        """Método para entrenar la puerta lógica."""
        pass
    
    def predict(self, x):
        """Método para realizar predicciones."""
        pass
    
    def get_name(self):
        """Devuelve el nombre de la puerta lógica."""
        return self.name

class PerceptronGate(LogicGate):
    """Clase base para puertas lógicas implementadas con perceptrón simple."""
    def __init__(self, name):
        super().__init__(name)
    
    def train(self, X, y, learning_rate=0.1, epochs=10, debug=False):
        """Entrena el perceptrón para la puerta lógica."""
        self.weights = train_perceptron(X, y, learning_rate, epochs, debug)
        self.trained = True
        return self.weights
    
    def predict(self, x):
        """Realiza una predicción con el perceptrón entrenado."""
        if not self.trained:
            raise Exception(f"La puerta {self.name} no ha sido entrenada aún.")
        return predict_perceptron(x, self.weights)

class AND(PerceptronGate):
    """Implementación de la puerta lógica AND."""
    def __init__(self):
        super().__init__("AND")
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 0, 0, 1])

class OR(PerceptronGate):
    """Implementación de la puerta lógica OR."""
    def __init__(self):
        super().__init__("OR")
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 1, 1, 1])

class NOT(PerceptronGate):
    """Implementación de la puerta lógica NOT."""
    def __init__(self):
        super().__init__("NOT")
        self.X = np.array([[0], [1]])
        self.y = np.array([1, 0])

class NAND(PerceptronGate):
    """Implementación de la puerta lógica NAND (NOT AND)."""
    def __init__(self):
        super().__init__("NAND")
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([1, 1, 1, 0])

class NOR(PerceptronGate):
    """Implementación de la puerta lógica NOR (NOT OR)."""
    def __init__(self):
        super().__init__("NOR")
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([1, 0, 0, 0]) 