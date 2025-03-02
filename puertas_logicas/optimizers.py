import numpy as np

class Optimizer:
    """Clase base para los optimizadores de la red neuronal."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, weights, gradients):
        """Método base para actualizar los pesos."""
        pass

class SGD(Optimizer):
    """Optimizador de Descenso de Gradiente Estocástico."""
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, weights, gradients):
        """Actualiza los pesos usando SGD."""
        for key in weights:
            weights[key] += self.learning_rate * gradients[key]

class Adam(Optimizer):
    """Optimizador Adam."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        """Actualiza los pesos usando Adam."""
        if self.m is None:
            self.m = {key: np.zeros_like(grad) for key, grad in gradients.items()}
            self.v = {key: np.zeros_like(grad) for key, grad in gradients.items()}
        self.t += 1
        for key in weights:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            weights[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop(Optimizer):
    """Optimizador RMSprop."""
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, weights, gradients):
        """Actualiza los pesos usando RMSprop."""
        if self.cache is None:
            self.cache = {key: np.zeros_like(grad) for key, grad in gradients.items()}
        
        for key in weights:
            self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * (gradients[key] ** 2)
            weights[key] -= self.learning_rate * gradients[key] / (np.sqrt(self.cache[key]) + self.epsilon) 