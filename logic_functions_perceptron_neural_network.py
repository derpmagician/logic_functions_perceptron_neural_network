import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# ==========================
# FUNCIONES COMUNES
# ==========================

def sigmoid(z):
    """Función sigmoide para la red neuronal."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivada de la función sigmoide."""
    return sigmoid(z) * (1 - sigmoid(z))

def activation_fn_step(z):
    """Función escalón para el perceptrón simple."""
    return 1 if z >= 0 else 0

# ==========================
# PERCEPTRÓN SIMPLE
# ==========================

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

# ==========================
# RED NEURONAL MULTICAPA (PARA XOR)
# ==========================

def train_neural_network(X, y, hidden_size=2, optimizer=None, epochs=10000, debug=False):
    """
    Entrena una red neuronal para resolver XOR usando un optimizador.
    - X: Datos de entrada.
    - y: Etiquetas esperadas.
    - hidden_size: Número de neuronas en la capa oculta.
    - optimizer: Instancia de un optimizador.
    - epochs: Número de épocas.
    - debug: Si es True, muestra detalles del entrenamiento.
    """
    input_size = X.shape[1]
    output_size = 1
    np.random.seed(42)
    
    # Inicialización de pesos
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    
    # Convertir pesos a un diccionario para compatibilidad con optimizadores
    weights = {
        "input_hidden": weights_input_hidden,
        "hidden_output": weights_hidden_output
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

    return weights["input_hidden"], weights["hidden_output"]

def test_neural_network(X, weights_input_hidden, weights_hidden_output):
    """
    Evalúa la red neuronal después del entrenamiento.
    - X: Datos de entrada.
    - weights_input_hidden: Pesos entre la capa de entrada y la capa oculta.
    - weights_hidden_output: Pesos entre la capa oculta y la capa de salida.
    """
    outputs = []
    for i in range(len(X)):
        hidden_layer_input = np.dot(X[i], weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)
        outputs.append(output_layer_output[0])
    return outputs

# ==========================
# GRÁFICOS DE VISUALIZACIÓN
# ==========================

def plot_decision_boundary(canvas_frame, weights, X, y, title):
    """
    Grafica la línea de decisión para un perceptrón simple.
    - canvas_frame: Frame de tkinter donde se mostrará el gráfico.
    - weights: Pesos del perceptrón (incluyendo el sesgo).
    - X: Datos de entrada.
    - y: Etiquetas esperadas.
    - title: Título del gráfico.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([predict_perceptron(point, weights) for point in grid])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Integrar el gráfico en la interfaz gráfica
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ==========================
# INTERFAZ GRÁFICA
# ==========================

class LogicGateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelo de Funciones Lógicas")
        
        # Variables para almacenar datos
        self.X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_and = np.array([0, 0, 0, 1])
        self.X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_or = np.array([0, 1, 1, 1])
        self.X_not = np.array([[0], [1]])
        self.y_not = np.array([1, 0])
        self.X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_xor = np.array([[0], [1], [1], [0]])
        
        # Interfaz
        self.create_widgets()

    def create_widgets(self):
        # Etiqueta principal
        tk.Label(self.root, text="Seleccione una función lógica:", font=("Arial", 14)).pack(pady=10)
        
        # Botones para seleccionar funciones
        functions = ["AND", "OR", "NOT", "XOR"]
        for func in functions:
            btn = ttk.Button(self.root, text=func, command=lambda f=func: self.train_and_test(f))
            btn.pack(fill="x", padx=20, pady=5)
        
        # Área para mostrar gráficos
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)

    def train_and_test(self, function_name):
        # Limpiar el área de gráficos
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Entrenar y evaluar el modelo
        if function_name == "XOR":
            optimizer = Adam(learning_rate=0.001)
            weights_input_hidden, weights_hidden_output = train_neural_network(
                getattr(self, f"X_{function_name.lower()}"),
                getattr(self, f"y_{function_name.lower()}"),
                optimizer=optimizer,
                epochs=10000,
                debug=False
            )
            outputs = test_neural_network(
                getattr(self, f"X_{function_name.lower()}"),
                weights_input_hidden,
                weights_hidden_output
            )
        else:
            weights = train_perceptron(
                getattr(self, f"X_{function_name.lower()}"),
                getattr(self, f"y_{function_name.lower()}"),
                debug=False
            )
            outputs = [
                predict_perceptron(x, weights)
                for x in getattr(self, f"X_{function_name.lower()}")
            ]
            # Mostrar gráfico de decisión
            if function_name == "NOT":
                # Para NOT, crear un gráfico 1D
                fig, ax = plt.subplots(figsize=(5, 4))
                x = np.linspace(-0.5, 1.5, 100)
                y = [predict_perceptron(np.array([xi]), weights) for xi in x]
                ax.plot(x, y, 'b-', label='Función NOT')
                ax.scatter([0, 1], [1, 0], c='r', s=100, label='Puntos de entrenamiento')
                ax.set_title(f"Función {function_name}")
                ax.set_xlabel("Entrada")
                ax.set_ylabel("Salida")
                ax.grid(True)
                ax.legend()
                
                # Integrar el gráfico en la interfaz gráfica
                canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                plot_decision_boundary(
                    self.canvas_frame,
                    weights,
                    getattr(self, f"X_{function_name.lower()}"),
                    getattr(self, f"y_{function_name.lower()}"),
                    f"Línea de decisión para {function_name}"
                )

        # Mostrar resultados
        result_text = f"Resultados para {function_name}:\n"
        inputs = getattr(self, f"X_{function_name.lower()}")
        labels = getattr(self, f"y_{function_name.lower()}")
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            result_text += f"Entrada: {inp} -> Salida: {out:.4f} (Esperado: {labels[i]})\n"
        tk.Label(self.canvas_frame, text=result_text, font=("Arial", 12), justify="left").pack()

# ==========================
# OPTIMIZADORES
# ==========================

class Optimizer:
    """Clase base para los optimizadores de la red neuronal."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

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

# ==========================
# EJECUCIÓN DE LA INTERFAZ
# ==========================

if __name__ == "__main__":
    root = tk.Tk()
    app = LogicGateApp(root)
    root.mainloop()