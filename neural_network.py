import math
import random

class NeuralNetwork:
    """Neural network implementation."""

    def __init__(self, num_inputs=4, num_hidden=8, learning_rate=0.1, seed=1):
        self._num_inputs = num_inputs
        self._num_hidden= num_hidden
        self._learning_rate = learning_rate
        random.seed(seed)  # Startvärde för slumpgeneratorn

        # Hidden layer weights: (num_hidden x num_inputs)
        self.hidden_weights = []
        for _ in range(num_hidden):
            neuron_weights = []
            for _ in range(num_inputs):
                neuron_weights.append(random.uniform(-1, 1))  # vikt för input
            self.hidden_weights.append(neuron_weights)

        # Hidden layer bias: (num_hidden)
        self.hidden_bias = [random.uniform(-1, 1) for _ in range(num_hidden)]

        # Output neuron weights: (num_hidden) + bias: (1)
        self.output_weights = [random.uniform(-1, 1) for _ in range(num_hidden)]
        self.output_bias = random.uniform(-1, 1)

    def predict_one(self, inputs):
        """Gör en prediktion för en enda input-vektor (returnerar sannolikhet 0–1)."""
        _, output = self._forward(inputs)
        return output

    def predict_label(self, inputs, threshold=0.5):
        """Gör en prediktion och returnerar en binär label baserat på tröskelvärde."""
        output = self.predict_one(inputs)
        return 1 if output >= threshold else 0

    def train(self, inputs, targets, epochs=6000, print_every=500):
        """Tränar nätverket med backpropagation (binär output med sigmoid)."""
        for epoch in range(epochs):
            total_abs_error = 0.0

            for x, target in zip(inputs, targets):
                # ----- FORWARD -----
                hidden_outputs, output = self._forward(x)

                # ----- ERROR -----
                error = target - output
                total_abs_error += abs(error)

                # ----- BACKPROP: output neuron -----
                output_delta = error * self._sigmoid_derivative(output)

                # spara gamla output_weights (viktigt)
                old_output_weights = self.output_weights[:]

                # update output weights + bias
                for h in range(self._num_hidden):
                    self.output_weights[h] += self._learning_rate * output_delta * hidden_outputs[h]
                self.output_bias += self._learning_rate * output_delta

                # ----- BACKPROP: hidden layer -----
                hidden_deltas = []
                for h in range(self._num_hidden):
                    hidden_error = output_delta * old_output_weights[h]
                    hidden_delta = hidden_error * self._sigmoid_derivative(hidden_outputs[h])
                    hidden_deltas.append(hidden_delta)

                # update hidden weights + bias
                for h in range(self._num_hidden):
                    for i_in in range(self._num_inputs):
                        self.hidden_weights[h][i_in] += self._learning_rate * hidden_deltas[h] * x[i_in]
                    self.hidden_bias[h] += self._learning_rate * hidden_deltas[h]

            if print_every and epoch % print_every == 0:
                avg_err = total_abs_error / len(inputs)
                print(f"Epoch {epoch} | avg_error={avg_err:.4f}")
    
    def _sigmoid(self, x):
        """Trycker ihop ett värde till intervallet 0–1."""
        return 1 / (1 + math.exp(-x))

    def _sigmoid_derivative(self, sigmoid_output):
        """Derivata av sigmoid-funktionen (tar sigmoid(x) som input)."""
        return sigmoid_output * (1 - sigmoid_output)

    def _dot_product(self, values, weights):
        """Beräknar skalärprodukt mellan values och weights."""
       
        return sum(v * w for v, w in zip(values, weights))

    def _forward(self, inputs):
        """Kör ett forward pass: inputs → hidden layer → output."""
        hidden_outputs = []

        # Hidden layer
        for h in range(self._num_hidden):
            s = self._dot_product(inputs, self.hidden_weights[h]) + self.hidden_bias[h]
            hidden_outputs.append(self._sigmoid(s))

        # Output layer
        out_sum = self._dot_product(hidden_outputs, self.output_weights) + self.output_bias
        output = self._sigmoid(out_sum)

        return hidden_outputs, output
