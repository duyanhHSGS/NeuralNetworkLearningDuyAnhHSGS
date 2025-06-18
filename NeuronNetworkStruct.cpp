#include <vector>
#include <cmath>
#include <functional>
#include <iostream>

// === Activation Function (Leaky ReLU for now) ===
double leaky_relu(double x) {
	return x >= 0 ? x : 0.01 * x;
}

double d_leaky_relu(double x) {
	return x >= 0 ? 1.0 : 0.01;
}

// === Deterministic Weight Generator ===
double deterministic_weight(int neuron_index, int input_index) {
	return ((neuron_index + 1) * 10 + input_index) * 0.01; // e.g., 0.11, 0.12, etc.
}

double deterministic_bias(int neuron_index) {
	return -0.05 * (neuron_index + 1); // e.g., -0.05, -0.10, etc.
}

// === Neuron ===
struct Neuron {
	std::vector<double> weights;
	double bias;
	std::function<double(double)> activation;
	std::function<double(double)> activation_derivative;

	Neuron(int num_inputs,
	       std::function<double(double)> act,
	       std::function<double(double)> dact,
	       int neuron_index)
		: activation(act), activation_derivative(dact) {
		for (int i = 0; i < num_inputs; ++i)
			weights.push_back(deterministic_weight(neuron_index, i));
		bias = deterministic_bias(neuron_index);
	}

	double compute_output(const std::vector<double>& inputs, double& z_out) {
		double z = bias;
		for (size_t i = 0; i < weights.size(); ++i)
			z += weights[i] * inputs[i];
		z_out = z;
		return activation(z);
	}

	void train_once(const std::vector<double>& inputs, double error, double learning_rate, double z_cache) {
		double delta = error * activation_derivative(z_cache);
		for (size_t i = 0; i < weights.size(); ++i)
			weights[i] -= learning_rate * delta * inputs[i];
		bias -= learning_rate * delta;
	}
};

// === Layer ===
struct Layer {
	std::vector<Neuron> neurons;

	Layer(int num_neurons, int inputs_per_neuron,
	      std::function<double(double)> act,
	      std::function<double(double)> dact) {
		for (int i = 0; i < num_neurons; ++i)
			neurons.emplace_back(inputs_per_neuron, act, dact, i);
	}

	std::vector<double> forward(const std::vector<double>& inputs,
	                            std::vector<double>& z_caches) {
		std::vector<double> outputs;
		z_caches.clear();
		for (auto& neuron : neurons) {
			double z;
			double output = neuron.compute_output(inputs, z);
			outputs.push_back(output);
			z_caches.push_back(z);
		}
		return outputs;
	}

	void train_once(const std::vector<double>& inputs_to_layer,
	                const std::vector<double>& errors_for_layer,
	                const std::vector<double>& z_values_for_layer,
	                double learning_rate) {
		for (size_t i = 0; i < neurons.size(); ++i) {
			neurons[i].train_once(inputs_to_layer,
			                      errors_for_layer[i],
			                      learning_rate,
			                      z_values_for_layer[i]);
		}
	}
};

// === Network ===
struct NeuralNetwork {
	std::vector<Layer> hidden_layers;
	Layer output_layer;
	std::function<double(double)> activation;
	std::function<double(double)> activation_derivative;

	NeuralNetwork(const std::vector<int>& hidden_layer_sizes,
	              int input_size, int output_size,
	              std::function<double(double)> act,
	              std::function<double(double)> dact)
		: activation(act), activation_derivative(dact),
		  output_layer(output_size,
		               hidden_layer_sizes.empty() ? input_size : hidden_layer_sizes.back(),
		               act, dact) {
		int prev_size = input_size;
		for (int num_neurons : hidden_layer_sizes) {
			hidden_layers.emplace_back(num_neurons, prev_size, act, dact);
			prev_size = num_neurons;
		}
	}

	std::vector<double> feed_forward(const std::vector<double>& input_data,
	                                 std::vector<std::vector<double>>& layer_zs,
	                                 std::vector<std::vector<double>>& layer_activations) {
		layer_activations.clear();
		layer_zs.clear();

		std::vector<double> current_input = input_data;
		layer_activations.push_back(current_input);

		for (Layer& layer : hidden_layers) {
			std::vector<double> z_cache;
			current_input = layer.forward(current_input, z_cache);
			layer_activations.push_back(current_input);
			layer_zs.push_back(z_cache);
		}

		std::vector<double> z_cache_output;
		std::vector<double> output = output_layer.forward(current_input, z_cache_output);
		layer_activations.push_back(output);
		layer_zs.push_back(z_cache_output);

		return output;
	}

	void train(const std::vector<double>& input_data,
	           const std::vector<double>& expected_output,
	           double learning_rate) {
		std::vector<std::vector<double>> zs;
		std::vector<std::vector<double>> activations;
		std::vector<double> output = feed_forward(input_data, zs, activations);

		std::vector<double> output_errors(output.size());
		for (size_t i = 0; i < output.size(); ++i)
			output_errors[i] = output[i] - expected_output[i];

		for (size_t i = 0; i < output_layer.neurons.size(); ++i) {
			output_layer.neurons[i].train_once(
			    activations[activations.size() - 2],
			    output_errors[i],
			    learning_rate,
			    zs.back()[i]
			);
		}

		std::vector<double> next_layer_error = output_errors;

		for (int layer_idx = hidden_layers.size() - 1; layer_idx >= 0; --layer_idx) {
			Layer& current_layer = hidden_layers[layer_idx];
			Layer& next_layer = (layer_idx == (int)hidden_layers.size() - 1)
			                    ? output_layer
			                    : hidden_layers[layer_idx + 1];

			std::vector<double> current_errors(current_layer.neurons.size(), 0.0);
			for (size_t i = 0; i < current_layer.neurons.size(); ++i) {
				double error = 0.0;
				for (size_t j = 0; j < next_layer.neurons.size(); ++j)
					error += next_layer.neurons[j].weights[i] * next_layer_error[j];
				current_errors[i] = error;
			}
			current_layer.train_once(activations[layer_idx],
			                         current_errors,
			                         zs[layer_idx],
			                         learning_rate);
			next_layer_error = current_errors;
		}
	}
};

// === MAIN ===
int main() {
	auto act = leaky_relu;
	auto dact = d_leaky_relu;

	NeuralNetwork brain({4}, 2, 1, act, dact);

	std::vector<std::vector<double>> training_inputs = {
		{0, 0}, {0, 1}, {1, 0}, {1, 1}
	};
	std::vector<std::vector<double>> training_outputs = {
		{0}, {1}, {1}, {0}
	};

	double learning_rate = 0.1;
	int epochs = 1000;

	for (int epoch = 0; epoch < epochs; ++epoch) {
		for (size_t i = 0; i < training_inputs.size(); ++i)
			brain.train(training_inputs[i], training_outputs[i], learning_rate);
	}

	std::cout << "Trained Network Outputs for XOR:\n";
	for (size_t i = 0; i < training_inputs.size(); ++i) {
		std::vector<std::vector<double>> dummy_zs, dummy_activations;
		std::vector<double> prediction = brain.feed_forward(training_inputs[i], dummy_zs, dummy_activations);
		std::cout << "Input: " << training_inputs[i][0] << ", "
		          << training_inputs[i][1]
		          << " ? Output: " << prediction[0] << "\n";
	}

	return 0;
}
