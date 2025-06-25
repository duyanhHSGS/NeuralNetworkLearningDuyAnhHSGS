#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <omp.h>

// ===================== ACTIVATION FUNCTIONS =====================
struct Activate {
	struct ReLU {
		static inline constexpr double activate(double x) {
			return x >= 0 ? x : 0.0;
		}
		static inline constexpr double derivative(double x) {
			return x >= 0 ? 1.0 : 0.0;
		}
	};

	struct LeakyReLU {
		static inline constexpr double activate(double x) {
			return x >= 0 ? x : 0.01 * x;
		}
		static inline constexpr double derivative(double x) {
			return x >= 0 ? 1.0 : 0.01;
		}
	};

	struct Sigmoid {
		static inline double activate(double x) {
			return 1.0 / (1.0 + std::exp(-x));
		}
		static inline double derivative(double x) {
			double y = activate(x);
			return y * (1.0 - y);
		}
	};

	struct CustomSigmoid {
		static inline constexpr double fast_exp(double x) {
			return 1.0 + x * (1.0 + x * (0.5 + x * 0.166666));
		}
		static inline constexpr double activate(double x) {
			return 1.0 / (1.0 + fast_exp(-x));
		}
		static inline constexpr double derivative(double x) {
			double y = activate(x);
			return y * (1.0 - y);
		}
	};

	struct CustomSigmoid2 {
		static inline constexpr double dumb_exp(double x) {
			return 1.0 + x + 0.5 * x * x;
		}

		static inline constexpr double activate(double x) {
			return 1.0 / (1.0 + dumb_exp(-x));
		}

		static inline constexpr double derivative(double x) {
			double y = activate(x);
			return y * (1.0 - y);
		}
	};


};

// ===================== RANDOM INITIALIZATION =====================
inline double generate_random() {
	static uint32_t seed = 123456789u;
	seed ^= seed << 13;
	seed ^= seed >> 15;
	seed ^= seed << 5;
	return ((seed & 0xFFFFFFu) / 16777215.0) * 2.0 - 1.0;
}

// ===================== NEURON =====================
template <int NumInputs, class Activation>
struct Neuron {
	std::array<double, NumInputs> weights;
	double bias;
	double z_cache = 0;

	inline Neuron() {
		for (auto& w : weights) w = generate_random();
		bias = generate_random();
	}

	inline Neuron(const std::array<double, NumInputs>& init_weights, double init_bias)
		: weights(init_weights), bias(init_bias) {}

	inline double compute_output(const double inputs[NumInputs]) {
		double z = bias;
		for (int i = 0; i < NumInputs; ++i)
			z += weights[i] * inputs[i];
		z_cache = z;
		return Activation::activate(z);
	}

	inline void train_once(const double inputs[NumInputs], double error, double lr) {
		double delta = error * Activation::derivative(z_cache);
		for (int i = 0; i < NumInputs; ++i)
			weights[i] -= lr * delta * inputs[i];
		bias -= lr * delta;
	}
};

// ===================== LAYER =====================
template <int NumInputs, int NumNeurons, class Activation>
struct Layer {
	std::array<Neuron<NumInputs, Activation>, NumNeurons> neurons;
	std::array<double, NumInputs> last_input{};
	std::array<double, NumNeurons> last_output{};

	inline Layer() = default;

	inline Layer(const std::array<std::array<double, NumInputs>, NumNeurons>& init_weights,
	             const std::array<double, NumNeurons>& init_biases) {
		for (int i = 0; i < NumNeurons; ++i)
			neurons[i] = Neuron<NumInputs, Activation>(init_weights[i], init_biases[i]);
	}

	inline std::array<double, NumNeurons> compute_output(const std::array<double, NumInputs>& input) {
		last_input = input;
		for (int i = 0; i < NumNeurons; ++i)
			last_output[i] = neurons[i].compute_output(input.data());
		return last_output;
	}

	inline void train_once(const std::array<double, NumInputs>& input, const std::array<double, NumNeurons>& errors, double lr) {
		for (int i = 0; i < NumNeurons; ++i)
			neurons[i].train_once(input.data(), errors[i], lr);
	}
};

// ===================== METAPROGRAMMING UTILITIES =====================
template <std::size_t I, int First, int... Rest>
struct NthDim : NthDim<I - 1, Rest...> {};

template <int First, int... Rest>
struct NthDim<0, First, Rest...> {
	static constexpr int value = First;
};

template <std::size_t Index, std::size_t Total, typename HiddenAct, typename OutputAct>
using ActivationFor = std::conditional_t<Index == Total - 1, OutputAct, HiddenAct>;

// ===================== NETWORK =====================
template <typename HiddenAct, typename OutputAct, int... Dims>
struct Network {
		static constexpr std::size_t NumLayers = sizeof...(Dims) - 1;
		using Indices = std::make_index_sequence<NumLayers>;

		template <std::size_t I>
		using ThisLayer = Layer<
		                  NthDim<I, Dims...>::value,
		                  NthDim<I + 1, Dims...>::value,
		                  ActivationFor<I, NumLayers, HiddenAct, OutputAct>
		                  >;

		template <std::size_t... Is>
		static auto make_layers(std::index_sequence<Is...>) {
			return std::tuple<ThisLayer<Is>...> {};
		}

		using LayersTuple = decltype(make_layers(Indices{}));
		LayersTuple layers;

		inline Network() = default;

		template <typename... Inits>
		inline Network(Inits&&... inits) {
			static_assert(sizeof...(Inits) == NumLayers);
			init_layers<0>(std::forward<Inits>(inits)...);
		}

		template <std::size_t I = 0, typename Input>
		inline auto compute_forward_impl(const Input& input) {
			auto& layer = std::get<I>(layers);
			auto output = layer.compute_output(input);
			if constexpr (I + 1 < NumLayers)
				return compute_forward_impl<I + 1>(output);
			else
				return output;
		}

		inline auto compute_forward(const std::array<double, NthDim<0, Dims...>::value>& input) {
			return compute_forward_impl<0>(input);
		}

		template <typename TargetArray>
		inline void train_once(const std::array<double, NthDim<0, Dims...>::value>& input,
		                       const TargetArray& expected,
		                       double lr) {
			auto output = compute_forward(input);
			std::array<double, std::tuple_size<TargetArray>::value> output_errors{};
			for (size_t i = 0; i < output.size(); ++i)
				output_errors[i] = output[i] - expected[i];
			backprop<NumLayers - 1>(output_errors, lr);
		}

	private:
		template <std::size_t I, typename First, typename... Rest>
		inline void init_layers(First&& first, Rest&&... rest) {
			std::get<I>(layers) = ThisLayer<I>(std::get<0>(first), std::get<1>(first));
			if constexpr (sizeof...(Rest) > 0)
				init_layers<I + 1>(std::forward<Rest>(rest)...);
		}

		template <std::size_t LayerIndex, typename Errors>
		inline void backprop(const Errors& errors, double lr) {
			auto& layer = std::get<LayerIndex>(layers);
			if constexpr (LayerIndex > 0) {
				constexpr int PrevSize =
				    std::tuple_size<decltype(std::get<LayerIndex - 1>(layers).last_output)>::value;

				std::array<double, PrevSize> prev_errors{};
				for (size_t i = 0; i < errors.size(); ++i) {
					for (size_t j = 0; j < PrevSize; ++j) {
						prev_errors[j] += layer.neurons[i].weights[j] *
						                  errors[i] *
						                  ActivationFor<LayerIndex, NumLayers, HiddenAct, OutputAct>::derivative(layer.neurons[i].z_cache);
					}
				}
				layer.train_once(layer.last_input, errors, lr);
				backprop<LayerIndex - 1>(prev_errors, lr);
			} else {
				layer.train_once(layer.last_input, errors, lr);
			}
		}
};

// ======================= THE MAIN CHARACTER =========================
int main() {
	using Input = std::array<double, 1>;
	using Output = std::array<double, 1>;

	Network<Activate::LeakyReLU, Activate::LeakyReLU, 1, 8, 1> net;

	std::array<Input, 21> inputs = {{
			{-5}, {-4}, {-3}, {-2}, {-1}, {-0.5}, {0.0}, {0.5}, {1}, {1.5},
			{2}, {2.5}, {3}, {3.5}, {4}, {4.5}, {5}, {5.5}, {6}, {7}, {8}
		}
	};
	std::array<Output, 21> expected = {{
			{25}, {16}, {9}, {4}, {1}, {0.25}, {0}, {0.25}, {1}, {2.25},
			{4}, {6.25}, {9}, {12.25}, {16}, {20.25}, {25}, {30.25}, {36}, {49}, {64}
		}
	};


	double lr = 0.01;
	int epochs = 100000;

	std::cout << "Training the square function... please hold\n";
	for (int epoch = 0; epoch < epochs; ++epoch)
		for (int i = 0; i < inputs.size(); ++i)
			net.train_once(inputs[i], expected[i], lr);

	std::cout << "=== Training complete! Let's gooo! ===\n";

	while (true) {
		std::cout << "\nEnter a number to square (or type 'q' to quit): ";
		std::string user_input;
		std::getline(std::cin, user_input);

		if (user_input == "q") break;

		try {
			double x = std::stod(user_input);
			Input input = {x};
			auto output = net.compute_forward(input);
			double predicted = output[0];
			double actual = x * x;
			double err = std::abs(predicted - actual);

			std::cout << "Predicted: " << predicted << "\n";
			std::cout << "Actual:    " << actual << "\n";
			std::cout << "Error:     " << err << "\n";
		} catch (...) {
			std::cout << "Bruh that's not a number. Try again.\n";
		}
	}

	std::cout << "Peace out\n";
}

