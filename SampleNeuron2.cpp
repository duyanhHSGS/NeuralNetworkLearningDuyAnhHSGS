#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// -----------------------------------------
// Sigmoid Activation Function + Derivative
// -----------------------------------------
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s); // sigmoid'(z) = s * (1 - s)
}

// -----------------------------------------
// Neuron with Sigmoid Activation
// -----------------------------------------
struct Neuron {
    std::vector<double> weights;
    double bias;

    // Forward pass using sigmoid activation
    double forward(const std::vector<double>& inputs, double& z_out) {
        double z = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            z += weights[i] * inputs[i];
        }
        z += bias;
        z_out = z;
        return sigmoid(z);
    }

    // Training step
    void train(const std::vector<double>& inputs, double target, double learning_rate) {
        double z = 0.0;
        double output = forward(inputs, z);
        double loss = 0.5 * std::pow(output - target, 2);

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  z = " << z << ", a = " << output << ", Loss: " << loss << std::endl;

        // Backpropagation
        double dL_dout = output - target;
        double dout_dz = sigmoid_derivative(z);
        double dL_dz = dL_dout * dout_dz;

        std::cout << "  dL/dOut = " << dL_dout << ", Sigmoid'(z) = " << dout_dz << ", dL/dZ = " << dL_dz << std::endl;

        // Update weights
        for (size_t i = 0; i < weights.size(); ++i) {
            double dL_dw = dL_dz * inputs[i];
            double old_w = weights[i];
            weights[i] -= learning_rate * dL_dw;

            std::cout << "  Weight[" << i << "] changed: " << old_w << " -> " << weights[i]
                      << " (Gradient: " << dL_dw << ")" << std::endl;
        }

        // Update bias
        double old_b = bias;
        bias -= learning_rate * dL_dz;
        std::cout << "  Bias changed: " << old_b << " -> " << bias << std::endl;
    }
};

// -----------------------------------------
// Main Function
// -----------------------------------------
int main() {
    Neuron n = {{0.5, 1.5}, 0.2};  // SAME weights and bias as before
    std::vector<double> input = {1.0, 2.0};
    double target = 0.5;
    double lr = 0.1;

    for (int i = 0; i < 50; ++i) {
        std::cout << "=== Epoch " << i + 1 << " ===" << std::endl;
        n.train(input, target, lr);
        std::cout << std::endl;
    }

    return 0;
}
