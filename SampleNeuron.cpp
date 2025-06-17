#include <iostream>
#include <vector>
#include <cmath>  // For std::fmax (ReLU) and std::pow (squared error)

// ---------------------------------------------------
// ReLU Activation Function
// ---------------------------------------------------
// ReLU(x) = max(0, x)
// Example:
//   relu(3.5) => 3.5
//   relu(-2.1) => 0
double relu(double x) {
    return std::fmax(0.0, x);
}

// ---------------------------------------------------
// Derivative of ReLU (used in backprop)
// ---------------------------------------------------
// ReLU'(x) = 1 if x > 0, else 0
// Example:
//   relu_derivative(2.0) => 1.0
//   relu_derivative(-5.0) => 0.0
double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// ---------------------------------------------------
// Neuron Struct - A single perceptron unit
// ---------------------------------------------------
// This neuron takes multiple inputs and learns to 
// adjust weights and bias to reduce the error.
//
// Mathematically:
//   z = w1*x1 + w2*x2 + ... + wn*xn + bias
//   a = activation(z)
struct Neuron {
    std::vector<double> weights;  // one weight per input
    double bias;                  // additive constant (also trained)

    // -------------------------------------------
    // Forward Pass
    // -------------------------------------------
    // Inputs:
    //   inputs = [x1, x2, ..., xn]
    //
    // Computes:
    //   z = S(wi * xi) + bias
    //   a = ReLU(z)
    //
    // Example:
    //   weights = [0.5, -1.5], bias = 0.2
    //   inputs = [1.0, 2.0]
    //   => z = 0.5*1.0 + (-1.5)*2.0 + 0.2 = -2.3
    //   => ReLU(-2.3) = 0
    double forward(const std::vector<double>& inputs, double& z_out) {
        double z = 0.0;

        // Weighted sum of inputs
        for (size_t i = 0; i < weights.size(); ++i) {
            z += weights[i] * inputs[i];
        }

        z += bias;     // add bias term
        z_out = z;     // output raw z for use in training
        return relu(z); // apply activation function
    }

    // -------------------------------------------------------
    // Training (1 sample): Gradient Descent Backprop Step
    // -------------------------------------------------------
    // Updates the neuron's weights and bias to reduce error
    //
    // Loss function:
    //   L = 0.5 * (output - target)^2   ? Mean Squared Error
    //
    // Backpropagation:
    //   dL/dOutput = output - target
    //   dOutput/dZ = ReLU'(z)
    //   dZ/dW = input[i]
    //
    // Final:
    //   weight[i] -= learning_rate * (dL/dOutput * ReLU'(z) * input[i])
    //   bias      -= learning_rate * (dL/dOutput * ReLU'(z))
    void train(const std::vector<double>& inputs, double target, double learning_rate) {
        double z = 0.0;                         // raw sum (before activation)
        double output = forward(inputs, z);     // run forward pass

        // --------------------------------------------------
        // Calculate current loss:
        // Example:
        //   target = 1.0, output = 0.2
        //   loss = 0.5 * (0.2 - 1.0)^2 = 0.5 * 0.64 = 0.32
        // --------------------------------------------------
        double loss = 0.5 * std::pow(output - target, 2);
        std::cout << "Loss: " << loss << std::endl;

        // Gradient of loss w.r.t. neuron output
        double dL_dout = output - target;

        // Gradient of activation (ReLU derivative)
        double dout_dz = relu_derivative(z);

        // Gradient of loss w.r.t. z (pre-activation)
        double dL_dz = dL_dout * dout_dz;

        // ---------------------------
        // Update Weights
        // ---------------------------
        for (size_t i = 0; i < weights.size(); ++i) {
            // Example:
            //   inputs[i] = 1.0
            //   dL_dz = -0.6
            //   learning_rate = 0.1
            //   delta_w = -0.6 * 1.0 = -0.6
            //   new_w = old_w - 0.1 * -0.6 = old_w + 0.06
            double dL_dw = dL_dz * inputs[i];
            weights[i] -= learning_rate * dL_dw;
        }

        // ---------------------------
        // Update Bias
        // ---------------------------
        // Gradient of loss w.r.t. bias is same as dL/dz
        bias -= learning_rate * dL_dz;
    }
};

// -----------------------------------------------------------
// MAIN PROGRAM
// -----------------------------------------------------------
// Creates a single neuron, feeds it input, and trains it
// repeatedly to learn a target output.
int main() {
    // Create a neuron with:
    //   weights = [0.5, -1.5]
    //   bias = 0.2
    //
    // Interpretation:
    //   input 1 has positive influence
    //   input 2 has strong negative influence
    Neuron n = {{0.5, 1.5}, 0.2};

    // Input vector:
    //   e.g., [height, weight] or [x, y]
    std::vector<double> input = {1.0, 2.0};

    // Desired output:
    //   e.g., target classification or value to learn
    double target = 1.0;

    // Learning rate:
    //   Controls how aggressively weights are updated
    double lr = 0.1;

    // Run 10 training steps
    for (int i = 0; i < 100; ++i) {
        std::cout << "Step " << i + 1 << ": ";
        n.train(input, target, lr);
    }

    return 0;
}
