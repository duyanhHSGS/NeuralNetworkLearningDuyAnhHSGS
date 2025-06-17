#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

struct Neuron {
    std::vector<double> w;
    double b;
    std::function<double(double)> act;
    std::function<double(double)> dact;

    double forward(const std::vector<double>& x, double& z) {
        z = b;
        for (size_t i = 0; i < x.size(); ++i) z += w[i] * x[i];
        return act(z);
    }

    void train(const std::vector<double>& x, double y, double lr, int epochs) {
        for (int i = 0; i < epochs; ++i) {
            double z, out = forward(x, z);
            double loss = 0.5 * (out - y) * (out - y);
            double dL = (out - y) * dact(z);
            for (size_t j = 0; j < x.size(); ++j) w[j] -= lr * dL * x[j];
            b -= lr * dL;
            std::cout << loss << " ";
        }
        std::cout << "\n";
    }
};

double relu(double x) { return x > 0 ? x : 0; }
double drelu(double x) { return x > 0 ? 1 : 0; }

double lrelu(double x) { return x > 0 ? x : 0.01 * x; }
double dlrelu(double x) { return x > 0 ? 1 : 0.01; }

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double dsigmoid(double x) { double s = sigmoid(x); return s * (1 - s); }

double tanh_act(double x) { return std::tanh(x); }
double dtanh(double x) { double t = tanh_act(x); return 1 - t * t; }

double identity(double x) { return x; }
double didentity(double) { return 1.0; }

double expx(double x) { return std::exp(x); }
double dexpx(double x) { return std::exp(x); }

int main() {
    std::vector<std::string> names = {"ReLU", "Leaky", "Sigmoid", "Tanh", "Identity", "Exp(x)"};
    std::vector<std::function<double(double)>> acts = {relu, lrelu, sigmoid, tanh_act, identity, expx};
    std::vector<std::function<double(double)>> dacts = {drelu, dlrelu, dsigmoid, dtanh, didentity, dexpx};

    for (size_t i = 0; i < acts.size(); ++i) {
        std::cout << names[i] << ": ";
        Neuron n = {{0.5, 1.5}, 0.2, acts[i], dacts[i]};
        n.train({1.0, 2.0}, 1.0, 0.1, 30);
    }
}
