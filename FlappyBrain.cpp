#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <conio.h>
#include <windows.h>

// === Activation ===
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
double d_sigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// === Deterministic Weight Gen ===
double deterministic_weight(int neuron_index, int input_index) {
    return 0.5 + 0.01 * neuron_index - 0.005 * input_index;
}
double deterministic_bias(int neuron_index) {
    return 0.1 + 0.01 * neuron_index;
}

struct Neuron {
    std::vector<double> weights;
    double bias;

    Neuron(int num_inputs, int index) {
        for (int i = 0; i < num_inputs; ++i)
            weights.push_back(deterministic_weight(index, i));
        bias = deterministic_bias(index);
    }

    double compute(const std::vector<double>& inputs) {
        double z = bias;
        for (size_t i = 0; i < weights.size(); ++i)
            z += weights[i] * inputs[i];
        return sigmoid(z);
    }
};

struct Layer {
    std::vector<Neuron> neurons;
    Layer(int n_neurons, int inputs_per_neuron) {
        for (int i = 0; i < n_neurons; ++i)
            neurons.emplace_back(inputs_per_neuron, i);
    }

    std::vector<double> forward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        for (auto& neuron : neurons)
            outputs.push_back(neuron.compute(inputs));
        return outputs;
    }
};

struct SimpleNetwork {
    Layer hidden, output;
    SimpleNetwork() : hidden(4, 4), output(1, 4) {}

    double predict(const std::vector<double>& inputs) {
        std::vector<double> hidden_out = hidden.forward(inputs);
        return output.forward(hidden_out)[0];
    }

    void save(const std::string& path) {
        std::ofstream out(path);
        for (const auto& n : hidden.neurons) {
            for (double w : n.weights) out << w << " ";
            out << n.bias << "\n";
        }
        for (const auto& n : output.neurons) {
            for (double w : n.weights) out << w << " ";
            out << n.bias << "\n";
        }
    }

    void load(const std::string& path) {
        std::ifstream in(path);
        for (auto& n : hidden.neurons) {
            for (double& w : n.weights) in >> w;
            in >> n.bias;
        }
        for (auto& n : output.neurons) {
            for (double& w : n.weights) in >> w;
            in >> n.bias;
        }
    }
};

struct Pipe {
    int x, gapY, gapSize;
};

struct Game {
    const int width = 40, height = 20;
    int birdY = height / 2, score = 0;
    bool gameOver = false;
    std::vector<Pipe> pipes;
    SimpleNetwork ai;

    void reset() {
        birdY = height / 2;
        score = 0;
        gameOver = false;
        pipes.clear();
        pipes.push_back({width - 1, rand() % (height - 6) + 3, 5});
    }

    void draw() {
        system("cls");
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool printed = false;
                if (x == 0 || x == width - 1) {
                    std::cout << "#";
                } else if (x == 5 && y == birdY) {
                    std::cout << "O";
                } else {
                    for (const auto& p : pipes)
                        if (x == p.x && (y < p.gapY || y > p.gapY + p.gapSize)) {
                            std::cout << "|"; printed = true; break;
                        }
                    if (!printed) std::cout << " ";
                }
            }
            std::cout << "\n";
        }
        std::cout << "Score: " << score << "\n";
    }

    void update() {
        // AI decision
        Pipe& p = pipes[0];
        double dY = (double)(birdY - (p.gapY + p.gapSize / 2)) / height;
        double dPipeX = (double)(p.x - 5) / width;
        std::vector<double> input = {
            (double)birdY / height,
            (double)p.gapY / height,
            (double)p.x / width,
            dY
        };
        double output = ai.predict(input);
        if (output > 0.5) birdY -= 3;

        // gravity
        birdY += 1;
        if (birdY <= 0 || birdY >= height - 1)
            gameOver = true;

        for (auto& p : pipes) {
            p.x--;
            if (p.x == 5) {
                if (birdY < p.gapY || birdY > p.gapY + p.gapSize) {
                    gameOver = true;
                } else {
                    score += 1;
                }
            }
        }

        if (pipes.size() > 0 && pipes[0].x < 0)
            pipes.erase(pipes.begin());

        if (pipes.empty() || pipes.back().x < width - 15)
            pipes.push_back({width - 1, rand() % (height - 6) + 3, 5});
    }

    void run() {
        reset();
        while (!gameOver) {
            draw();
            update();
            Sleep(100);
        }
        std::cout << "\nGame Over! Final Score: " << score << "\n";
    }
};

int main() {
    srand(time(0));
    Game game;
    game.ai.load("weights.txt"); // optional
    game.run();
    game.ai.save("weights.txt");
    return 0;
}
