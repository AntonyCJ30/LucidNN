#include "Network.h"
#include <iostream>
#include <stdexcept>


void Network::initNetwork(
    int input_size,
    const std::vector<std::pair<int, ActivationType>>& hidden_layers,
    std::pair<int, ActivationType> output_layer)
{
    layers.clear();

    if (input_size <= 0)
        throw std::runtime_error("Input size must be > 0");

    // Input layer (no weights)
    layers.emplace_back(input_size, 0, ActivationType::LINEAR);

    int previous_size = input_size;

    // Hidden layers
    for (const auto& hl : hidden_layers) {
        int neurons = hl.first;
        ActivationType type = hl.second;

        if (neurons <= 0)
            throw std::runtime_error("Hidden layer must have > 0 neurons");

        layers.emplace_back(neurons, previous_size, type);
        previous_size = neurons;
    }

    // Output layer
    int output_neurons = output_layer.first;
    ActivationType output_type = output_layer.second;

    if (output_neurons <= 0)
        throw std::runtime_error("Output layer must have > 0 neurons");

    layers.emplace_back(output_neurons, previous_size, output_type);
}

void Network::printStructure() const {
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i << " | Neurons: "
                  << layers[i].neurons.size();

        if (i > 0) {
            std::cout << " | Weights per neuron: "
                      << layers[i].neurons[0].weights.size();
        }

        std::cout << std::endl;
    }
}
// --- ADDED IMPLEMENTATION ---
void Network::setNeuronParams(int layer_idx, int neuron_idx, double bias, const std::vector<double>& weights) {
    if (layer_idx < 0 || layer_idx >= layers.size()) return;
    if (neuron_idx < 0 || neuron_idx >= layers[layer_idx].neurons.size()) return;
    layers[layer_idx].neurons[neuron_idx].bias = bias;
    layers[layer_idx].neurons[neuron_idx].weights = weights;
}
std::vector<double> Network::forward(const std::vector<double>& input) {

    // Set input layer activations
    for (size_t i = 0; i < input.size(); ++i) {
        layers[0].neurons[i].activation = input[i];
    }

    // Forward through each layer
    for (size_t l = 1; l < layers.size(); ++l) {

        Layer& current = layers[l];
        Layer& previous = layers[l - 1];

        for (size_t n = 0; n < current.neurons.size(); ++n) {

            Neuron& neuron = current.neurons[n];

            double z = neuron.bias;

            for (size_t w = 0; w < neuron.weights.size(); ++w) {
                z += neuron.weights[w] * previous.neurons[w].activation;
            }

            neuron.z_value = z;
            neuron.activation =
                Activation::activate(z, current.activationType);
        }
    }

    // Collect output layer activations
    std::vector<double> output;
    for (auto& neuron : layers.back().neurons) {
        output.push_back(neuron.activation);
    }

    return output;
}

double Network::computeMSE(const std::vector<double>& output,
                           const std::vector<double>& target) {

    double mse = 0.0;

    for (size_t i = 0; i < output.size(); ++i) {
        double diff = output[i] - target[i];
        mse += diff * diff;
    }

    return mse / output.size();
}

void Network::backward(const std::vector<double>& target,
                       double learning_rate) {

    // Vector to store deltas for each layer
    std::vector<std::vector<double>> deltas(layers.size());

    // --- Output layer ---
    size_t L = layers.size() - 1;

    deltas[L].resize(layers[L].neurons.size());

    for (size_t n = 0; n < layers[L].neurons.size(); ++n) {

        Neuron& neuron = layers[L].neurons[n];

        double error = neuron.activation - target[n];

        double delta = error *
            Activation::derivative(neuron.z_value,
                                   layers[L].activationType);

        deltas[L][n] = delta;
    }

    // --- Hidden layers ---
    for (int l = L - 1; l > 0; --l) {

        deltas[l].resize(layers[l].neurons.size());

        for (size_t i = 0; i < layers[l].neurons.size(); ++i) {

            double sum = 0.0;

            for (size_t j = 0; j < layers[l + 1].neurons.size(); ++j) {
                sum += layers[l + 1].neurons[j].weights[i]
                       * deltas[l + 1][j];
            }

            double delta = sum *
                Activation::derivative(
                    layers[l].neurons[i].z_value,
                    layers[l].activationType);

            deltas[l][i] = delta;
        }
    }

    // --- Update weights and biases ---
    for (size_t l = 1; l < layers.size(); ++l) {

        for (size_t n = 0; n < layers[l].neurons.size(); ++n) {

            Neuron& neuron = layers[l].neurons[n];

            neuron.bias -= learning_rate * deltas[l][n];

            for (size_t w = 0; w < neuron.weights.size(); ++w) {
                neuron.weights[w] -= learning_rate *
                    deltas[l][n] *
                    layers[l - 1].neurons[w].activation;
            }
        }
    }
}

nlohmann::ordered_json Network::trainAndReturnHistory(
    const std::vector<double>& input,
    const std::vector<double>& target,
    int epochs,
    double learning_rate
) {
    // 1. Root object
    nlohmann::ordered_json result;
    result["status"] = "success";
    result["history"] = nlohmann::json::array(); // Standard array is fine here

    for (int e = 1; e <= epochs; ++e) {

        std::vector<double> output = forward(input);
        double error = computeMSE(output, target);
        backward(target, learning_rate);

        // 2. Snapshot object (Ensures epoch -> error -> actual -> expected)
        nlohmann::ordered_json snapshot;
        snapshot["epoch"] = e;
        snapshot["error"] = error;
        snapshot["actual_output"] = output;
        snapshot["expected_output"] = target;

        // 3. Network state object
        nlohmann::ordered_json network_state;

        for (size_t l = 1; l < layers.size(); ++l) {
            for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
                std::string key = "L" + std::to_string(l) + "_N" + std::to_string(n);
                network_state[key]["bias"] = layers[l].neurons[n].bias;
                network_state[key]["weights"] = layers[l].neurons[n].weights;
            }
        }

        snapshot["network_state"] = network_state;
        result["history"].push_back(snapshot);
    }

    return result;
}