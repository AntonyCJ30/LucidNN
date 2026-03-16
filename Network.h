#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "Layer.h"

class Network {
public:
    std::vector<Layer> layers;

    void initNetwork(int input_size,
                     const std::vector<std::pair<int, ActivationType>>& hidden_layers,
                     std::pair<int, ActivationType> output_layer);

    void printStructure() const;

    void setNeuronParams(int layer_idx, int neuron_idx, double bias, const std::vector<double>& weights);
    std::vector<double> forward(const std::vector<double>& input);
    double computeMSE(const std::vector<double>& output, const std::vector<double>& target);
    void backward(const std::vector<double>& target, double learning_rate);
    nlohmann::ordered_json trainAndReturnHistory(
        const std::vector<double>& input,
        const std::vector<double>& target,
        int epochs,
        double learning_rate
    );
};

#endif
