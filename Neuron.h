#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double activation;
    double z_value;

    Neuron(int num_inputs);
};

#endif
