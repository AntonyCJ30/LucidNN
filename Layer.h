#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"
#include "Activation.h"

class Layer {
public:
    std::vector<Neuron> neurons;
    ActivationType activationType;

    Layer(int num_neurons, int inputs_per_neuron, ActivationType type);
};

#endif

