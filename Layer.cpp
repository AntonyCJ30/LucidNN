#include "Layer.h"

Layer::Layer(int num_neurons, int inputs_per_neuron, ActivationType type)
    : activationType(type)
{
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(inputs_per_neuron);
    }
}
