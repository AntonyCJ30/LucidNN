#ifndef JSON_HANDLER_H
#define JSON_HANDLER_H

#include <string>
#include <vector>
#include "Layer.h"
#include <utility>


struct TrainingData {
    std::vector<double> inputs;
    std::vector<double> targets;
};

struct NetworkConfig {
    int input_size;
    std::vector<std::pair<int, ActivationType>> hidden_layers;
    std::pair<int, ActivationType> output_layer;

    int epochs;
    double learning_rate;

    TrainingData training_data;   // 🔥 ADD THIS

    nlohmann::json initial_state;
};

class JsonHandler {
public:
    static NetworkConfig parseInitNetwork(const std::string& file_path);
};

#endif