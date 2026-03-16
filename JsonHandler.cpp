#include "JsonHandler.h"
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

ActivationType stringToActivation(const std::string& s) {
    if (s == "relu") return ActivationType::RELU;
    if (s == "sigmoid") return ActivationType::SIGMOID;
    if (s == "tanh") return ActivationType::TANH;
    if (s == "linear") return ActivationType::LINEAR;
    if (s == "leaky relu") return ActivationType::LEAKY_RELU;
    throw std::runtime_error("Unknown activation type: " + s);
}

NetworkConfig JsonHandler::parseInitNetwork(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file)
        throw std::runtime_error("Cannot open JSON file");

    json j;
    file >> j;

    if (j["type"] != "INIT_NETWORK")
        throw std::runtime_error("JSON is not INIT_NETWORK");

    auto net = j["network"];

    NetworkConfig config;
    config.input_size = net["input_size"];

    for (auto& hl : net["hidden_layers"]) {
        int neurons = hl["neurons"];
        ActivationType type = stringToActivation(hl["activation"]);
        config.hidden_layers.push_back({neurons, type});
    }

    int out_neurons = net["output_layer"]["neurons"];
    ActivationType out_type =
        stringToActivation(net["output_layer"]["activation"]);

    config.output_layer = {out_neurons, out_type};

    // --- Extract Hyperparameters ---
    if (j.contains("hyperparameters")) {
        config.epochs = j["hyperparameters"].value("epochs", 100);
        config.learning_rate = j["hyperparameters"].value("learning_rate", 0.01);
    } else {
        config.epochs = 100;
        config.learning_rate = 0.01;
    }
    if (j.contains("initial_state")) {
        config.initial_state = j["initial_state"];
    }
    if (j.contains("training_data")) {
        config.training_data.inputs =
            j["training_data"]["inputs"]
            .get<std::vector<double>>();

        config.training_data.targets =
            j["training_data"]["targets"]
            .get<std::vector<double>>();
    }

    return config;
}

