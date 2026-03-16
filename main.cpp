#include <iostream>
#include "Network.h"
#include "JsonHandler.h"
#include <fstream>
#include <nlohmann/json.hpp>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cout << "Usage: ./app config.json\n";
        return 1;
    }

    try {
        NetworkConfig config =
            JsonHandler::parseInitNetwork(argv[1]);

        Network net;
        net.initNetwork(config.input_size,
                        config.hidden_layers,
                        config.output_layer);
        
        
        // --- ADDED ---
        std::cout << "--- Training Config ---\n";
        std::cout << "Epochs: " << config.epochs << "\n";
        std::cout << "Learning Rate: " << config.learning_rate << "\n";
        std::cout << "-----------------------\n";
        if (!config.initial_state.empty()) {
            std::cout << "Loading specific weights and biases from JSON...\n";
            for (size_t l = 1; l < net.layers.size(); ++l) {
                for (size_t n = 0; n < net.layers[l].neurons.size(); ++n) {
                    std::string key = "L" + std::to_string(l) + "_N" + std::to_string(n);
                    if (config.initial_state.contains(key)) {
                        double b = config.initial_state[key]["bias"];
                        std::vector<double> w = config.initial_state[key]["weights"].get<std::vector<double>>();
                        net.setNeuronParams(l, n, b, w);
                    }
                }
            }
            std::cout << "Successfully synchronized weights with Python UI!\n\n";
        }
        nlohmann::ordered_json results =
            net.trainAndReturnHistory(
                config.training_data.inputs,
                config.training_data.targets,
                config.epochs,
                config.learning_rate
            );

        std::ofstream out("results.json");
        out << results.dump(4);

        net.printStructure();
    }
    catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}