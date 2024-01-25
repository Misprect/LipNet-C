#include <torch/torch.h>

class MyModel : public torch::nn::Module {
public:
    MyModel() {
        conv1 = register_module("conv1", torch::nn::Conv3d(torch::nn::Conv3dOptions(1, 128, 3).padding(1)));
        relu = register_module("relu", torch::nn::ReLU());
        maxpool1 = register_module("maxpool1", torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({1, 2, 2})));

        conv2 = register_module("conv2", torch::nn::Conv3d(torch::nn::Conv3dOptions(128, 256, 3).padding(1)));
        maxpool2 = register_module("maxpool2", torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({1, 2, 2})));

        conv3 = register_module("conv3", torch::nn::Conv3d(torch::nn::Conv3dOptions(256, 75, 3).padding(1)));
        maxpool3 = register_module("maxpool3", torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({1, 2, 2})));

        time_distributed = register_module("time_distributed", torch::nn::TimeDistributed(torch::nn::Flatten()));

        lstm1 = register_module("lstm1", torch::nn::LSTM(torch::nn::LSTMOptions(75, 128).bidirectional(true)));
        dropout1 = register_module("dropout1", torch::nn::Dropout(0.5));

        lstm2 = register_module("lstm2", torch::nn::LSTM(torch::nn::LSTMOptions(256, 128).bidirectional(true)));
        dropout2 = register_module("dropout2", torch::nn::Dropout(0.5));

        dense = register_module("dense", torch::nn::Linear(256, char_to_num.vocabulary_size()+1));
        softmax = register_module("softmax", torch::nn::Softmax(torch::nn::SoftmaxOptions(1)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = maxpool1(relu(conv1(x)));
        x = maxpool2(relu(conv2(x)));
        x = maxpool3(relu(conv3(x)));
        x = time_distributed(x);
        
        x = lstm1(x).output;
        x = dropout1(x);

        x = lstm2(x).output;
        x = dropout2(x);

        x = dense(x);
        x = softmax(x, 1);

        return x;
    }

private:
    torch::nn::Conv3d conv1, conv2, conv3;
    torch::nn::ReLU relu;
    torch::nn::MaxPool3d maxpool1, maxpool2, maxpool3;
    torch::nn::TimeDistributed time_distributed;
    torch::nn::LSTM lstm1, lstm2;
    torch::nn::Dropout dropout1, dropout2;
    torch::nn::Linear dense;
    torch::nn::Softmax softmax;
};

int main() {
    // Assuming char_to_num is defined and initialized in your code.
    MyModel model;
    
    // Example input tensor shape: (batch_size, channels, depth, height, width)
    torch::Tensor input = torch::randn({1, 1, 75, 46, 140});
    
    // Forward pass
    torch::Tensor output = model.forward(input);
    
    // Print the output shape
    std::cout << "Output shape: " << output.sizes() << std::endl;

    return 0;
}
