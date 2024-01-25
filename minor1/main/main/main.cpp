
//#include <main.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/optim/adam.h>

using namespace std;

vector<char> vocab = { ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\'', '!', '?', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '@'};

template <typename T>
void print(const std::vector<T>& vec) {
    for (const auto& element : vec) {
        std::cout << element << std::endl;
    }
}

class MyModel : public torch::nn::Module {
public:
    MyModel(int64_t vocabulary_size)
        : conv1(torch::nn::Conv3dOptions(1, 128, /*kernel_size=*/3).padding(1)),
        relu1(torch::nn::ReLU()),
        maxpool1(torch::nn::MaxPool3dOptions({ 1, 2, 2 })),
        conv2(torch::nn::Conv3dOptions(128, 256, /*kernel_size=*/3).padding(1)),
        relu2(torch::nn::ReLU()),
        maxpool2(torch::nn::MaxPool3dOptions({ 1, 2, 2 })),
        conv3(torch::nn::Conv3dOptions(256, 75, /*kernel_size=*/3).padding(1)),
        relu3(torch::nn::ReLU()),
        maxpool3(torch::nn::MaxPool3dOptions({ 1, 2, 2 })),
        flatten(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2))),
        bidirectional_lstm1(torch::nn::LSTMOptions(6375, 128).bidirectional(true).batch_first(true)),
        dropout1(torch::nn::Dropout(0.5)),
        bidirectional_lstm2(torch::nn::LSTMOptions(256, 128).bidirectional(true).batch_first(true)),
        dropout2(torch::nn::Dropout(0.5)),
        dense(torch::nn::Linear(256, vocabulary_size)) {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1(x);
        x = relu1(x);
        x = maxpool1(x);
        x = conv2(x);
        x = relu2(x);
        x = maxpool2(x);
        x = conv3(x);
        x = relu3(x);
        x = maxpool3(x);
        x = x.permute({ 0, 2, 3, 4, 1 });
        x = flatten(x);
        x = std::get<0>(bidirectional_lstm1(x));
        x = dropout1(x);
        x = std::get<0>(bidirectional_lstm2(x));
        x = dropout2(x);
        x = torch::softmax(dense(x), /*dim=*/1);
        return x;
    }

private:
    torch::nn::Conv3d conv1, conv2, conv3;
    torch::nn::ReLU relu1, relu2, relu3;
    torch::nn::MaxPool3d maxpool1, maxpool2, maxpool3;
    torch::nn::Flatten flatten;
    torch::nn::LSTM bidirectional_lstm1, bidirectional_lstm2;
    torch::nn::Dropout dropout1, dropout2;
    torch::nn::Linear dense;
};

vector<std::string> readLinesFromFile(const std::string& filePath);
torch::Tensor load_video(const std::string& path);
std::string load_alignments(std::string path);
vector<std::string> split(std::string line, char del);
void normalizeFrame(cv::Mat& frame, const cv::Scalar& mean, const cv::Scalar& stddev);
std::string vectorToString(const std::vector<int>& indices, const std::vector<char>& vocab);
std::vector<int> stringToVector(const std::string& input, const std::vector<char>& vocab);


torch::Tensor ctc_loss(torch::Tensor logits, std::vector<int>& targets, std::vector<int>& input_lengths, std::vector<int>& target_lengths) {
    torch::Tensor input_lengths_tensor = torch::tensor(input_lengths);
    torch::Tensor target_lengths_tensor = torch::tensor(target_lengths);

    torch::Tensor loss = torch::nn::functional::ctc_loss(logits.log_softmax(2), torch::tensor(targets), input_lengths_tensor, target_lengths_tensor);
    return loss;
}




int main() {
    std::string path;
    const int dim1 = 50;
    const int dim2 = 75;
    const int dim3 = 46;
    const int dim4 = 140;
    const int dim5 = 1;

    std::ifstream file("W:/Projects/Minor Project 1/minor1/main/main/data50.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    std::streampos startPos = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos endPos = file.tellg();
    std::streamsize size = endPos - startPos;
    file.seekg(0, std::ios::beg);

    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();
    torch::Tensor tensor = torch::from_blob(data.data(), { dim1, dim2, dim3, dim4, dim5 }, torch::kFloat);

    MyModel model(41);
    model.to(torch::kCUDA);

    torch::Tensor tensor1 = tensor.index({ torch::indexing::Slice(0, 1), torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice() });
    tensor1 = tensor1.permute({ 0, 4, 1, 2, 3 });

    torch::Tensor output = model.forward(tensor1);

    auto output_tensor = output.squeeze().argmax(1).to(torch::kInt32);
    std::vector<int> output_vector(output_tensor.data_ptr<int>(), output_tensor.data_ptr<int>() + output_tensor.numel());

    std::string op = vectorToString(output_vector, vocab);
    std::cout << op << std::endl;
    std::cout << std::endl;


    std::vector<std::string> alignmentlines = readLinesFromFile("W:/Projects/Minor Project 1/minor1/main/main/alignments.txt");
    std::vector<std::vector<int>> alignments(dim1, std::vector<int>(75, 0));
    for (int i = 0; i < dim1; i++) {
        std::string align = load_alignments(alignmentlines[i]);
        //std::cout << align << std::endl;
        std::vector<int> alignvec = stringToVector(align, vocab);
        for (int j = 0; j < 75; j++) {
            alignments[i][j] = (float)alignvec[j];
        }
    }
    torch::Tensor alignTensor = torch::from_blob(alignments.data(), { dim1, 75 }, torch::kFloat);

    double learning_rate = 0.001;
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));



    try {
        const int num_epochs = 2;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            torch::Tensor loss;
            int batch_size = 5;
            int batch_num = 1;
            for (int i = 0; i < batch_num; i += batch_size) {

                std::cout << "hello" << std::endl;
                torch::Tensor input_batch = tensor.index({ torch::indexing::Slice(i, i + batch_size), torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice() });
                input_batch = input_batch.permute({ 0, 4, 1, 2, 3 });
                torch::Tensor output = model.forward(input_batch).permute({ 1, 0, 2 }).to(torch::kCUDA);
                std::cout << output.sizes() << std::endl;
                torch::Tensor y_batch = alignTensor.index({ torch::indexing::Slice(i, i + batch_size), torch::indexing::Slice() });
                y_batch = y_batch.to(torch::kLong).to(torch::kCUDA);
                std::cout << y_batch.sizes() << std::endl;

                torch::Tensor input_lengths = torch::full({ batch_size }, 75, torch::kLong);
                torch::Tensor target_lengths = torch::randint(0, 27, { batch_size }, torch::kLong);
                std::cout << target_lengths[1] << std::endl;
                loss = ctc_loss(output, y_batch, input_lengths, target_lengths);

                if (loss.numel() == 0) {
                    std::cerr << "Error: Empty loss tensor." << std::endl;
                }

                loss.backward();

                optimizer.step();
            }

            std::cout << "Epoch: " << epoch + 1 << ", Loss: " << loss.item<float>() << std::endl;
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "Caught exception: " << e.what_without_backtrace() << std::endl;
    }

    try{
    torch::Tensor tensor10 = tensor1;

    torch::Tensor output10 = model.forward(tensor10);

    auto output10_tensor = output10.squeeze().argmax(1).to(torch::kInt32);
    std::vector<int> output10_vector(output10_tensor.data_ptr<int>(), output10_tensor.data_ptr<int>() + output10_tensor.numel());

    std::string op10 = vectorToString(output10_vector, vocab);
    std::cout << op10 << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Caught exception: " << e.what_without_backtrace() << std::endl;
    }


    return 0;

}



vector<std::string> readLinesFromFile(const std::string& filePath) {
    ifstream file(filePath);
    vector<std::string> lines;
    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            lines.push_back(line);
        }
        file.close();
    }
    else {
        cerr << "Unable to open file" << endl;
    }
    return lines;
}

void normalizeFrame(cv::Mat& frame, const cv::Scalar& mean, const cv::Scalar& stddev) {
    // Normalize the frame
    frame = (frame - mean[0]) / stddev;
}

void load_video(const std::string& inputPath, const std::string& outputPath) {
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error opening video file");
    }
    double originalFps = cap.get(cv::CAP_PROP_FPS);

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Empty frame detected" << std::endl;
            break;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        frame = frame(cv::Rect(80, 190, 140, 46));

        cv::Scalar mean, stddev;
        cv::meanStdDev(frame, mean, stddev);
        cout << mean << " " << stddev << endl;
        normalizeFrame(frame, mean, stddev);

        frames.push_back(frame);
        cv::imshow("frame", frame);
        int key = cv::waitKey(100);
        if (key == 27) {
            break;
        }
    }
    cap.release();

    std::string fileName = "";
    for (int i = inputPath.length() - 1; i >= 0; i--) {
        if (inputPath[i] == '/') {
            break;
        }
        fileName = inputPath[i] + fileName;
    }

    cv::imwrite(outputPath + "img.jpg", frames[0]);
    cv::VideoWriter videoWriter(outputPath + "/" + "bbaf5a.mpg", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), originalFps, frames[0].size());

    for (const auto& f : frames) {
        videoWriter.write(f);
    }

    videoWriter.release();
}

std::string load_alignments(std::string path) {
    ifstream align(path);
    std::string line;
    vector<std::string> lines;
    vector<std::string> temp;
    while (getline(align, line)) {
        temp = split(line, ' ');
        if (temp[2] != "sil") {
            lines.push_back(temp[2]);
        }
    }
    std::string s = "";
    for (std::string strr : lines) {
        for (char c : strr) {
            s += c;
        }
        s += " ";
    }
    s.erase(s.size() - 1);
    int l = s.size();
    if (s.size() != 75) {
        for (int i = l; s.size() != 75; i++) {
            s += " ";
        }
    }
    return s;
}


std::vector <int> stringToVector(const std::string& input, const std::vector<char>& vocab) {
    std::vector<int> indices;
    for (char c : input) {
        auto it = std::find(vocab.begin(), vocab.end(), c);
        if (it != vocab.end()) {
            indices.push_back(std::distance(vocab.begin(), it));
        }
        else {
            indices.push_back(vocab.size());
        }
    }
    return indices;
}

std::string vectorToString(const std::vector<int>& indices, const std::vector<char>& vocab) {
    std::string result = "";
    for (int index : indices) {
        if (index >= 0 && index < static_cast<int>(vocab.size())) {
            result += vocab[index];
        }
        else {
            std::cerr << "Warning: Index " << index << " out of range. Skipping." << std::endl;
        }
    }
    return result;
}
vector<std::string> split(std::string line, char del) {
    vector<std::string> splits;
    std::string str = "";
    for (char c : line) {
        str += c;
        if (c == del) {
            str.erase(str.size() - 1);
            splits.push_back(str);
            str = "";
        }
    }
    splits.push_back(str);
    return splits;
}



/*int64_t frames = 75;
int64_t rows = 46;
int64_t columns = 140;
int64_t channels = 1;

torch::Tensor random_input = torch::randn({ 1, channels, frames, rows, columns });*/

/*try {
    // Your forward pass code here
    torch::Tensor output = model.forward(tensor1);

}
catch (const c10::Error& e) {
    std::cerr << "Caught exception during forward pass: " << e.what() << std::endl;
}*/

/*std::ifstream modelFile("model.pth", std::ios::binary);
if (!modelFile.is_open()) {
    std::cerr << "Error opening model file" << std::endl;
    return 1;
}
torch::load(model, modelFile);
modelFile.close();*/


/*import cv2
import torch

# Assuming you have a video tensor 'video_tensor' with shape (T, H, W, C)
# T: Number of frames, H: Height, W: Width, C: Channels

# Convert the video tensor to a NumPy array
video_array = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype('uint8')

# Get video properties
height, width, _ = video_array.shape
fps = 30  # Set the frames per second

# Create a VideoWriter object
video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

# Write each frame to the video file
for frame in video_array:
    video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()

# Play the video using OpenCV
cap = cv2.VideoCapture('output_video.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    # Press 'q' to exit the video playback
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
*/