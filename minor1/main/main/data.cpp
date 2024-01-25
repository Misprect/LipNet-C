
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

vector<char> vocab = { ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\'', '!', '?', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '@' };

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

    /*torch::Tensor video_tensor = tensor.index({torch::indexing::Slice(0,1), torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice()});
    std::cout << video_tensor.sizes() << endl;
    video_tensor = torch::squeeze(video_tensor, 0);
    std::cout << video_tensor.sizes() << endl;

    const std::string video_path = "M:/output_video.mp4";
    cv::VideoWriter video_writer(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(video_tensor.size(2), video_tensor.size(1)), true);

    for (int i = 0; i < video_tensor.size(0); ++i) {
        cv::Mat single_channel_frame(video_tensor.size(1), video_tensor.size(2), CV_32FC1, video_tensor[i].data<float>());

        single_channel_frame *= 255.0;
        single_channel_frame.convertTo(single_channel_frame, CV_8U);

        cv::Mat bgr_frame;
        cv::cvtColor(single_channel_frame, bgr_frame, cv::COLOR_GRAY2BGR);

        video_writer.write(bgr_frame);
    }

    video_writer.release();*/


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

    
    cv::VideoCapture cap("w:/Projects/Minor Project 1/minor1/data/s1/bbaf2n.mpg");

    // Check if the video file opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file." << std::endl;
        return -1;
    }

    // Create a window to display the video
    cv::namedWindow("Video Player", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Player", 1920, 1200);  // Adjust the size as needed

    while (true) {
        // Read a frame from the video
        cv::Mat frame;
        cap >> frame;

        // Check if the video has ended
        if (frame.empty()) {
            std::cout << "End of video" << std::endl;
            break;
        }

        // Write on each frame
        cv::putText(frame, op, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        // Display the frame in the window
        cv::imshow("Video Player", frame);

        // Wait for a key event (parameter is the delay in milliseconds, 0 means wait indefinitely)
        int key = cv::waitKey(100);

        // Check for the 'Esc' key (ASCII code 27) to exit the loop
        if (key == 27) {
            break;
        }
    }

    // Release the video capture object and close the window
    cap.release();
    cv::destroyAllWindows();


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