#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace cv = cv;
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using namespace torch::indexing;
using namespace std;


vector<char> vocab = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\'', '!', '?', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

template <typename T>
void print(const std::vector<T>& vec) {
    for (const auto& element : vec) {
        std::cout << element << std::endl;
    }
}

vector<string> readLinesFromFile(const string& filePath);
torch::Tensor load_video(const string& path);
vector<string> load_alignments(string path);
torch::Tensor char_to_num(const vector<char>& chars, const vector<char>& vocab);
vector<char> num_to_char(const torch::Tensor& nums, const vector<char>& vocab);
vector<string> split(string line, char del);
tuple<torch::Tensor, torch::Tensor>(string path_vid, string path_align);






















int main(){
    string path;
    
}




vector<string> readLinesFromFile(const string& filePath) {
    ifstream file(filePath);
    vector<string> lines;
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            lines.push_back(line);
        }
        file.close();
    } else {
        cerr << "Unable to open file" << endl;
    }
    return lines;
}

torch::Tensor load_video(const string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        exit(-1);
    }

    vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame)) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        frame = frame(cv::Rect(80, 190, 140, 46)); // Cropping the frame
        frames.push_back(frame);
    }
    cap.release();

    // Convert frames to type CV_32F for processing
    vector<cv::Mat> processed_frames;
    for (const auto& f : frames) {
        cv::Mat temp;
        f.convertTo(temp, CV_32F);
        temp = (temp - temp.mean()) / temp.std();
        processed_frames.push_back(temp);
    }

    // Create a tensor from the processed frames
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).layout(torch::kStrided).requires_grad(false);
    torch::Tensor tensor = torch::from_blob(processed_frames.data(), {static_cast<long>(frames.size()), 46, 140}, options).clone();

    return tensor;
}

torch::Tensor char_to_num(const vector<char>& chars, const vector<char>& vocab) {
    map<char, int> char_to_numv;
    for (size_t i = 0; i < vocab.size(); ++i) {
        char_to_numv[vocab[i]] = i;
    }
    vector<int> indices;
    for (char c : chars) {
        indices.push_back(char_to_numv[c]);
    }
    return torch::tensor(indices, torch::kLong);
}

vector<char> num_to_char(const torch::Tensor& nums, const vector<char>& vocab) {
    map<int, char> num_to_charv;
    for (size_t i = 0; i < vocab.size(); ++i) {
        num_to_charv[i] = vocab[i];
    }
    vector<char> chars;
    for (auto num : nums) {
        chars.push_back(num_to_charv[num.item<int>()]);
    }
    return chars;
}

vector<string> split(string line, char del){
    vector<string> splits;
    string str = "";
    for(char c : line){
        str += c;
        if(c == del){
            splits.push_back(str);
            str = "";
        }
    }
    splits.push_back(str);
    return splits;
}

vector<string> load_alignments(string path){
    ifstream align(path);
    string line;
    vector<string> lines;
    vector<string> temp;
    while(getline(align, line)){
        temp = split(line, ' ');
        if(temp[2] != "sil"){
            lines.push_back(temp[2]);
        }
    }
    vector<char> tokens;
    for(string s : lines){
        for(char c : s){
            tokens.push_back(c);
        }
        tokens.push_back(' ');
    }
    tokens.pop_back();
    return char_to_num(tokens, vocab);
}

tuple<torch::Tensor, torch::Tensor>(string path_vid, string path_align){
    torch::Tensor tensor1 = load_video(path_vid);
    torch::Tensor tensor2 = load_alignments(path_align);
    return make_tuple(tensor1, tensor2);
}