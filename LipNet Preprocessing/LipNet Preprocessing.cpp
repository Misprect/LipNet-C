// LipNet Preprocessing.cpp : Defines the entry point for the application.
//

#include "LipNet Preprocessing.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

using namespace std;

vector<string> readLinesFromFile(const string& filePath);
void save_video(const std::string& inputPath, const std::string& outputPath);
void normalizeFrame(cv::Mat& frame, const cv::Scalar& mean, const cv::Scalar& stddev);

int main()
{
    vector<string> path = readLinesFromFile("W:/Projects/Minor Project 1/LipNet Preprocessing/LipNet Preprocessing/dataset.txt");

    if (!path.empty()) {
        cout << path[0] << endl;
    }
    else {
        cerr << "The vector is empty." << endl;
    }
    save_video(path[0], "W:/Projects/Minor Project 1/LipNet Preprocessing/ProcessedVideo");
    cout << "success" << endl;

    return 0;
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

void normalizeFrame(cv::Mat& frame, const cv::Scalar& mean, const cv::Scalar& stddev) {
    // Normalize the frame
    frame = (frame - mean[0]) / stddev;
}

void save_video(const std::string& inputPath, const std::string& outputPath) {
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

        // Accumulate mean and stddev for normalization
        cv::Scalar mean, stddev;
        cv::meanStdDev(frame, mean, stddev);
        cout << mean<< " " << stddev << endl;
        // Normalize the frame
        normalizeFrame(frame, mean, stddev);

        frames.push_back(frame);
        cv::imshow("frame", frame);
        int key = cv::waitKey(100);
        if (key == 27) {
            break;
        }
    }
    cap.release();

    // Save preprocessed and normalized frames as a new video
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