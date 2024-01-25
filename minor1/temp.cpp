#include <iostream>
#include <maths>
#include <vector>
#include <map>

std::vector<char> vocab = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\'', '!', '?', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

// Function to convert a list of characters to their corresponding numbers
std::vector<int> char_to_num(const std::vector<char>& chars, const std::vector<char>& vocab) {
    std::map<char, int> char_to_numv;
    for (size_t i = 0; i < vocab.size(); ++i) {
        char_to_numv[vocab[i]] = i;
    }
    std::vector<int> indices;
    for (char c : chars) {
        indices.push_back(char_to_numv[c]);
    }
    return indices;
}

// Function to convert a list of numbers to their corresponding characters
std::vector<char> num_to_char(const std::vector<int>& nums, const std::vector<char>& vocab) {
    std::map<int, char> num_to_charv;
    for (size_t i = 0; i < vocab.size(); ++i) {
        num_to_charv[i] = vocab[i];
    }
    std::vector<char> chars;
    for (int num : nums) {
        chars.push_back(num_to_charv[num]);
    }
    return chars;
}

int main() {
    std::vector<int> nums = {14, 9, 3, 11};
    std::vector<char> result_chars = num_to_char(nums, vocab);
    for (char c : result_chars) {
        std::cout << c;
    }
    std::cout << std::endl;

    std::vector<char> input_chars = {'n', 'i', 'c', 'k'};
    std::vector<int> result_nums = char_to_num(input_chars, vocab);
    for (int num : result_nums) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
