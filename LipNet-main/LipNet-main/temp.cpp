

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {
    const int dim1 = 10;
    const int dim2 = 75;
    const int dim3 = 46;
    const int dim4 = 140;
    const int dim5 = 1;

    // Read the binary file into a vector
    std::ifstream file("data.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    // Calculate the size of the array
    std::streampos startPos = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos endPos = file.tellg();
    std::streamsize size = endPos - startPos;
    file.seekg(0, std::ios::beg);

    // Read the binary data into a vector
    std::vector<double> data(size / sizeof(double));
    file.read(reinterpret_cast<char*>(data.data()), size);
    std::cout<<data[10];
    file.close();
    // Reshape the data into a 5D array
    std::vector<int> dimensions = {dim1, dim2, dim3, dim4, dim5};
    int total_size = 1;
    for (int dim : dimensions) {
        total_size *= dim;
    }

    if (data.size() != total_size) {
        std::cerr << "Error: Data size does not match expected array size." << std::endl;
        return 1;
    }

    // Create a 5D array
    double array5D[dim1][dim2][dim3][dim4][dim5];

    // Fill the 5D array with the loaded data
    int index = 0;
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                for (int l = 0; l < dim4; ++l) {
                    for (int m = 0; m < dim5; ++m) {
                        array5D[i][j][k][l][m] = data[index++];
                    }
                }
            }
        }
    }

    // Access and print a specific element of the 5D array
    int i = 0;
    int j = 74;
    int k = 40;
    int l = 7;
    int m = 0;

    std::cout << "array5D[" << i << "][" << j << "][" << k << "][" << l << "][" << m << "] = " << array5D[i][j][k][l][m] << std::endl;

    return 0;
}
