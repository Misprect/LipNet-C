#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

int main(){
    string a = "sdf";
    string b = "sdfa";
    int n = (a.length()>b.length() ? b.length() : a.length());
    cout<<n<<endl;
    return 0;
}

