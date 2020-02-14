//
// Created by Aakash on 2/14/20.
//
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

vector<int> getSpecialProducts(vector<int> input) {
    int n = input.size();
    vector<int> output (n, 1);

    if (n == 1) {
        output.at(0) = 0;
        return output;
    }

    // Product of all terms in the vector before the ith term is first computed and stored in vector output
    int i, prod (1);
    for (i = 0; i < n; i++){
        output.at(i) = 1;
        output.at(i) *= prod;
        prod *= input.at(i);
    }

    // The vector output is then updated with the product of all terms after the ith term
    prod = 1;
    for (i = n-1; i >= 0; i--) {
        output.at(i) *= prod;
        prod *= input.at(i);
    }

    return output;


}

int main() {
    vector<int> v1{2};
    vector<int> v2{2, 4, 3, 5};
    vector<int> out = getSpecialProducts(v1);
    for (int x : out) cout << x << " ";
    out = getSpecialProducts(v2);
    cout << endl;
    for (int x : out) cout << x << " ";
    return 0;
}