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

void computeSpecialProduct(string infile) {
    ifstream fin (infile);
    if (! fin.is_open()) {
        cout << "Unable to open file " << infile << endl;
    }
    else {
        cout << "Opened input file " << infile << endl;
        vector<int> vin;
        string line;
        while (getline(fin, line)) {
            char * word;
            word = strtok(const_cast<char *>(line.c_str()), " ,");
            while (word != nullptr) {
                vin.push_back(stoi(word));
                word = strtok(nullptr, " .");
            }
        }
        vector<int> vout = getSpecialProducts(vin);
//            for (int x : vout) cout << x << " ";
        string ofname ("vector_outputs.txt");
        ofstream outfile(ofname);
        for(vector<int>::const_iterator i = vout.begin(); i != vout.end(); ++i) {
            outfile << *i << " ";
        }
        cout << "Wrote the outputs to " << ofname << endl;
    }
}

int main() {
    vector<int> v1{2};
    vector<int> v2{2, 4, 3, 5};
    vector<int> v3{2, 4, 3, 0};
    vector<int> v4{2, 0, 3, 0};
    vector<int> out = getSpecialProducts(v1);
    for (int x : out) cout << x << " ";
    out = getSpecialProducts(v2);
    cout << endl;
    for (int x : out) cout << x << " ";
    out = getSpecialProducts(v3);
    cout << endl;
    for (int x : out) cout << x << " ";
    out = getSpecialProducts(v4);
    cout << endl;
    for (int x : out) cout << x << " ";
    cout << endl;

    string infile ("vector_inputs.txt");
    computeSpecialProduct(infile);

    return 0;
}