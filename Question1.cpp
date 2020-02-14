//
// Created by Aakash on 2/13/20.
//
#include <cstdarg>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;

void process_files(int n...) {
    va_list(fnames);
    va_start(fnames, n);
    const char *fname;

    vector<string> input_file_name;
    vector<string> number_of_words;
    vector<string> number_of_unique_words;
    vector<string> number_of_palindrome_words;
    vector<string> number_of_characters;
    vector<string> number_of_lines;

    string line;
    for (int i = 0; i < n - 1; ++i) {

        fname = va_arg(fnames, const char*);

        ifstream fin(fname);
        if (fin.is_open()) {

            input_file_name.emplace_back(fname);

            int line_count(0);
            while (getline(fin, line)) {

                int line_length(strlen(line.c_str()));
                if (line_length > 0) {
                    ++line_count;
                }
//                cout << line << " - " << line_length << endl;
            }
            fin.close();

            number_of_lines.push_back(to_string(line_count));

        } else cout << "Unable to open file " << fname << endl;
    }

    fname = va_arg(fnames, const char*);

    ofstream fout(fname);
    if (fout.is_open()) {

        for (int i = 0; i < n - 1; ++i) {
            fout << input_file_name[i] << ", " << number_of_lines[i] << endl;
        }

        fout.close();

    } else cout << "Unable to open file";


    va_end(fnames);
}

int main() {
    const char *f1("in_1.txt");
    const char *f2("in_2.txt");
    const char *f3("out.txt");
    process_files(3, f1, f2, f3);
    return 0;
}