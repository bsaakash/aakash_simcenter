//
// Created by Aakash on 2/13/20.
//
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_set>

using namespace std;

vector<int> count_stuff(const string& filename) {
    vector<int> v;

    ifstream infile(filename);
    if (infile.is_open()) {
        int line_count(0);
        int word_count(0);
        unordered_set <string> unique_words;
        int palindrome_count(0);
        int char_count(0);

        string line;
        while (getline(infile, line)) {
            int line_length (strlen(line.c_str()));
            for(int i = 0; i != line_length; i++) {
                if(line[i] != ' ') ++char_count;
            }
            if (line_length > 0) {
                ++line_count;
                char * word;
                word = strtok(const_cast<char *>(line.c_str()), " .");
                while (word != nullptr) {
                    ++word_count;
                    string s (word);
                    transform(s.begin(), s.end(), s.begin(), ::tolower);
                    unique_words.insert(s);
                    if (strcmp(s.c_str(), string(s.rbegin(), s.rend()).c_str()) == 0) ++palindrome_count;
                    word = strtok(nullptr, " .");
                }
            }
        }

        v.push_back(word_count);
        v.push_back(unique_words.size());
        v.push_back(palindrome_count);
        v.push_back(char_count);
        v.push_back(line_count);
    }

    return v;
}


template <typename filename, typename... fs>
void process_files(filename inp, fs... all) {
    vector<string> filenames = {inp, all...};

    if (filenames.size() < 2) {
        cout << "At least two file names must be passed as arguments." << endl;
        return;
    }


    ofstream outfile(filenames[filenames.size()-1]);
    for (int i(0); i < filenames.size()-1; i++) {
        ifstream fin (filenames[i]);
        if (! fin.is_open()) {
            cout << "Unable to open file " << filenames[i] << endl;
        }
        else {
            cout << "Opened input file " << filenames[i] << endl;
            vector<int> v = count_stuff(filenames[i]);

            outfile << filenames[i] << ", " << to_string(v[0])
                    << ", " << to_string(v[1]) << ", " << to_string(v[2])
                    << ", " << to_string(v[3]) << ", " << to_string(v[4]);

            if (i < filenames.size()-2) outfile << endl;
            else cout << "Finished processing all files." << endl;

        }
    }
}


int main() {
    string in1 ("in_1.txt");
    string in2 ("in_2.txt");
    string out ("out.txt");
    process_files(in1, in2, out);
    return 0;
}