//
// Created by Aakash on 2/13/20.
//
#include <cstdarg>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_set>

using namespace std;

//void process_files(int n...) {
//    va_list(fnames);
//    va_start(fnames, n);
//    const char *fname;
//
//    vector<string> input_file_name;
//    vector<string> number_of_words;
//    vector<string> number_of_unique_words;
//    vector<string> number_of_palindrome_words;
//    vector<string> number_of_characters;
//    vector<string> number_of_lines;
//
//    string line;
//    for (int i = 0; i < n - 1; ++i) {
//
//        fname = va_arg(fnames, const char*);
//
//        ifstream fin(fname);
//        if (fin.is_open()) {
//
//            int line_count(0);
//            int word_count(0);
//            int unique_word_count(0);
//            unordered_set <char*> unique_word_set;
//            int palindrome_count(0);
//
//            while (getline(fin, line)) {
//
//                int line_length(strlen(line.c_str()));
//                if (line_length > 0) {
//                    ++line_count;
//                    char * word;
//                    word = strtok(const_cast<char *>(line.c_str()), " .");
//                    while (word != nullptr) {
//                        ++word_count;
//                        cout << "Word: " << word << endl;
//                        if (unique_word_set.find(word) == unique_word_set.end()){
//                            cout << "is unique" << endl;
//                            unique_word_set.insert(word);
//                            ++unique_word_count;
//                            string s (word);
//                            transform(s.begin(), s.end(), s.begin(), ::tolower);
//                            if (s == string(s.rbegin(), s.rend())) {
//                                ++palindrome_count;
//                                cout << word << " is a palindrome.\n";
//                            }
//                            else
//                                cout << word << " is NOT a palindrome.\n";
//                        }
//                        else cout << "is not unique" << endl;
//                        word = strtok(nullptr, " .");
//                    }
//                }
//            }
//            fin.close();
//
//
//            input_file_name.emplace_back(fname);
//            number_of_words.emplace_back(to_string(word_count));
//            number_of_unique_words.emplace_back(to_string(unique_word_count));
//            number_of_palindrome_words.emplace_back(to_string(palindrome_count));
//            number_of_lines.push_back(to_string(line_count));
//
//        } else cout << "Unable to open file " << fname << endl;
//    }
//
//    fname = va_arg(fnames, const char*);
//
//    ofstream fout(fname);
//    if (fout.is_open()) {
//
//        for (int i = 0; i < n - 1; ++i) {
//            fout << input_file_name[i] << ", " << number_of_words[i] << ", "
//            << number_of_unique_words[i] << ", " << number_of_palindrome_words[i] << ", "
//            << number_of_lines[i] << endl;
//        }
//
//        fout.close();
//
//    } else cout << "Unable to open file";
//
//
//    va_end(fnames);
//}


template <typename filename, typename... fs>
void process_filenames(filename inp, fs... all) {
    vector<string> filenames = {inp, all...};
    if (filenames.size() < 2) {
        cout << "At least two filenames must be passed as arguments." << endl;
        return;
    }
    vector<string> number_of_words;
    vector<string> number_of_unique_words;
    vector<string> number_of_palindrome_words;
    vector<string> number_of_characters;
    vector<string> number_of_lines;

    ofstream outfile(filenames[filenames.size()-1]);

    for (int i(0); i < filenames.size()-1; i++) {
        ifstream fin (filenames[i]);
        if (! fin.is_open()) {
            cout << "Unable to open file " << filenames[i] << endl;
        }
        else {
            cout << "Opened input file " << filenames[i] << endl;



            outfile << filenames[i] << endl;

        }
    }



int main() {
//    const char *f1("in_1.txt");
//    const char *f2("in_2.txt");
//    const char *f3("out.txt");
//    process_files(3, f1, f2, f3);
    string in1 ("in_1.txt");
    string in2 ("in_2.txt");
    string out ("out.txt");
    process_filenames(in1, in2);
    return 0;
}