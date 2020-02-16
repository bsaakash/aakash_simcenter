//
// Created by Aakash on 2/13/20.
//
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <cstdlib>

using namespace std;

//This function takes a string file name as input, counts the number of lines,
vector<int> count_stuff(const string &filename) {
    vector<int> v;

    ifstream infile(filename);
    if (infile.is_open()) {

        // Initialize the counts and the unordered set used to store the unique words
        int line_count(0);
        int word_count(0);
        unordered_set <string> unique_words;
        int palindrome_count(0);
        int char_count(0);

        string line;
        while (getline(infile, line)) { // Gets one line at a time from the file

            // If the line is not blank
            if (!line.empty()) {

                // Increments the line count
                ++line_count;

                // Counts the number of characters in the line excluding spaces and increments the character count
                for (auto c: line) {
                    if (isgraph(c)) ++char_count;
                }

                // Tokenizing the line - only spaces and periods are considered as separators
                char *word;
                // strtok works on c_strings, not strings and expects a cstring as argument on first call
                word = strtok(const_cast<char *>(line.c_str()), " .");
                // strtok returns a pointer to the start of each token and nullptr if the end of the cstring is reached
                while (word != nullptr) { // while end of cstring is not reached

                    // Increment the word count
                    ++word_count;

                    // Make a string of the word and transform it to all lower case letters to check if it is palindrome
                    string s(word);
                    transform(s.begin(), s.end(), s.begin(), ::tolower);

                    // Insert the string into the unordered set. This only occurs if the string does not already exist
                    // in the set. This also increments the size of the set by 1.
                    // The set always contains only unique words.
                    pair<unordered_set<string>::iterator, bool> result;
                    result = unique_words.insert(s);

                    // If the word was inserted into the set, it is unique and result.second will be true.
                    // Only then check if the string is a palindrome and increment the palindrome count.
                    if (result.second) {
                        // Check if a string is a palindrome by checking if the first half of the string is equal to the
                        // reverse of the last half of the string.
                        // If the string is a palindrome then the palindrome count is incremented.
                        // size() returns an unsigned integral type, so checking up to s.size()/2 is enough for both
                        // odd- and even-length strings.
                        if (equal(s.begin(), s.begin() + s.size() / 2, s.rbegin())) ++palindrome_count;
                    }

                    // strtok takes nullptr as argument on subsequent calls and returns a pointer to the beginning of
                    // the next token.
                    word = strtok(nullptr, " .");
                }
            }
        }

        // The counts are stored in a vector in the following order - number of words, number of unique words, number of
        // palindrome words in the set of unique words, number of characters, number of lines.
        v.push_back(word_count);
        v.push_back(unique_words.size());
        v.push_back(palindrome_count);
        v.push_back(char_count);
        v.push_back(line_count);
    }

    // Return the vector of all the counts for the file.
    return v;
}


// Define a function which can take N arguments
template<typename... file_name_types>
int process_files(file_name_types... file_names) {

    vector <string> file_name_vector = {file_names...}; // Specifying a vector of strings here is not the best design
    // and will cause an error on compilation if the function is attempted to be called with non-string arguments. But
    // if the function is called with string arguments, then this will execute correctly.

    // There should be at least 2 file names passed - one input and one output.
    // If two file names are not passed, return.
    if (sizeof...(file_names) < 2) {
        cerr << "At least two file names must be passed as arguments." << endl;
        return EXIT_FAILURE;
    }

    // Check if all the input files can be read
    for (int i(0); i < sizeof...(file_names) - 1; i++) {
//    for (int i(0); i < file_name_vector.size() - 1; i++) {

        // Create a stream object associated with the input file
        ifstream fin(file_name_vector[i]);

        // Check if the stream is associated to the file
        if (!fin.is_open()) {
            cerr << "Unable to open file: " << file_name_vector[i] << endl;
            cerr << "Exiting from the function without processing all files and without updating the output file."
                 << endl;
            return EXIT_FAILURE;
        } else { // If file opens
            continue;
        }
    }

    // Open the last file for writing output. It creates a new file if it doesn't exist and overwrites the contents
    // if the file exists.
    ofstream outfile(file_name_vector[file_name_vector.size() - 1]);

    // Loop over the first n-1 files
    for (int i(0); i < file_name_vector.size() - 1; i++) {

        cout << "Processing input file " << file_name_vector[i] << endl;

        // Call the function count_stuff to get the required counts
        vector<int> v = count_stuff(file_name_vector[i]);

        // Write a line to the output file with the required values
        outfile << file_name_vector[i] << ", " << to_string(v[0])
                << ", " << to_string(v[1]) << ", " << to_string(v[2])
                << ", " << to_string(v[3]) << ", " << to_string(v[4]);

        // Add newline character for all but the last file.
        if (i < file_name_vector.size() - 2) outfile << endl;
            // If the last input file has been processed, print message.
        else cout << "Finished processing all files." << endl;
    }
    return EXIT_SUCCESS;
}


int main() {
    // Create the names of the input files (these must exist) and the output file to pass to the function
    string in1("in_1.txt"); // The counts for this file must be: 38, 31, 8, 174, 5
    string in2("in_2.txt"); // The counts for this file must be: 23, 19, 18, 120, 23
    string out("out.txt");

    // Call the function which takes N arguments, of which the first N-1 are names of input files and the Nth argument
    // is the name of the output file to which the required outputs will be written for each input file.

//    int ret = process_files(in1, out);  // Processes one file and writes one line of output

    int ret = process_files(in1, in2, out);  // Processes 2 input files and writes two lines of output

//    int ret = process_files("not.txt", "there.txt");  // These input files do not exist and the function exits with
//    an error message

    return ret;
}