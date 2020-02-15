//
// Created by Aakash on 2/13/20.
//
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_set>

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

            // Computes the length of the line from the file
            int line_length(strlen(line.c_str()));

            // If the line is not blank
            if (line_length > 0) {

                // Increments the line count
                ++line_count;

                // Counts the number of characters in the line excluding spaces and increments the character count
                for (int i = 0; i != line_length; i++) {
                    if (line[i] != ' ') ++char_count;
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
                        // Check if a string is a palindrome by comparing it with the reverse of itself. If the string
                        // is a palindrome then the palindrome count is incremented.
                        if (strcmp(s.c_str(), string(s.rbegin(), s.rend()).c_str()) == 0) ++palindrome_count;
                    }

                    // strtok takes nullptr as argument on subsequent calls and returns a pointer to the beginning of
                    // the nest token.
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
template<typename... fs>
void process_files(fs... all) {

    vector <string> filenames = {all...};

    // There should be at least 2 file names passed - one input and one output.
    // If two file names are not passed, return.
    if (filenames.size() < 2) {
        cout << "At least two file names must be passed as arguments." << endl;
        return;
    }

    // Open the last file for writing output. It creates a new file if it doesn't exist an overwrites the contents
    // if the file exists.
    ofstream outfile(filenames[filenames.size() - 1]);

    // Loop over the first n-1 files
    for (int i(0); i < filenames.size() - 1; i++) {

        ifstream fin(filenames[i]);

        // Try to open the file
        if (!fin.is_open()) {
            cout << "Unable to open file " << filenames[i] << endl;
        } else { // If file opens
            cout << "Opened input file " << filenames[i] << endl;

            // Call the function to count_stuff to get the required counts
            vector<int> v = count_stuff(filenames[i]);

            // Write a line to the output file with the required values
            outfile << filenames[i] << ", " << to_string(v[0])
                    << ", " << to_string(v[1]) << ", " << to_string(v[2])
                    << ", " << to_string(v[3]) << ", " << to_string(v[4]);

            // Add newline character for all but the last file.
            if (i < filenames.size() - 2) outfile << endl;
                // If the last input file has been processed, print message.
            else cout << "Finished processing all files." << endl;

        }
    }
}


int main() {
    // Create the names of the input files (these must exist) and the output file to pass to the function
    string in1("in_1.txt"); // The counts for this file must be: 38, 31, 8, 174, 5
    string in2("in_2.txt"); // The counts for this file must be: 23, 19, 18, 120, 23
    string out("out.txt");

    // Call the function which takes N arguments, of which the first N-1 are names of input files and the Nth argument
    // is the name of the output file to which the required outputs will be written for each input file.
    process_files(in1, in2, out);

    return 0;
}