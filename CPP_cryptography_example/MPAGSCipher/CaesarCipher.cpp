
// Standard library includes
#include <iostream>
#include <string>
#include <vector>

#include <thread>
#include <future>

// Out project headers
#include "Alphabet.hpp"
#include "CaesarCipher.hpp"

CaesarCipher::CaesarCipher( const size_t key )
  : key_{key%Alphabet::size}
{
}

CaesarCipher::CaesarCipher( const std::string& key )
  : key_{0}
{
  // We have the key as a string, but the Caesar cipher needs an unsigned long, so we first need to convert it
  // We default to having a key of 0, i.e. no encryption, if no (valid) key was provided on the command line
  
  try {
    key_ = std::stoul(key);
    key_ = key_  % Alphabet::size;

  } catch (std::invalid_argument& e) {
    std::cout << "WARNING!!! Key: {" 
    << key 
    << "} could not be converted to an int.\nUsing the default key: {" 
    << key_
    << "} instead." 
    << std::endl; 
  } catch (std::out_of_range) {
    std::cout << "WARNING!!! Key: {" 
    << key 
    << "} could not be converted to an int as it is too long.\nUsing the default key: {" 
    << key_
    << "} instead." 
    << std::endl;

  };
/* 
  if ( ! key.empty() ) {
    // Before doing the conversion we should check that the string contains a
    // valid positive integer.
    // Here we do that by looping through each character and checking that it
    // is a digit. What is rather hard to check is whether the number is too
    // large to be represented by an unsigned long, so we've omitted that for
    // the time being.
    // (Since the conversion function std::stoul will throw an exception if the
    // string does not represent a valid unsigned long, we could check for and
    // handle that instead but we only cover exceptions very briefly on the
    // final day of this course - they are a very complex area of C++ that
    // could take an entire course on their own!)
    for ( const auto& elem : key ) {
      if ( ! std::isdigit(elem) ) {
	std::cerr << "[error] cipher key must be an unsigned long integer for Caesar cipher,\n"
	          << "        the supplied key (" << key << ") could not be successfully converted" << std::endl;
	return;
      }
    }
    key_ = std::stoul(key) % Alphabet::size;
  } */
}


std::string CaesarCipher::applyCipher( const std::string& inputText, const CipherMode cipherMode ) const
{

  size_t nThreads{4};

  std::vector<std::future<std::string>> futures; 
  const int key_copy = key_;

  auto process_chunk = [&cipherMode, &key_copy] (std::string chunk) {
        // Create the output string
    std::string outputText {};

    // Loop over the input text
    char processedChar {'x'};
    for ( const auto& origChar : chunk ) {

      // For each character in the input text, find the corresponding position in
      // the alphabet by using an indexed loop over the alphabet container
      for ( Alphabet::AlphabetSize i{0}; i < Alphabet::size; ++i ) {

        if ( origChar == Alphabet::alphabet[i] ) {

    // Apply the appropriate shift (depending on whether we're encrypting
    // or decrypting) and determine the new character
    // Can then break out of the loop over the alphabet
    switch ( cipherMode ) {
      case CipherMode::Encrypt :
        processedChar = Alphabet::alphabet[ (i + key_copy) % Alphabet::size ];
        break;
      case CipherMode::Decrypt :
        processedChar = Alphabet::alphabet[ (i + Alphabet::size - key_copy) % Alphabet::size ];
        break;
    }
    break;
        }
      }

      // Add the new character to the output text
      outputText += processedChar;
    }

    return outputText;
  };


  size_t initial_pos{0};
  size_t jump{inputText.size() / nThreads};
  size_t end_pos{jump};

  for (size_t i{0}; i < nThreads; ++i ) {

    if (end_pos > inputText.size()) {
      end_pos = inputText.size();
    }

    std::string chunk = inputText.substr(initial_pos, end_pos);

    futures.push_back(std::async(std::launch::async, process_chunk, chunk));


    initial_pos = i*jump;
    end_pos = initial_pos + jump -1;
  }

std::string outStr{""};
for ( size_t i{0}; i<nThreads; ++i ) {
  outStr += futures[i].get();
}

return outStr;
}
