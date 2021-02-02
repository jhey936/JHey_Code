#ifndef MPAGSCIPHER_EXCEPTIONS_HPP
#define MPAGSCIPHER_EXCEPTIONS_HPP

// Standard library includes
#include <string>
#include <stdexcept>

/**
 * \brief Exception to be thrown if a command line flag is not followed by the correct number of arguments
 * 
 * \param message the message to be printed on raising 
 */
class MissingArgument: public std::invalid_argument {
  public: 
    MissingArgument( const std::string& message ) :
      std::invalid_argument(message)
      {
      }
};

/**
 * \brief Exception to be thrown if an un-implemented cipher is passed as an arguement
 * \param message the message to be printed on raising 
 */
class InvalidCipher: public std::invalid_argument {
  public:
    InvalidCipher ( const std::string& message ): 
      std::invalid_argument(message)
      {
      }
};

class UnknownArgument: public std::invalid_argument {
  public:
    UnknownArgument ( const std::string& message ) :
    std::invalid_argument(message)
    {
    }
};

class InvalidKey: public std::invalid_argument {
  public:
    InvalidKey ( const std::string& message ) :
    std::invalid_argument(message)
    {
    }
};

#endif // MPAGSCIPHER_PROCESSCOMMANDLINE_HPP 