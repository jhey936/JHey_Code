//! Unit Tests for MPAGSCipher processCommandLine interface
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ProcessCommandLine.hpp"
#include "MPAGSExceptions.hpp"

TEST_CASE("Help found correctly", "[commandline]") {

  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "--help"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.helpRequested );
}

TEST_CASE("Version found correctly", "[commandline]") {

  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "--version"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.versionRequested );
}

TEST_CASE("Encrypt mode activated"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "--encrypt"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.cipherMode == CipherMode::Encrypt  );
}

TEST_CASE("Decrypt mode activated"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "--decrypt"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.cipherMode == CipherMode::Decrypt  );
}

TEST_CASE("Key entered with no key specified"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-k"};
  
  bool exception_thrown{false}; 

  try {
  processCommandLine(cmd_line, prog_set);
  } catch ( MissingArgument& e ) {
    exception_thrown = true;
  }; 

  REQUIRE( exception_thrown );
}

TEST_CASE("Key entered with key specified"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-k", "4"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.cipherKey == "4");
}

TEST_CASE("Input file declared without using input file"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-i"};


  bool exception_thrown{false}; 

  try {
  processCommandLine(cmd_line, prog_set);
  } catch ( MissingArgument& e ) {
    exception_thrown = true;
  }; 

  REQUIRE( exception_thrown );
}

TEST_CASE("Input file declared"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-i", "input.txt"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.inputFile == "input.txt");
}

TEST_CASE("Output file declared without specifying output file"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-o"};

  bool exception_thrown{false}; 

  try {
  processCommandLine(cmd_line, prog_set);
  } catch ( MissingArgument& e ) {
    exception_thrown = true;
  }; 

  REQUIRE( exception_thrown );
}

TEST_CASE("Output file declared"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-o", "output.txt"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.outputFile == "output.txt");
}

TEST_CASE("Cipher type declared without specifying cipher"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-c"};

  bool exception_thrown{false}; 

  try {
  processCommandLine(cmd_line, prog_set);
  } catch ( MissingArgument& e ) {
    exception_thrown = true;
  }; 

  REQUIRE( exception_thrown );
}

TEST_CASE("Cipher type declared with unknown cipher"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-c", "rubbish"};

  bool exception_thrown{false}; 

  try {
  processCommandLine(cmd_line, prog_set);
  } catch ( InvalidCipher& e ) {
    exception_thrown = true;
  }; 

  REQUIRE( exception_thrown );
}

TEST_CASE("Cipher type declared with Caesar cipher"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-c", "caesar"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.cipherType == CipherType::Caesar);
}

TEST_CASE("Cipher type declared with Playfair cipher"){
  
  ProgramSettings prog_set{false, false, "", "", "", CipherMode::Encrypt, CipherType::Caesar};
  std::vector<std::string> cmd_line = {"mpags-cipher", "-c", "playfair"};
  processCommandLine(cmd_line, prog_set);

  REQUIRE( prog_set.cipherType == CipherType::Playfair);
}
