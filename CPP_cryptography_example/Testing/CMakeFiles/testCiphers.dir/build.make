# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/john/J_Hey_code_examples/CPP_cryptography_example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/john/J_Hey_code_examples/CPP_cryptography_example

# Include any dependencies generated for this target.
include Testing/CMakeFiles/testCiphers.dir/depend.make

# Include the progress variables for this target.
include Testing/CMakeFiles/testCiphers.dir/progress.make

# Include the compile flags for this target's objects.
include Testing/CMakeFiles/testCiphers.dir/flags.make

Testing/CMakeFiles/testCiphers.dir/testCiphers.cpp.o: Testing/CMakeFiles/testCiphers.dir/flags.make
Testing/CMakeFiles/testCiphers.dir/testCiphers.cpp.o: Testing/testCiphers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/john/J_Hey_code_examples/CPP_cryptography_example/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Testing/CMakeFiles/testCiphers.dir/testCiphers.cpp.o"
	cd /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testCiphers.dir/testCiphers.cpp.o -c /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing/testCiphers.cpp

Testing/CMakeFiles/testCiphers.dir/testCiphers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testCiphers.dir/testCiphers.cpp.i"
	cd /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing/testCiphers.cpp > CMakeFiles/testCiphers.dir/testCiphers.cpp.i

Testing/CMakeFiles/testCiphers.dir/testCiphers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testCiphers.dir/testCiphers.cpp.s"
	cd /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing/testCiphers.cpp -o CMakeFiles/testCiphers.dir/testCiphers.cpp.s

# Object files for target testCiphers
testCiphers_OBJECTS = \
"CMakeFiles/testCiphers.dir/testCiphers.cpp.o"

# External object files for target testCiphers
testCiphers_EXTERNAL_OBJECTS =

Testing/testCiphers: Testing/CMakeFiles/testCiphers.dir/testCiphers.cpp.o
Testing/testCiphers: Testing/CMakeFiles/testCiphers.dir/build.make
Testing/testCiphers: MPAGSCipher/libMPAGSCipher.a
Testing/testCiphers: Testing/CMakeFiles/testCiphers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/john/J_Hey_code_examples/CPP_cryptography_example/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testCiphers"
	cd /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testCiphers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Testing/CMakeFiles/testCiphers.dir/build: Testing/testCiphers

.PHONY : Testing/CMakeFiles/testCiphers.dir/build

Testing/CMakeFiles/testCiphers.dir/clean:
	cd /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing && $(CMAKE_COMMAND) -P CMakeFiles/testCiphers.dir/cmake_clean.cmake
.PHONY : Testing/CMakeFiles/testCiphers.dir/clean

Testing/CMakeFiles/testCiphers.dir/depend:
	cd /home/john/J_Hey_code_examples/CPP_cryptography_example && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/john/J_Hey_code_examples/CPP_cryptography_example /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing /home/john/J_Hey_code_examples/CPP_cryptography_example /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing /home/john/J_Hey_code_examples/CPP_cryptography_example/Testing/CMakeFiles/testCiphers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Testing/CMakeFiles/testCiphers.dir/depend

