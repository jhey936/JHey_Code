# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/john/Software/cpp_course/6_week6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/john/Software/cpp_course/6_week6/build

# Include any dependencies generated for this target.
include Testing/CMakeFiles/testPlayfairCipher.dir/depend.make

# Include the progress variables for this target.
include Testing/CMakeFiles/testPlayfairCipher.dir/progress.make

# Include the compile flags for this target's objects.
include Testing/CMakeFiles/testPlayfairCipher.dir/flags.make

Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o: Testing/CMakeFiles/testPlayfairCipher.dir/flags.make
Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o: ../Testing/testPlayfairCipher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/john/Software/cpp_course/6_week6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o"
	cd /home/john/Software/cpp_course/6_week6/build/Testing && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o -c /home/john/Software/cpp_course/6_week6/Testing/testPlayfairCipher.cpp

Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.i"
	cd /home/john/Software/cpp_course/6_week6/build/Testing && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/john/Software/cpp_course/6_week6/Testing/testPlayfairCipher.cpp > CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.i

Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.s"
	cd /home/john/Software/cpp_course/6_week6/build/Testing && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/john/Software/cpp_course/6_week6/Testing/testPlayfairCipher.cpp -o CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.s

Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.requires:

.PHONY : Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.requires

Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.provides: Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.requires
	$(MAKE) -f Testing/CMakeFiles/testPlayfairCipher.dir/build.make Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.provides.build
.PHONY : Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.provides

Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.provides.build: Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o


# Object files for target testPlayfairCipher
testPlayfairCipher_OBJECTS = \
"CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o"

# External object files for target testPlayfairCipher
testPlayfairCipher_EXTERNAL_OBJECTS =

Testing/testPlayfairCipher: Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o
Testing/testPlayfairCipher: Testing/CMakeFiles/testPlayfairCipher.dir/build.make
Testing/testPlayfairCipher: MPAGSCipher/libMPAGSCipher.a
Testing/testPlayfairCipher: Testing/CMakeFiles/testPlayfairCipher.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/john/Software/cpp_course/6_week6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testPlayfairCipher"
	cd /home/john/Software/cpp_course/6_week6/build/Testing && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testPlayfairCipher.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Testing/CMakeFiles/testPlayfairCipher.dir/build: Testing/testPlayfairCipher

.PHONY : Testing/CMakeFiles/testPlayfairCipher.dir/build

Testing/CMakeFiles/testPlayfairCipher.dir/requires: Testing/CMakeFiles/testPlayfairCipher.dir/testPlayfairCipher.cpp.o.requires

.PHONY : Testing/CMakeFiles/testPlayfairCipher.dir/requires

Testing/CMakeFiles/testPlayfairCipher.dir/clean:
	cd /home/john/Software/cpp_course/6_week6/build/Testing && $(CMAKE_COMMAND) -P CMakeFiles/testPlayfairCipher.dir/cmake_clean.cmake
.PHONY : Testing/CMakeFiles/testPlayfairCipher.dir/clean

Testing/CMakeFiles/testPlayfairCipher.dir/depend:
	cd /home/john/Software/cpp_course/6_week6/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/john/Software/cpp_course/6_week6 /home/john/Software/cpp_course/6_week6/Testing /home/john/Software/cpp_course/6_week6/build /home/john/Software/cpp_course/6_week6/build/Testing /home/john/Software/cpp_course/6_week6/build/Testing/CMakeFiles/testPlayfairCipher.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Testing/CMakeFiles/testPlayfairCipher.dir/depend

