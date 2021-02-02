Code style:

Try and be consistent with PEP8. (https://www.python.org/dev/peps/pep-0008)

I have been a little inconsistent with my docstrings, but I am now trying to follow Google's conventions:
https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments
New docstrings should all be in this style

If you need to change a docstring please update it to this style
Note: there is a pycharm option to help with this

Try to be consistent with class/method/variable naming.
Classes: CamelCase
Instances: lowercase
Methods/functions: lower_case_with_underscores


I am trying to provide type hints with all public function and method calls where possible, so if you spot an un-hinted
 argument, feel free to correct it. Also try to type hint any new functions etc.



Project interpreter:

I am only testing using python version 3.6.4 but this project should work with any version above 3.5

I suggest using a clean virtual environment to run this project:
requirements.txt as well as Pipfiles for using pipenv should be kept up to date in the top-level directory.



This preamble should be added at the top of any new files:

"""

bmpga: A program for finding global minima
Copyright (C) 2018- ; John Hey
This file is part of bmpga.

bmpga is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License V3.0 as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

bmpga is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License V3.0 for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


=========================================================================
Author Name (Created: Date)

Short description

=========================================================================
"""