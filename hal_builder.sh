#!/bin/sh
git clone https://github.com/emsec/hal.git && cd hal
./install_dependencies.sh
mkdir build && cd build
cmake '../' 
make
