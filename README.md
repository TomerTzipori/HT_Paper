# Hardware Trojan Detection Project

This repository contains the code and resources for our hardware trojan detection project. Please follow the instructions below to set up and run the project.

## Installation Guide

### 1. Clone the Repository
To clone this repository to your local machine, run the following command in your terminal:
```bash
git clone https://github.com/TomerTzipori/HT_Paper.git
```

### 2. Open Terminal
- Navigate to the directory of the cloned Git repository on your system.
- Right-click within the directory and select 'Open in Terminal'.
- Ensure you are in the correct directory by typing `pwd` in the terminal and verifying the output ends with `/<git-project-name>`.

### 3. Verify Git Installation
- Check if Git is installed by running `git --version` in the terminal.
- If Git is not installed, install it using `sudo apt install git`.

### 4. Build HAL Library
- If the HAL library in the project's directory is missing or faulty, build it by running `bash hal_builder.sh` in the terminal.
- The `hal_builder.sh` script includes the build instructions from [HAL's official build instructions](https://github.com/emsec/hal/wiki/Building-HAL#build-instructions).
  - Note: There is a known issue with a missing '/' in the 'cmake .. [OPTIONS]' instruction. Ensure the correct path is used.
- This step also builds the `hal_py` library used in our project. Please wait for the build to complete before running any code that depends on this library.

## Run Guide

### Running the Project
1. Navigate to the `<git-project-name>` directory.
2. Open a terminal in this directory.
3. Run the command `python3 HT_detection`. The terminal will display the required command-line parameters.
4. Provide the appropriate parameters. The script will use HAL to analyze a set netlist via our FANCI implementation.

### Output
- The results will be saved in the `output` folder, in subfolders named after the respective gates.
- To display the results, run `calculate_benchmark.py`.

## Contact
For any issues or questions, please open an issue on this repository or contact [us](mailto:tomerz073@gmail.com).

 
 
 


