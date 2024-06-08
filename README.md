Installation Guide:

1. Terminal:
 - from the HTproject directory, press right click, then select 'open in terminal'
 - a terminal should open. to make sure we are in the /HTproject directory, type the 'cwd' command in the terminal
   and make sure that the output ends with '/HTproject'
 * for the rest of the tutorial we will run commands from this specific terminal session

2. GIT:
 - use 'git --version' via the terminal to check if git is installed on your system 
 - if the command fails, install git using 'sudo apt install git'
 
3. Building HAL:
 - if the hal library on the project`s document is faulty or non existant, use 'bash hal_builder.sh' via the terminal
 - 'hal_builder.sh' includes the build instructions in https://github.com/emsec/hal/wiki/Building-HAL#build-instructions  
 * there is a missing '/' in the original 'cmake .. [OPTIONS]' instruction, atleast in plain sight due to a bug. 
 - this step also builds the hal_py library we commonly use in our project,
   so make sure to wait for the build to finish (take lots of time) before running any code using said library
   
Run Guide:

 - In order to run the project, from the HTproject directory, press right click, then select 'open in terminal'
 - then type the command 'python3 HT_detection'. the script uses HAL to analyze a set netlist using FANCI

 
 
 


