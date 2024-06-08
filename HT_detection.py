"""
The following code analyzes a VHDL netlist using the FANCI algorithm.
For each output wire (except the output wires of flipflops) inside the netlist, we calculate the "control value" of its inputs.
* Only netlists that are compatible with saed90nm_max.lib are supported

More information can be found in our project book.
"""

# importing libraries and modules
import sys, os
import random
import itertools
from statistics import mean, median
import argparse
import pandas as pd
import time

# necessary configuration for importing hal_py
HAL_BASE = "hal/build/"
os.environ["HAL_BASE_PATH"] = HAL_BASE
sys.path.append(HAL_BASE + "lib/")  # location of hal_py
import hal_py
hal_py.plugin_manager.load_all_plugins()
# path for our saed library
gate_library_path = "libs/saed90nm_max.lib"

'''
Taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
'''
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    iter_str = str(iteration) + "/" + str(total)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} {percent}% |{bar}| {iter_str} {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


'''
    A function for determining if a gate is either a latch or FF according to saed90nm library
    - a flipflop module will contain FF in its name
    - a latch module will start with "L"
    
    ** also filters vcc and ground
'''
def is_ff_or_latch(gate):
    gate_name = gate.get_type().get_name() # gets the name of the type of the gate (not instance name, module name)
    return "FF" in gate_name or gate_name.startswith("L") or gate.is_vcc_gate() or gate.is_gnd_gate() # checks for conditions stated above

'''
    Transforms a list of randomly generated boolean values to hal class to enable boolean evaluation with the hal tool
'''
def bin_list_to_hal_list(bin_list):
    return [hal_py.BooleanFunction.Value(b) for b in bin_list]

'''
    Computes the control value of each one of the inputs and returns a control value list.
    -input_list: a list containing all the nets in the fan in of the gate
    -bool_func: a boolean function that`s built from the wires in inputs_list
'''
NUM_SAMPLES = 10**4
LIST_LIM = 16
def calc_cv(input_list, bool_func):
    num_inputs = len(input_list)
    cv_list = []

    # the maximum amount of rows we will use in our calculation
    max_num_rows = min(NUM_SAMPLES, 2**(num_inputs-1))

    comb_list = None
    initial_comb_list = None
    # if num inputs <= 16, we store a list of all combinations (so we obtain 2^15 combinations)
    if num_inputs <= LIST_LIM:
        initial_comb_list = list(itertools.product([0, 1], repeat=num_inputs - 1))

    # for each index in our (eventual) random binary list
    printProgressBar(0, num_inputs, prefix='Progress:', suffix='Complete', length=25)
    for inp in range(num_inputs):
        if initial_comb_list:
            random.shuffle(initial_comb_list)
            comb_list = list(itertools.islice(initial_comb_list, max_num_rows))

        missmatch_count = 0
        i = 0
        while i < max_num_rows:
            bin_list = None
            if initial_comb_list:
                bin_list = list(comb_list.pop(0))
            else:
                bin_list = [random.choice([0, 1]) for _ in range(num_inputs - 1)]

            bin_list.insert(inp, 0) # our rand bin list is a num_inputs-1 vector + an added cofactor

            # cofactor equals 0
            hal_list = bin_list_to_hal_list(bin_list)
            input_dict = {input_list[i]: hal_list[i] for i in range(num_inputs)}
            result_0 = bool_func.evaluate(input_dict) # evaluating for cofactor 0

            # cofactor equals 1
            bin_list[inp] = 1
            hal_list = bin_list_to_hal_list(bin_list)
            input_dict = {input_list[i]: hal_list[i] for i in range(num_inputs)}
            result_1 = bool_func.evaluate(input_dict) # evaluating for cofactor 1

            # according to fancy, if evaluation for cofactor 1 and cofactor 0 isnt equal, adding 1 to the count
            if result_0 != result_1:
                missmatch_count += 1
            i += 1
        # adding the control value at index inp to the cv list
        cv_list.append(missmatch_count/max_num_rows)
        printProgressBar(inp+1, num_inputs, prefix='Progress:', suffix='Complete', length=25)
    return cv_list

'''
    dfs search of the netlist with a depth limit
    backtrace from net to global input or output of a FF (defines a module or sub-tree for which we calculate a control value)
    return all gates inbetween
'''
def dfs_from_net(gate_output_net, depth_limit=-1):
    # inits
    net_stack = []  # stack of nets
    sub_gates = set() # for calculating boolean func
    cur_depth = 0 # calculating the depth of the search

    # start of alg
    net_stack.insert(0, (gate_output_net, cur_depth))
    while net_stack:
        cur_net, cur_depth = net_stack.pop(0)
        cur_depth += 1
        # check for depth search limit
        if (depth_limit != -1 and cur_depth <= depth_limit) or depth_limit == -1:
            # searching for each gate net is connected to as the output
            for endpoint in cur_net.get_sources():
                cur_gate = endpoint.get_gate()
                if not is_ff_or_latch(cur_gate) and cur_gate not in sub_gates:
                    sub_gates.add(cur_gate) # if gate not a ff and isnt already in sub_gates, adds it to the list
                    # adding all the gate inputs that are not global inputs to the stack
                    for net in cur_gate.get_fan_in_nets():
                        if not net.is_global_input_net() and not net.is_gnd_net() and not net.is_vcc_net():
                            net_stack.insert(0, (net, cur_depth))
    return sub_gates

# helper for determining where to store the output and how to name the file
def process_output_dest(args):
    # Get the base name of the file without extension
    base_name = os.path.splitext(os.path.basename(args.netlist_path))[0]

    # Create the output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a subdirectory with the base name if it doesn't exist
    sub_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    lim_str = ""
    if args.dfs_depth >= 1:
        lim_str = "lim" + str(args.dfs_depth)
    elif args.dfs_depth == -1:
        lim_str = "inf"
    return os.path.join(sub_dir, base_name + "_" + lim_str + "_" + time.strftime("%Y%m%d-%H%M%S") + ".csv")

# START OF IMPLEMENTATION:
if __name__ == "__main__":
    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dfs_depth', type=int, help='Depth for Depth-First Search')
    parser.add_argument('netlist_path', type=str, help='Path of netlist to analyze')
    args = parser.parse_args()

    csv_path = process_output_dest(args)

    # generating the netlist and acquiring its gates
    netlist = hal_py.NetlistFactory.load_netlist(args.netlist_path, gate_library_path)
    gates = netlist.get_gates()

    # filtering out all the flipflops and latches from the gate list.
    filtered_gates = list(filter(lambda g: not is_ff_or_latch(g), gates))

    # storing values in a pandas df
    df = pd.DataFrame()

    """ implementation of alg presented in the FANCI paper, to our understanding """
    # iterates through the filtered gates
    for gate in filtered_gates:
        # prints the name and the type of the current gate
        print("CURRENT GATE:", gate.get_name(), "TYPE:", gate.get_type().get_name())
        print("------------------------------------------------------------------------------------")

        # gets output wires
        fanout_net = gate.get_fan_out_nets()
        # calculates sub gates (gates that are influencing the output of the current output net) with dfs
        sub_gates = list(dfs_from_net(fanout_net[0], args.dfs_depth))

        print("Calculated dfs") #debug

        # calculates bool func, variables and control values
        sub_bool_func = hal_py.NetlistUtils.get_subgraph_function(fanout_net[0], sub_gates).simplify()
        variables = list(sub_bool_func.get_variable_names())
        variables.sort()

        # prints results
        print("BOOLEAN FUNCTION:", sub_bool_func, "of size:", len(variables), "\nwith variable list:", variables)

        control_values = []
        mean_cv = 0
        median_cv = 0
        if len(variables):
            control_values = calc_cv(variables, sub_bool_func)
            print("CONTROL VECTOR:", control_values)

            # calculating mean and median of the control value vector
            mean_cv = mean(control_values)
            median_cv = median(control_values)

            print("Mean:", mean)
            print("Median:", median, "\n")

        if not df.empty:
            df = pd.concat([df, pd.DataFrame({'GATE_NAME': gate.get_name(), 'GATE_TYPE': gate.get_type().get_name(), 'BOOL_FUNC': sub_bool_func,
                                        'VARIABLES': [variables], 'CONTROL_VECTOR': [control_values], 'MEAN': mean_cv, 'MEDIAN': median_cv, 'SIZE': len(variables)})])
        else:
            df = pd.DataFrame({'GATE_NAME': gate.get_name(), 'GATE_TYPE': gate.get_type().get_name(), 'BOOL_FUNC': sub_bool_func,
                                        'VARIABLES': [variables], 'CONTROL_VECTOR': [control_values], 'MEAN': mean_cv, 'MEDIAN': median_cv, 'SIZE': len(variables)})
        df.to_csv(csv_path)
    exit(0)
