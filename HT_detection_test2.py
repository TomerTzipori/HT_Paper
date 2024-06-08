# importing libraries and modules
import sys, os
import random
import itertools
from statistics import mean, median
import argparse
import pandas as pd
import time
from tqdm import tqdm
from Trojan_Mapping import *
import numpy as np
# necessary configuration for importing hal_py
HAL_BASE = "hal/build/"
os.environ["HAL_BASE_PATH"] = HAL_BASE
sys.path.append(HAL_BASE + "lib/")  # location of hal_py
import hal_py
hal_py.plugin_manager.load_all_plugins()

wire_dict = {}

"""
this is a test version of HT_detection.py
it adds the following features:
- only boolean functions above 100 inputs will get simplified (unlike the old file which simplifies all)
    -> in some cases simplification changes the CV completely (in uart it can change mean values from 0 to 1 which is not desirable)
- generates random truth table entries per gate and not per cofactor like in the old file
- changes the way we traverse the gates:
    -> first, we traverse the gates in order and calculate their size (in inputs), then write the data gathered in our df to the csv file
    -> secondly, we either traverse the gates from the smaller ones to the bigger ones (input wise) and calculate their respected cv,
       or calculate only the gates inside the trojan mapping (also in the order described above)
    -> we also added an option to do the same process starting from a specific entry in the original order (to compensate power outages - thanks Afula ^_^)
- we removed printing the entire boolean function in both the console and the df/csv file
- progress bar will be printed with tqdm from now on (not the old func)
- added use of Trojan_Mapping.py (importing modules)
- added fstring in places where str1 + int(non string) + str2 was used (helps to simplify stuff)
- updated function commenting according to pycharm conventions
- tweaked the command line parameters:
    -> implemented a custom argparser class
    -> added proper description and example usage
    -> for dfs depth, typing "inf" in the command line will replicate the behaviour of the old py file when typing "-1"
    -> for starting from a specific gate, added a param called starting_gate
    -> for only calculating for trojan gates, added a param called only_trojan
    -> removed the positional arguments, now before typing each value in the command line, a prefix will have to get typed as well
"""

"""
The following code analyzes a VHDL netlist using the FANCI algorithm.
For each output wire (except the output wires of flipflops) inside the netlist, we calculate the "control value" of its inputs.
* Only netlists that are compatible with saed90nm_max.lib are supported

More information can be found in our project book.
"""

# consts
GATE_LIB_PATH = "libs/saed90nm_max.lib"  # the gate library we use to parse the files
NUM_SAMPLES = 10 ** 4  # number of rows to sample from truth table
LIST_LIM = 16  # input threshold for storing a list with all binary combs


def is_ff_or_latch(gate):
    """
    A function for determining if a gate is either a latch or FF. \n
    according to the saed90nm library: \n
    * a flipflop module will contain FF in its type name \n
    * a latch module will start with "L" \n
    therefore, our function checks if the gate`s type satisfies one of the 2 statements above
    :param gate: a hal Gate object
    :return: True if the gate is a latch or a flipflop according to our conditions, else False
    """
    gate_type = gate.get_type().get_name()  # gets the name of the type of the gate (not instance name, module name)
    return "FF" in gate_type or gate_type.startswith("L") or gate.is_vcc_gate() or gate.is_gnd_gate()  # checks for conditions stated above


def bin_list_to_hal_list(bin_list):
    """
    Converts a list with values that are either 0 or 1, to a list with hal_py.BooleanFunction.Value objects \n
    for example, if our bin_list is [0, 1], the returned list will be [hal_py.BooleanFunction.ZERO, hal_py.BooleanFunction.ONE]
    :param bin_list:  our boolean list described earlier
    :return: a list with type conversion of bin_list items to hal_py.BooleanFunction.Value objects
    """
    return [hal_py.BooleanFunction.Value(b) for b in bin_list]


def calc_cv(input_list, bool_func, gate_name):
    """
    Computes the control value of each one of the inputs and returns a control value list.
    :param input_list: a list containing all the nets in the fan-in of the gate
    :param bool_func: a boolean function that`s built from the wires in inputs_list
    :return: cv_list: a list containing the control values of each input
    """

    num_inputs = len(input_list)
    cv_list = []

    # the maximum amount of rows we will use in our calculation
    max_num_rows = min(NUM_SAMPLES, 2**(num_inputs-1))

    comb_list = None
    initial_comb_list = None
    # if num inputs <= 16, we store a list of all combinations (so we obtain 2^15 combinations)
    if num_inputs <= LIST_LIM:
        initial_comb_list = list(itertools.product([0, 1], repeat=num_inputs - 1))

    # if num inputs <= 16, we chose randomly max_num_rows amount of items from initial_comb_list
    if initial_comb_list:
        random.shuffle(initial_comb_list)
        comb_list = list(itertools.islice(initial_comb_list, max_num_rows))
    # else, we take the risk and calculate max_num_rows amount of random vectors (there could be duplicates but its not likely)
    else:
        comb_list = [[random.choice([0, 1]) for _ in range(num_inputs - 1)] for i in range(max_num_rows)]
        
    # calculates the cv for each input
    for cofactor in tqdm(range(num_inputs)):
        missmatch_count = 0
        i = 0
        # evaluates the boolean function for each sampled row, for both cofactor options
        while i < max_num_rows:
            bin_list = list(comb_list[i]).copy()
            bin_list.insert(cofactor, 0)  # our rand bin list is a num_inputs-1 vector + an added cofactor

            # cofactor equals 0
            hal_list = bin_list_to_hal_list(bin_list)
            input_dict = {input_list[i]: hal_list[i] for i in range(num_inputs)}
            result_0 = bool_func.evaluate(input_dict)  # evaluating for cofactor 0

            # cofactor equals 1
            bin_list[cofactor] = 1
            hal_list = bin_list_to_hal_list(bin_list)
            input_dict = {input_list[i]: hal_list[i] for i in range(num_inputs)}
            result_1 = bool_func.evaluate(input_dict)  # evaluating for cofactor 1

            # according to fancy, if evaluation for cofactor==1 and cofactor==0 isn't equal, adding 1 to the count
            if result_0 != result_1:
                missmatch_count += 1
            i += 1
        # adding the control value at index inp to the cv list
        cv_list.append(missmatch_count/max_num_rows)
        if input_list[cofactor] in wire_dict.keys():
            wire_dict[input_list[cofactor]].append((missmatch_count/max_num_rows, gate_name))
        else:
            wire_dict[input_list[cofactor]] = [(missmatch_count/max_num_rows, gate_name)]

    return cv_list


def dfs_from_net(gate_output_net, depth_limit="inf"):
    """
    dfs search of the netlist with a depth limit \n
    backtrace from net to global input or output of a FF (defines a module or sub-tree for which we calculate a control value)
    :param gate_output_net: the wire we start our search from
    :param depth_limit: maximum number of gates we can traverse vertically
    :return: sub_gates - all the gates that are connected through wires to the given gate (according to the dfs )
    """

    # init
    net_stack = []  # stack of nets
    sub_gates = set()  # for calculating boolean func
    cur_depth = 0  # calculating the depth of the search
    wires = {}

    # start of alg
    net_stack.insert(0, (gate_output_net, cur_depth, 1))
    while net_stack:
        cur_net, cur_depth, cur_cv = net_stack.pop(0)
        print(f'popped net: net_{cur_net.get_id()}')
        cur_depth += 1
        # check for depth search limit
        if (depth_limit != "inf" and cur_depth <= int(depth_limit)) or depth_limit == "inf":
            print(f'popped net: net_{cur_net.get_id()}')
            # searching for each gate net is connected to as the output
            for endpoint in cur_net.get_sources():
                cur_gate = endpoint.get_gate()
                if not is_ff_or_latch(cur_gate) and cur_gate not in sub_gates:
                    sub_gates.add(cur_gate)  # if gate not a ff and isn't already in sub_gates, adds it to the list
                    # adding all the gate inputs that are not global inputs to the stack
                    for net in cur_gate.get_fan_in_nets():
                        if not net.is_global_input_net() and not net.is_gnd_net() and not net.is_vcc_net():
                            net_stack.insert(0, (net, cur_depth, cur_cv * wire_dict[f'net_{net.get_id()}']))
                            wires[f'net_{net.get_id()}'] = cur_cv * wire_dict[f'net_{net.get_id()}']
            print(wires)

    return wires

def dfs_from_net2(gate_output_net, depth_limit="1"):
    """
    dfs search of the netlist with a depth limit \n
    backtrace from net to global input or output of a FF (defines a module or sub-tree for which we calculate a control value)
    :param gate_output_net: the wire we start our search from
    :param depth_limit: maximum number of gates we can traverse vertically
    :return: sub_gates - all the gates that are connected through wires to the given gate (according to the dfs )
    """

    # init
    net_stack = []  # stack of nets
    sub_gates = set()  # for calculating boolean func
    cur_depth = 0  # calculating the depth of the search

    # start of alg
    net_stack.insert(0, (gate_output_net, cur_depth))
    while net_stack:
        cur_net, cur_depth = net_stack.pop(0)
        cur_depth += 1
        # check for depth search limit
        if (depth_limit != "inf" and cur_depth <= int(depth_limit)) or depth_limit == "inf":
            # searching for each gate net is connected to as the output
            for endpoint in cur_net.get_sources():
                cur_gate = endpoint.get_gate()
                if not is_ff_or_latch(cur_gate) and cur_gate not in sub_gates:
                    sub_gates.add(cur_gate)  # if gate not a ff and isn't already in sub_gates, adds it to the list
                    # adding all the gate inputs that are not global inputs to the stack
                    for net in cur_gate.get_fan_in_nets():
                        if not net.is_global_input_net() and not net.is_gnd_net() and not net.is_vcc_net():
                            net_stack.insert(0, (net, cur_depth))
    return sub_gates

class CustomParser(argparse.ArgumentParser):
    """
    Custom class which inherits from argparse.ArgumentParser \n
    We created this class in order to override the error message
    """
    def error(self, message):
        """
        writes the error message and the help message of the parser into stderr. then exists the program
        :param message: error message to display
        """
        sys.stderr.write('error: %s' % message)
        self.print_help(sys.stderr)
        sys.exit(2)


def process_csv_name(args):
    """
    determines where our csv table should be saved based on netlist_path, dfs_depth and current time
    :param args: the arges we parsed from the command line
    :return: the path of the csv output file
    """
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

    return os.path.join(sub_dir, f'{base_name}_{args.dfs_depth}_{time.strftime("%Y%m%d-%H%M%S")}_test.csv')



description = ("The program attempts to replicate the algorithm presented in the FANCI paper, "
                "which tries to detect suspicious wires in a netlist based on their \"control values\".\n"
                "The program computes the control value for each input within the gate's fan-in, "
                "and additionally determines the mean and median values based on all of the calculated control values from the gate's inputs."
                "The results will be saved inside a csv file, under \'output\\<base netlist name>\' (the base netlist name is without file extensions)")


# START OF IMPLEMENTATION:
if __name__ == "__main__":
    # parsing command line arguments
    parser = CustomParser()

    parser.add_argument('-depth', '-dfs_depth', dest='dfs_depth', type=str, required=True, help='Depth for Depth-First Search, either "inf" or a positive integer')
    parser.add_argument('-path', '-netlist_path', dest='netlist_path', type=str, required=True, help='Path of netlist to analyze')
    parser.add_argument('-gate', '-starting_gate', dest='starting_gate', type=str, required=False, help='Gate to start from')
    parser.add_argument('--only_trojan', action='store_true', required=False, help='Trojan gates analysis only')
    parser.add_argument('--sort_gates', action='store_true', required=False, help='Traverses the gates in the analysis according to their'
                                                                              ' input size (from bottom to top)')
    args = parser.parse_args()

    # checking for validity of dfs_depth
    if args.dfs_depth != "inf":
        if not args.dfs_depth.isdigit() or int(args.dfs_depth) <= 0:
            parser.error('dfs_depth can only be "inf" or a positive integer')

    # checking for validity of netlist_path
    if not os.path.exists(args.netlist_path):
        parser.error(f'the path "{args.netlist_path}" doesnt exists')

    # determines a suitable path name to our output
    csv_path = process_csv_name(args)

    # generating the netlist and acquiring its gates
    netlist = hal_py.NetlistFactory.load_netlist(args.netlist_path, GATE_LIB_PATH)
    gates = netlist.get_gates()

    filtered_gates = []
    # check if only_trojan option was selected
    if args.only_trojan:
        # if given netlist is indeed mapped, filtered_gates will only contain to mapped gates to the netlist. else, prints error.
        if is_netlist_in_mapping(args.netlist_path):
            # not is_ff_or_latch(g) is included in lambda just in case
            filtered_gates = list(filter(lambda g: not is_ff_or_latch(g) and is_trojan_gate(g.get_name(), args.netlist_path), gates))
        else:
            parser.error(f'--only_trojan was selected, but the given netlist base name "{os.path.basename(args.netlist_path)}" '
                         f'is not mapped inside trojan_mapping')
    else:
        # normal filter - filtering out all the flipflops and latches from the gate list.
        filtered_gates = list(filter(lambda g: not is_ff_or_latch(g), gates))

    # check for validity of starting_gate
    if args.starting_gate:
        if len(list(filter(lambda g: args.starting_gate == g.get_name(), filtered_gates))) == 0:
            parser.error(f'The gate {args.starting_gate} is not found in the netlist file "{os.path.basename(args.netlist_path)}"')
        else:
            # Find the index of the gate with args.starting_gate
            index_of_starting_gate = next((i for i, gate in enumerate(filtered_gates) if gate.get_name() == args.starting_gate), None)

            # cut the gates that have a lower index than starting gate in the list
            filtered_gates = filtered_gates[index_of_starting_gate:]

    # setting up an empty df
    columns = ['GATE_NAME', 'GATE_TYPE', 'BOOL_FUNC', 'VARIABLES', 'CONTROL_VECTOR', 'MEAN', 'MEDIAN', 'SIZE']
    df = pd.DataFrame(columns=columns)

    # for saving the results for each gate
    bool_func_dict = {}

    print("Performing First Iteration...")
    for gate in filtered_gates:
        # gets output wires
        fanout_net = gate.get_fan_out_nets()
        # calculates sub gates (gates that are influencing the output of the current output net) with dfs
        sub_gates = gate

        # calculates bool func, variables and control values
        sub_bool_func = hal_py.NetlistUtils.get_subgraph_function(fanout_net[0], [sub_gates])
        variables = list(sub_bool_func.get_variable_names())
        # simplifies the bool func if there are more than 100 variables in the original func
        if len(variables) >= 100:
            sub_bool_func = sub_bool_func.simplify()
            variables = list(sub_bool_func.get_variable_names())
        variables.sort()  # we do this to have a single order between inputs. sometimes the order of the list got mixed

        # saving the results (in the most generic way possible haha)
        bool_func_dict[gate.get_name()] = (sub_bool_func, variables)

        calc_cv(variables, sub_bool_func, gate.get_name())

    print("Performing lil bro begg Iteration...")
    for gate in filtered_gates:

        # gets output wires
        fanout_net = gate.get_fan_out_nets()
        # calculates sub gates (gates that are influencing the output of the current output net) with dfs
        sub_gates = gate

        # calculates bool func, variables and control values
        sub_bool_func = hal_py.NetlistUtils.get_subgraph_function(fanout_net[0], [sub_gates])
        variables = list(sub_bool_func.get_variable_names())
        # simplifies the bool func if there are more than 100 variables in the original func
        if len(variables) >= 100:
            sub_bool_func = sub_bool_func.simplify()
            variables = list(sub_bool_func.get_variable_names())
        variables.sort()  # we do this to have a single order between inputs. sometimes the order of the list got mixed

        wires = dfs_from_net(fanout_net[0], args.dfs_depth)
        print(wires)
        print(wire_dict)

        control_vector = []
        for var in variables:
            control_vector.append(wires[var])

        print(f'CONTROL VECTOR: {control_vector}')

        # calculating mean and median of the control value vector
        mean_cv = mean(control_vector)
        median_cv = median(control_vector)
        print(f'MEAN: {mean_cv}')
        print(f'MEDIAN: {median_cv}')

        # building the structure of our table, and adding values we already calculated
        df = pd.concat([df, pd.DataFrame({'GATE_NAME': gate.get_name(), 'GATE_TYPE': gate.get_type().get_name(),
                                          'BOOL_FUNC': np.nan, 'VARIABLES': str(variables),
                                          'CONTROL_VECTOR': str(control_vector), 'MEAN': mean_cv,
                                          'MEDIAN': median_cv, 'SIZE': len(variables)}, index=[len(df)])])
        df.to_csv(csv_path)


