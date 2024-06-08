import os

"""
    A mapping of all the trojan gates in netlists that contain a trojan horse

    We currently support the following netlists from trust-hub:
    - In the RS232 family:
        * RS232-T1100-uart.v
        * RS232-T1200-uart.v
        * RS232-T1300-uart.v
        * RS232-T1400-uart.v
        * RS232-T1500-uart.v
        * RS232-T1600-uart.v
    - In the s15850 family:
        * s15850-T100_scan_in.v
    - In the s35932 family:
        * s35932-T100_scan_in.v
        * s35932-T200_scan_in.v
        * s35932-T300_scan_in.v
    - In the s38417 family:
        * s38417-T100_scan_in.v
    - In the s38584 family:
        * s38584-T100_scan_in.v
        * s38584-T200_scan_in.v

    todo: add support to many of these as possible
    We are yet to support:
    - In the s38584 family:
        * s38584-T300_scan_in.v (check multiple references of the same gate in different modules)
    - vga_lcd-T100 (after we try to add multi-processing support)
    - wb_conmax-T100 (after we try to add multi-processing support)
    - B15-T100 (after we try to add multi-processing support)
    - B15-T200 (after we try to add multi-processing support)
    - B15-T400 (after we try to add multi-processing support)
"""
trojan_mapping = {
    # RS323 Family
    "RS232-T1100-uart.v": ("U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U305"),
    "RS232-T1200-uart.v": ("U292", "U293", "U294", "U295", "U296", "U297", "U300", "U301", "U302", "U303"),
    "RS232-T1300-uart.v": ("U292", "U293", "U294", "U295", "U296", "U297", "U302", "U303", "U304"),
    "RS232-T1400-uart.v": ("U292", "U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303"),
    "RS232-T1500-uart.v": ("U293", "U294", "U295", "U296", "U297", "U298", "U299", "U300", "U301", "U302", "U303", "U304", "U305"),
    "RS232-T1600-uart.v": ("U293", "U294", "U295", "U296", "U297", "U300", "U301", "U302", "U303", "U304"),

    # s15850 Family
    "s15850-T100_scan_in.v": ("Tg1_Trojan1", "Tg1_Trojan2", "Tg1_Trojan3", "Tg1_Trojan4", "Tg1_Trojan1234",
                         "Tg1_Trojan5", "Tg1_Trojan6", "Tg1_Trojan7", "Tg1_Trojan8", "Tg1_Trojan5678", "Tg1_Tj_Trigger",
                         "Tg2_Trojan1", "Tg2_Trojan2", "Tg2_Trojan3", "Tg2_Trojan4", "Tg2_Trojan1234",
                         "Tg2_Trojan5", "Tg2_Trojan6", "Tg2_Trojan7", "Tg2_Trojan8", "Tg2_Trojan5678", "Tg2_Tj_Trigger",
                         "INVtest_se", "Trojan_Trigger", "Trojan_Payload"),

    # s35932 Family
    "s35932-T100_scan_in.v": ("Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT", "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
                              "INV_test_se", "Trojan_Trigger", "Trojan_Payload1", "Trojan_Payload2"),
    "s35932-T200_scan_in.v": ("Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT", "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
                              "INVtest_se", "Trojan_Trigger", "U5548", "U5566", "U6740", "U6802"),
    "s35932-T300_scan_in.v": ("Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT", "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
                              "INVtest_se", "Trojan_Trigger", "TjPayload1", "TjPayload2", "TjPayload3", "TjPayload4", "TjPayload5", "TjPayload6",
                              "TjPayload7", "TjPayload8", "TjPayload9", "TjPayload10", "TjPayload11", "TjPayload12", "TjPayload13", "TjPayload14", "TjPayload15",
                              "TjPayload16", "TjPayload17", "TjPayload18", "TjPayload19", "TjPayload20", "TjPayload21", "TjPayload22", "TjPayload23", "TjPayload24"),

    # s38417 Family
    "s38417-T100_scan_in.v": ("Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT", "Trojan5", "Trojan6", "Trojan7", "Trojan8", "Trojan5678_NOT",
                         "Trojan_CLK_NOT", "Trojan_Payload"),

    # s38584 Family
    "s38584-T100_scan_in.v": ("Trojan1", "Trojan2", "Trojan3", "Trojan4", "Trojan1234_NOT",
                              "Trojan5", "NOT_test_se", "Trojan_Trigger", "Trojan_Payload"),
    "s38584-T200_scan_in.v": ("Trojan_U1", "Trojan_U2",
                              "Trojan_U1_1_1", "Trojan_U1_1_2", "Trojan_U1_1_3", "Trojan_U1_1_4", "Trojan_U1_1_5",
                              "Trojan_U1_1_6", "Trojan_U1_1_7", "Trojan_U1_1_8", "Trojan_U1_1_9", "Trojan_U1_1_10",
                              "Trojan_U1_1_11", "Trojan_U1_1_12", "Trojan_U1_1_13", "Trojan_U1_1_14", "Trojan_U1_1_15",
                              "Trojan_U1_1_16", "Trojan_U1_1_17", "Trojan_U1_1_18", "Trojan_U1_1_19", "Trojan_U1_1_20"
                              "Trojan_U1_1_21", "Trojan_U1_1_22", "Trojan_U1_1_23", "Trojan_U1_1_24", "Trojan_U1_1_25",
                              "Trojan_U1_1_26", "Trojan_U1_1_27", "Trojan_U1_1_28", "Trojan_U1_1_29", "Trojan_U1_1_30",
                              "Trojan_U14", "Trojan_U15", "Trojan_U16", "Trojan_U17", "Trojan_U18", "Trojan_U19", "Trojan_U20",
                              "Trojan_U21", "Trojan_U22", "Trojan_U23", "Trojan_U24", "Trojan_U25", "Trojan_U26", "Trojan_U27",
                              "Trojan_U28", "Trojan_U29", "Trojan_U30", "Trojan_U31",
                              "TrojanTrigger", "Trojan_Paylod"),
}


def is_netlist_in_mapping(netlist_path):
    """
    A function for determining if a netlist is mapped inside the trojan_mapping dict (checks if its located in trojan_mapping.keys())
        * if the netlist satisfies the condition above, it is a netlist with a trojan horse. if not, we cannot determine if it has a trojan horse or not
    :param netlist_path: the path of the netlist we want to check (can be a local or a global path)
    :return: True if netlist is mapped inside trojan_mapping, else False.
    """
    return os.path.basename(netlist_path) in trojan_mapping.keys()


def is_trojan_gate(gate_name, netlist_path):
    """
    A function for determining if a gate is a trojan gate or not in a given netlist.
    * We "assert" that the base name of the given netlist_path is a key in trojan_mapping
    :param gate: the name of the gate instance (according to the netlist file)
    :param netlist_path: the path of the netlist we want to check the condition on (can be a local or a global path)
    :return: True if gate is inside trojan_mapping[netlist base name].values, else False
    """
    return gate_name in trojan_mapping[os.path.basename(netlist_path)]

