/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:09:35 2023
/////////////////////////////////////////////////////////////


module b12 ( clock, reset, start, k, nloss, nl, speaker );
  input [3:0] k;
  output [3:0] nl;
  input clock, reset, start;
  output nloss, speaker;
  wire   play, sound_2_, N201, N202, N203, N208, N209, N210, N615, N616, N617,
         N793, N794, N795, N886, N887, N888, N889, N890, U3_U3_Z_0, U3_U3_Z_5,
         n493, n495, n500, n501, n508, n509, n516, n517, n518, n519, n520,
         n521, n522, n523, n524, n525, n526, n527, n528, n529, n530, n531,
         n532, n533, n534, n535, n536, n537, n538, n539, n540, n541, n542,
         n543, n544, n545, n546, n547, n548, n549, n550, n551, n552, n553,
         n554, n555, n556, n557, n558, n559, n560, n561, n562, n563, n564,
         n565, n566, n567, n568, n569, n570, n571, n572, n573, n574, n575,
         n576, n577, n578, n579, n580, n581, n582, n583, n584, n585, n586,
         n587, n588, n589, n590, n591, n592, n593, n594, n595, n596, n597,
         n598, n599, n600, n601, n602, n603, n604, n605, n606, n607, n608,
         n609, n610, n611, n612, n613, n614, n615, n616, n617, n618, n619,
         n620, n621, n622, n624, r271_carry_2_, r271_carry_3_, r271_carry_4_,
         r271_SUM_1_, r271_SUM_2_, r271_SUM_3_, r271_SUM_4_, r271_SUM_5_, n625,
         n626, n627, n628, n629, n630, n631, n632, n633, n634, n635, n636,
         n637, n638, n639, n640, n641, n642, n643, n644, n645, n646, n647,
         n648, n649, n650, n651, n652, n653, n654, n655, n656, n657, n658,
         n659, n660, n661, n662, n663, n664, n665, n666, n667, n668, n669,
         n670, n671, n672, n673, n674, n675, n676, n677, n678, n679, n680,
         n681, n682, n683, n684, n685, n686, n687, n688, n689, n690, n691,
         n692, n693, n694, n695, n696, n697, n698, n699, n700, n701, n702,
         n703, n704, n705, n706, n707, n708, n709, n710, n711, n712, n713,
         n714, n715, n716, n717, n718, n719, n720, n721, n722, n723, n724,
         n725, n726, n727, n728, n729, n730, n731, n732, n733, n734, n735,
         n736, n737, n738, n739, n740, n741, n742, n743, n744, n745, n746,
         n747, n748, n749, n750, n751, n752, n753, n754, n755, n756, n757,
         n758, n759, n760, n761, n762, n763, n764, n765, n766, n767, n768,
         n769, n770, n771, n772, n773, n774, n775, n776, n777, n778, n779,
         n780, n781, n782, n783, n784, n785, n786, n787, n788, n789, n790,
         n791, n792, n793, n794, n795, n796, n797, n798, n799, n800, n801,
         n802, n803, n804, n805, n806, n807, n808, n809, n810, n811, n812,
         n813, n814, n815, n816, n817, n818, n819, n820, n821, n822, n823,
         n824, n825, n826, n827, n828, n829, n830, n831, n832, n833, n834,
         n835, n836, n837, n838, n839, n840, n841, n842, n843, n844, n845,
         n846, n847, n848, n849, n850, n851, n852, n853, n854, n855, n856,
         n857, n858, n859, n860, n861, n862, n863, n864, n865, n866, n867,
         n868, n869, n870, n871, n872, n873, n874, n875, n876, n877, n878,
         n879, n880, n881, n882, n883, n884, n885, n886, n887, n888, n889,
         n890, n891, n892, n893, n894, n895, n896, n897, n898, n899, n900,
         n901, n902, n903, n904, n905, n906, n907, n908, n909, n910, n911,
         n912, n913, n914, n915, n916, n917, n918, n919, n920, n921, n922,
         n923, n924, n925, n926, n927, n928, n929, n930, n931, n932, n933,
         n934, n935, n936, n937, n938, n939, n940, n941, n942, n943, n944,
         n945, n946, n947, n948, n949, n950, n951, n952, n953, n954, n955,
         n956, n957, n958, n959, n960, n961, n962, n963, n964, n965, n966,
         n967, n968, n969, n970, n971, n972, n973, n974, n975, n976, n977,
         n978, n979, n980, n981, n982, n983, n984, n985, n986, n987, n988,
         n989, n990, n991, n992, n993, n994, n995, n996, n997, n998, n999,
         n1000, n1001, n1002, n1003, n1004, n1005, n1006, n1007, n1008, n1009,
         n1010, n1011, n1012, n1013, n1014, n1015, n1016, n1017, n1018, n1019,
         n1020, n1021, n1022, n1023, n1024, n1025, n1026, n1027, n1028, n1029,
         n1030, n1031, n1032, n1033, n1034, n1035, n1036, n1037, n1038, n1039,
         n1040, n1041, n1042, n1043, n1044, n1045, n1046, n1047, n1048, n1049,
         n1050, n1051, n1052, n1053, n1054, n1055, n1056, n1057, n1058, n1059,
         n1060, n1061, n1062, n1063, n1064, n1065, n1066, n1067, n1068, n1069,
         n1070, n1071;
  wire   [4:0] scan;
  wire   [3:0] max;
  wire   [5:0] timebase;
  wire   [4:2] r330_carry;
  wire   [4:2] r336_carry;

  DFFARX1 count_reg_0_ ( .D(n624), .CLK(clock), .RSTB(n647), .Q(n1042), .QN(
        n624) );
  DFFARX1 num_reg_1_ ( .D(N208), .CLK(clock), .RSTB(n649), .Q(n495) );
  DFFARX1 count_reg_1_ ( .D(N208), .CLK(clock), .RSTB(n650), .Q(n1043) );
  DFFARX1 num_reg_0_ ( .D(n624), .CLK(clock), .RSTB(n648), .Q(n493) );
  DFFARX1 gamma_reg_0_ ( .D(N886), .CLK(clock), .RSTB(n648), .Q(n1064), .QN(
        n634) );
  DFFARX1 gamma_reg_4_ ( .D(N890), .CLK(clock), .RSTB(n648), .Q(n1063), .QN(
        n625) );
  DFFARX1 count_reg2_3_ ( .D(n622), .CLK(clock), .RSTB(n650), .Q(n1061) );
  DFFARX1 count_reg2_5_ ( .D(n621), .CLK(clock), .RSTB(n648), .Q(n1060) );
  DFFARX1 gamma_reg_2_ ( .D(N888), .CLK(clock), .RSTB(n648), .Q(n1066), .QN(
        n627) );
  DFFARX1 gamma_reg_1_ ( .D(N887), .CLK(clock), .RSTB(n647), .Q(n1065), .QN(
        n635) );
  DFFARX1 max_reg_1_ ( .D(n620), .CLK(clock), .RSTB(n649), .Q(max[1]), .QN(
        n501) );
  DFFARX1 max_reg_0_ ( .D(n619), .CLK(clock), .RSTB(n647), .Q(max[0]), .QN(
        n500) );
  DFFARX1 max_reg_2_ ( .D(n618), .CLK(clock), .RSTB(n648), .Q(max[2]) );
  DFFARX1 max_reg_3_ ( .D(n617), .CLK(clock), .RSTB(n649), .Q(max[3]), .QN(
        n508) );
  DFFARX1 max_reg_4_ ( .D(n616), .CLK(clock), .RSTB(n650), .Q(n644), .QN(n509)
         );
  DFFARX1 gamma_reg_3_ ( .D(N889), .CLK(clock), .RSTB(n647), .Q(n1062), .QN(
        n645) );
  DFFARX1 ind_reg_1_ ( .D(n615), .CLK(clock), .RSTB(n649), .Q(n1059), .QN(n630) );
  DFFARX1 ind_reg_0_ ( .D(n614), .CLK(clock), .RSTB(n650), .Q(n1058), .QN(n640) );
  DFFARX1 wr_reg ( .D(n613), .CLK(clock), .RSTB(n647), .Q(n636) );
  DFFARX1 data_in_reg_1_ ( .D(n612), .CLK(clock), .RSTB(n648), .Q(n1057) );
  DFFARX1 data_in_reg_0_ ( .D(n611), .CLK(clock), .RSTB(n647), .Q(n1056) );
  DFFARX1 scan_reg_0_ ( .D(n610), .CLK(clock), .RSTB(n647), .Q(scan[0]) );
  DFFARX1 scan_reg_4_ ( .D(n609), .CLK(clock), .RSTB(n650), .Q(scan[4]) );
  DFFARX1 scan_reg_3_ ( .D(n608), .CLK(clock), .RSTB(n649), .Q(scan[3]) );
  DFFARX1 address_reg_3_ ( .D(n607), .CLK(clock), .RSTB(n647), .Q(n1055), .QN(
        n641) );
  DFFARX1 scan_reg_2_ ( .D(n606), .CLK(clock), .RSTB(n648), .Q(scan[2]) );
  DFFARX1 address_reg_2_ ( .D(n605), .CLK(clock), .RSTB(n647), .Q(n1054), .QN(
        n629) );
  DFFARX1 scan_reg_1_ ( .D(n604), .CLK(clock), .RSTB(n650), .Q(scan[1]) );
  DFFARX1 sound_reg_2_ ( .D(n603), .CLK(clock), .RSTB(n647), .Q(sound_2_), 
        .QN(n646) );
  DFFARX1 play_reg ( .D(n602), .CLK(clock), .RSTB(n650), .Q(play), .QN(n642)
         );
  DFFARX1 address_reg_4_ ( .D(n601), .CLK(clock), .RSTB(n649), .Q(n1053), .QN(
        n632) );
  DFFARX1 address_reg_0_ ( .D(n600), .CLK(clock), .RSTB(n650), .Q(n1052), .QN(
        n637) );
  DFFARX1 address_reg_1_ ( .D(n599), .CLK(clock), .RSTB(n649), .Q(n1051) );
  DFFARX1 memory_reg_26__0_ ( .D(n598), .CLK(clock), .RSTB(n650), .Q(n984) );
  DFFARX1 memory_reg_26__1_ ( .D(n597), .CLK(clock), .RSTB(n647), .Q(n1016) );
  DFFARX1 memory_reg_18__0_ ( .D(n596), .CLK(clock), .RSTB(n647), .Q(n992) );
  DFFARX1 memory_reg_18__1_ ( .D(n595), .CLK(clock), .RSTB(n650), .Q(n1024) );
  DFFARX1 memory_reg_10__0_ ( .D(n594), .CLK(clock), .RSTB(n650), .Q(n1000) );
  DFFARX1 memory_reg_10__1_ ( .D(n593), .CLK(clock), .RSTB(n647), .Q(n1032) );
  DFFARX1 memory_reg_2__0_ ( .D(n592), .CLK(clock), .RSTB(n650), .Q(n1008) );
  DFFARX1 memory_reg_2__1_ ( .D(n591), .CLK(clock), .RSTB(n647), .Q(n1040) );
  DFFARX1 memory_reg_27__0_ ( .D(n590), .CLK(clock), .RSTB(n648), .Q(n985) );
  DFFARX1 memory_reg_27__1_ ( .D(n589), .CLK(clock), .RSTB(n649), .Q(n1017) );
  DFFARX1 memory_reg_19__0_ ( .D(n588), .CLK(clock), .RSTB(n647), .Q(n993) );
  DFFARX1 memory_reg_19__1_ ( .D(n587), .CLK(clock), .RSTB(n648), .Q(n1025) );
  DFFARX1 memory_reg_11__0_ ( .D(n586), .CLK(clock), .RSTB(n648), .Q(n1001) );
  DFFARX1 memory_reg_11__1_ ( .D(n585), .CLK(clock), .RSTB(n649), .Q(n1033) );
  DFFARX1 memory_reg_3__0_ ( .D(n584), .CLK(clock), .RSTB(n648), .Q(n1009) );
  DFFARX1 memory_reg_3__1_ ( .D(n583), .CLK(clock), .RSTB(n649), .Q(n1041) );
  DFFARX1 memory_reg_31__0_ ( .D(n582), .CLK(clock), .RSTB(n650), .Q(n978) );
  DFFARX1 memory_reg_31__1_ ( .D(n581), .CLK(clock), .RSTB(n647), .Q(n1010) );
  DFFARX1 memory_reg_23__0_ ( .D(n580), .CLK(clock), .RSTB(n649), .Q(n986) );
  DFFARX1 memory_reg_23__1_ ( .D(n579), .CLK(clock), .RSTB(n650), .Q(n1018) );
  DFFARX1 memory_reg_15__0_ ( .D(n578), .CLK(clock), .RSTB(n650), .Q(n994) );
  DFFARX1 memory_reg_15__1_ ( .D(n577), .CLK(clock), .RSTB(n647), .Q(n1026) );
  DFFARX1 memory_reg_7__0_ ( .D(n576), .CLK(clock), .RSTB(n650), .Q(n1002) );
  DFFARX1 memory_reg_7__1_ ( .D(n575), .CLK(clock), .RSTB(n647), .Q(n1034) );
  DFFARX1 memory_reg_30__0_ ( .D(n574), .CLK(clock), .RSTB(n648), .Q(n979) );
  DFFARX1 memory_reg_30__1_ ( .D(n573), .CLK(clock), .RSTB(n649), .Q(n1011) );
  DFFARX1 memory_reg_22__0_ ( .D(n572), .CLK(clock), .RSTB(n647), .Q(n987) );
  DFFARX1 memory_reg_22__1_ ( .D(n571), .CLK(clock), .RSTB(n648), .Q(n1019) );
  DFFARX1 memory_reg_14__0_ ( .D(n570), .CLK(clock), .RSTB(n648), .Q(n995) );
  DFFARX1 memory_reg_14__1_ ( .D(n569), .CLK(clock), .RSTB(n649), .Q(n1027) );
  DFFARX1 memory_reg_6__0_ ( .D(n568), .CLK(clock), .RSTB(n648), .Q(n1003) );
  DFFARX1 memory_reg_6__1_ ( .D(n567), .CLK(clock), .RSTB(n649), .Q(n1035) );
  DFFARX1 memory_reg_25__0_ ( .D(n566), .CLK(clock), .RSTB(n650), .Q(n982) );
  DFFARX1 memory_reg_25__1_ ( .D(n565), .CLK(clock), .RSTB(n647), .Q(n1014) );
  DFFARX1 memory_reg_17__0_ ( .D(n564), .CLK(clock), .RSTB(n649), .Q(n990) );
  DFFARX1 memory_reg_17__1_ ( .D(n563), .CLK(clock), .RSTB(n650), .Q(n1022) );
  DFFARX1 memory_reg_9__0_ ( .D(n562), .CLK(clock), .RSTB(n650), .Q(n998) );
  DFFARX1 memory_reg_9__1_ ( .D(n561), .CLK(clock), .RSTB(n647), .Q(n1030) );
  DFFARX1 memory_reg_1__0_ ( .D(n560), .CLK(clock), .RSTB(n650), .Q(n1006) );
  DFFARX1 memory_reg_1__1_ ( .D(n559), .CLK(clock), .RSTB(n647), .Q(n1038) );
  DFFARX1 memory_reg_24__0_ ( .D(n558), .CLK(clock), .RSTB(n648), .Q(n983) );
  DFFARX1 memory_reg_24__1_ ( .D(n557), .CLK(clock), .RSTB(n649), .Q(n1015) );
  DFFARX1 memory_reg_16__0_ ( .D(n556), .CLK(clock), .RSTB(n647), .Q(n991) );
  DFFARX1 memory_reg_16__1_ ( .D(n555), .CLK(clock), .RSTB(n648), .Q(n1023) );
  DFFARX1 memory_reg_8__0_ ( .D(n554), .CLK(clock), .RSTB(n648), .Q(n999) );
  DFFARX1 memory_reg_8__1_ ( .D(n553), .CLK(clock), .RSTB(n649), .Q(n1031) );
  DFFARX1 memory_reg_0__0_ ( .D(n552), .CLK(clock), .RSTB(n648), .Q(n1007) );
  DFFARX1 memory_reg_0__1_ ( .D(n551), .CLK(clock), .RSTB(n649), .Q(n1039) );
  DFFARX1 memory_reg_28__0_ ( .D(n550), .CLK(clock), .RSTB(n650), .Q(n980) );
  DFFARX1 memory_reg_28__1_ ( .D(n549), .CLK(clock), .RSTB(n647), .Q(n1012) );
  DFFARX1 memory_reg_20__0_ ( .D(n548), .CLK(clock), .RSTB(n649), .Q(n988) );
  DFFARX1 memory_reg_20__1_ ( .D(n547), .CLK(clock), .RSTB(n650), .Q(n1020) );
  DFFARX1 memory_reg_12__0_ ( .D(n546), .CLK(clock), .RSTB(n650), .Q(n996) );
  DFFARX1 memory_reg_12__1_ ( .D(n545), .CLK(clock), .RSTB(n647), .Q(n1028) );
  DFFARX1 memory_reg_4__0_ ( .D(n544), .CLK(clock), .RSTB(n650), .Q(n1004) );
  DFFARX1 memory_reg_4__1_ ( .D(n543), .CLK(clock), .RSTB(n647), .Q(n1036) );
  DFFARX1 memory_reg_29__0_ ( .D(n542), .CLK(clock), .RSTB(n648), .Q(n981) );
  DFFARX1 memory_reg_29__1_ ( .D(n541), .CLK(clock), .RSTB(n649), .Q(n1013) );
  DFFARX1 memory_reg_21__0_ ( .D(n540), .CLK(clock), .RSTB(n647), .Q(n989) );
  DFFARX1 memory_reg_21__1_ ( .D(n539), .CLK(clock), .RSTB(n648), .Q(n1021) );
  DFFARX1 memory_reg_13__0_ ( .D(n538), .CLK(clock), .RSTB(n648), .Q(n997) );
  DFFARX1 memory_reg_13__1_ ( .D(n537), .CLK(clock), .RSTB(n649), .Q(n1029) );
  DFFARX1 memory_reg_5__0_ ( .D(n536), .CLK(clock), .RSTB(n650), .Q(n1005) );
  DFFARX1 data_out_reg_0_ ( .D(N210), .CLK(clock), .RSTB(n649), .Q(n1050), 
        .QN(n638) );
  DFFARX1 sound_reg_0_ ( .D(n535), .CLK(clock), .RSTB(n648), .Q(n639), .QN(
        n1070) );
  DFFARX1 memory_reg_5__1_ ( .D(n534), .CLK(clock), .RSTB(n648), .Q(n1037) );
  DFFARX1 data_out_reg_1_ ( .D(N209), .CLK(clock), .RSTB(n649), .Q(n1049), 
        .QN(n628) );
  DFFARX1 nl_reg_2_ ( .D(n533), .CLK(clock), .RSTB(n647), .Q(nl[2]) );
  DFFARX1 nl_reg_3_ ( .D(n532), .CLK(clock), .RSTB(n648), .Q(nl[3]) );
  DFFARX1 sound_reg_1_ ( .D(n531), .CLK(clock), .RSTB(n649), .Q(n626), .QN(
        n1071) );
  DFFARX1 counter_reg_0_ ( .D(N201), .CLK(clock), .RSTB(n649), .Q(n1069), .QN(
        n633) );
  DFFARX1 counter_reg_1_ ( .D(N202), .CLK(clock), .RSTB(n648), .Q(n1068), .QN(
        n643) );
  DFFARX1 counter_reg_2_ ( .D(N203), .CLK(clock), .RSTB(n647), .Q(n1067), .QN(
        n631) );
  DFFARX1 s_reg ( .D(n530), .CLK(clock), .RSTB(n650), .QN(n1048) );
  DFFARX1 speaker_reg ( .D(n529), .CLK(clock), .RSTB(n649), .Q(speaker) );
  DFFARX1 nl_reg_0_ ( .D(n528), .CLK(clock), .RSTB(n648), .Q(nl[0]) );
  DFFARX1 nl_reg_1_ ( .D(n527), .CLK(clock), .RSTB(n649), .Q(nl[1]) );
  DFFARX1 nloss_reg ( .D(n526), .CLK(clock), .RSTB(n650), .Q(nloss) );
  DFFARX1 timebase_reg_5_ ( .D(n525), .CLK(clock), .RSTB(n649), .Q(timebase[5]) );
  DFFARX1 timebase_reg_4_ ( .D(n524), .CLK(clock), .RSTB(n650), .Q(timebase[4]) );
  DFFARX1 count_reg2_4_ ( .D(n523), .CLK(clock), .RSTB(n647), .Q(n1047) );
  DFFARX1 timebase_reg_3_ ( .D(n522), .CLK(clock), .RSTB(n648), .Q(timebase[3]) );
  DFFARX1 timebase_reg_2_ ( .D(n521), .CLK(clock), .RSTB(n649), .Q(timebase[2]) );
  DFFARX1 count_reg2_2_ ( .D(n520), .CLK(clock), .RSTB(n650), .Q(n1046) );
  DFFARX1 timebase_reg_1_ ( .D(n519), .CLK(clock), .RSTB(n647), .Q(timebase[1]) );
  DFFARX1 count_reg2_1_ ( .D(n518), .CLK(clock), .RSTB(n648), .Q(n1045) );
  DFFARX1 timebase_reg_0_ ( .D(n517), .CLK(clock), .RSTB(n649), .Q(timebase[0]) );
  DFFARX1 count_reg2_0_ ( .D(n516), .CLK(clock), .RSTB(n650), .Q(n1044) );
  HADDX1 r330_U1_1_1 ( .A0(scan[1]), .B0(scan[0]), .C1(r330_carry[2]), .SO(
        N615) );
  HADDX1 r330_U1_1_2 ( .A0(scan[2]), .B0(r330_carry[2]), .C1(r330_carry[3]), 
        .SO(N616) );
  HADDX1 r330_U1_1_3 ( .A0(scan[3]), .B0(r330_carry[3]), .C1(r330_carry[4]), 
        .SO(N617) );
  HADDX1 r336_U1_1_1 ( .A0(max[1]), .B0(max[0]), .C1(r336_carry[2]), .SO(N793)
         );
  HADDX1 r336_U1_1_2 ( .A0(max[2]), .B0(r336_carry[2]), .C1(r336_carry[3]), 
        .SO(N794) );
  HADDX1 r336_U1_1_3 ( .A0(max[3]), .B0(r336_carry[3]), .C1(r336_carry[4]), 
        .SO(N795) );
  INVX0 U513 ( .INP(r271_carry_4_), .ZN(n656) );
  NOR4X1 U514 ( .IN1(n1060), .IN2(n1047), .IN3(n1061), .IN4(n920), .QN(n661)
         );
  INVX2 U515 ( .INP(reset), .ZN(n647) );
  INVX2 U516 ( .INP(reset), .ZN(n648) );
  INVX2 U517 ( .INP(reset), .ZN(n649) );
  INVX2 U518 ( .INP(reset), .ZN(n650) );
  XOR2X1 U519 ( .IN1(U3_U3_Z_5), .IN2(n651), .Q(r271_SUM_5_) );
  AND2X1 U520 ( .IN1(n656), .IN2(n655), .Q(n651) );
  NAND2X1 U521 ( .IN1(n832), .IN2(n652), .QN(r271_carry_2_) );
  NAND2X1 U522 ( .IN1(n658), .IN2(n653), .QN(r271_carry_3_) );
  INVX0 U523 ( .INP(r271_carry_2_), .ZN(n658) );
  NAND2X1 U524 ( .IN1(n657), .IN2(n654), .QN(r271_carry_4_) );
  INVX0 U525 ( .INP(r271_carry_3_), .ZN(n657) );
  AOI22X1 U526 ( .IN1(n845), .IN2(timebase[1]), .IN3(n1045), .IN4(n846), .QN(
        n652) );
  AOI22X1 U527 ( .IN1(n845), .IN2(timebase[2]), .IN3(n1046), .IN4(n846), .QN(
        n653) );
  AOI22X1 U528 ( .IN1(n845), .IN2(timebase[3]), .IN3(n1061), .IN4(n846), .QN(
        n654) );
  AOI22X1 U529 ( .IN1(n845), .IN2(timebase[4]), .IN3(n1047), .IN4(n846), .QN(
        n655) );
  XOR2X1 U530 ( .IN1(n655), .IN2(r271_carry_4_), .Q(r271_SUM_4_) );
  XOR2X1 U531 ( .IN1(n654), .IN2(r271_carry_3_), .Q(r271_SUM_3_) );
  XOR2X1 U532 ( .IN1(n653), .IN2(r271_carry_2_), .Q(r271_SUM_2_) );
  XOR2X1 U533 ( .IN1(n652), .IN2(U3_U3_Z_0), .Q(r271_SUM_1_) );
  MUX21X1 U534 ( .IN1(n659), .IN2(n1061), .S(n660), .Q(n622) );
  AO222X1 U535 ( .IN1(n661), .IN2(n662), .IN3(r271_SUM_3_), .IN4(n663), .IN5(
        timebase[3]), .IN6(n664), .Q(n659) );
  NAND2X0 U536 ( .IN1(n665), .IN2(n666), .QN(n663) );
  MUX21X1 U537 ( .IN1(n667), .IN2(n1060), .S(n660), .Q(n621) );
  AO221X1 U538 ( .IN1(timebase[5]), .IN2(n664), .IN3(r271_SUM_5_), .IN4(n668), 
        .IN5(n669), .Q(n667) );
  AO22X1 U539 ( .IN1(n670), .IN2(max[1]), .IN3(N793), .IN4(n671), .Q(n620) );
  MUX21X1 U540 ( .IN1(n670), .IN2(n671), .S(n500), .Q(n619) );
  AO22X1 U541 ( .IN1(n670), .IN2(max[2]), .IN3(N794), .IN4(n671), .Q(n618) );
  AO22X1 U542 ( .IN1(n670), .IN2(max[3]), .IN3(N795), .IN4(n671), .Q(n617) );
  MUX21X1 U543 ( .IN1(n672), .IN2(n673), .S(n509), .Q(n616) );
  AND2X1 U544 ( .IN1(n671), .IN2(r336_carry[4]), .Q(n673) );
  AO21X1 U545 ( .IN1(n671), .IN2(n674), .IN3(n670), .Q(n672) );
  INVX0 U546 ( .INP(r336_carry[4]), .ZN(n674) );
  NOR2X0 U547 ( .IN1(n670), .IN2(n675), .QN(n671) );
  AND3X1 U548 ( .IN1(n676), .IN2(n677), .IN3(n678), .Q(n670) );
  NAND2X0 U549 ( .IN1(n661), .IN2(n679), .QN(n676) );
  AO21X1 U550 ( .IN1(n680), .IN2(n681), .IN3(n682), .Q(n679) );
  NAND2X0 U551 ( .IN1(n683), .IN2(n684), .QN(n615) );
  NAND3X0 U552 ( .IN1(n685), .IN2(n686), .IN3(n687), .QN(n684) );
  AO21X1 U553 ( .IN1(n687), .IN2(n688), .IN3(n630), .Q(n683) );
  AO22X1 U554 ( .IN1(n1058), .IN2(n689), .IN3(n690), .IN4(n687), .Q(n614) );
  NOR2X0 U555 ( .IN1(k[0]), .IN2(n691), .QN(n690) );
  OAI21X1 U556 ( .IN1(k[0]), .IN2(k[2]), .IN3(n687), .QN(n689) );
  NAND2X0 U557 ( .IN1(n692), .IN2(n693), .QN(n613) );
  NAND3X0 U558 ( .IN1(n694), .IN2(n636), .IN3(n695), .QN(n693) );
  INVX0 U559 ( .INP(n696), .ZN(n692) );
  MUX21X1 U560 ( .IN1(n1057), .IN2(n495), .S(n696), .Q(n612) );
  MUX21X1 U561 ( .IN1(n1056), .IN2(n493), .S(n696), .Q(n611) );
  MUX21X1 U562 ( .IN1(n697), .IN2(n698), .S(scan[0]), .Q(n610) );
  MUX21X1 U563 ( .IN1(n699), .IN2(n700), .S(scan[4]), .Q(n609) );
  AO21X1 U564 ( .IN1(n697), .IN2(n701), .IN3(n698), .Q(n700) );
  NOR2X0 U565 ( .IN1(n702), .IN2(n701), .QN(n699) );
  INVX0 U566 ( .INP(r330_carry[4]), .ZN(n701) );
  AO22X1 U567 ( .IN1(n698), .IN2(scan[3]), .IN3(N617), .IN4(n697), .Q(n608) );
  AO222X1 U568 ( .IN1(n703), .IN2(scan[3]), .IN3(n1055), .IN4(n704), .IN5(n705), .IN6(max[3]), .Q(n607) );
  AO22X1 U569 ( .IN1(n698), .IN2(scan[2]), .IN3(N616), .IN4(n697), .Q(n606) );
  AO222X1 U570 ( .IN1(n703), .IN2(scan[2]), .IN3(n1054), .IN4(n704), .IN5(n705), .IN6(max[2]), .Q(n605) );
  AO22X1 U571 ( .IN1(n698), .IN2(scan[1]), .IN3(N615), .IN4(n697), .Q(n604) );
  INVX0 U572 ( .INP(n702), .ZN(n697) );
  NAND2X0 U573 ( .IN1(n706), .IN2(n707), .QN(n702) );
  AO21X1 U574 ( .IN1(n708), .IN2(n681), .IN3(n709), .Q(n706) );
  INVX0 U575 ( .INP(n707), .ZN(n698) );
  AO21X1 U576 ( .IN1(n661), .IN2(n710), .IN3(n696), .Q(n707) );
  AO21X1 U577 ( .IN1(n709), .IN2(n681), .IN3(n708), .Q(n710) );
  MUX21X1 U578 ( .IN1(sound_2_), .IN2(n711), .S(n712), .Q(n603) );
  MUX21X1 U579 ( .IN1(n713), .IN2(play), .S(n714), .Q(n602) );
  NOR2X0 U580 ( .IN1(n712), .IN2(n715), .QN(n714) );
  NAND3X0 U581 ( .IN1(n716), .IN2(n717), .IN3(n718), .QN(n713) );
  AO222X1 U582 ( .IN1(n703), .IN2(scan[4]), .IN3(n1053), .IN4(n704), .IN5(n705), .IN6(n644), .Q(n601) );
  AO222X1 U583 ( .IN1(n703), .IN2(scan[0]), .IN3(n1052), .IN4(n704), .IN5(n705), .IN6(max[0]), .Q(n600) );
  AO222X1 U584 ( .IN1(n703), .IN2(scan[1]), .IN3(n1051), .IN4(n704), .IN5(n705), .IN6(max[1]), .Q(n599) );
  NOR2X0 U585 ( .IN1(n705), .IN2(n703), .QN(n704) );
  AND4X1 U586 ( .IN1(n719), .IN2(n720), .IN3(n721), .IN4(n627), .Q(n703) );
  NOR2X0 U587 ( .IN1(start), .IN2(n1063), .QN(n721) );
  MUX21X1 U588 ( .IN1(n1062), .IN2(n1064), .S(n1065), .Q(n719) );
  MUX21X1 U589 ( .IN1(n984), .IN2(n1056), .S(n722), .Q(n598) );
  MUX21X1 U590 ( .IN1(n1016), .IN2(n1057), .S(n722), .Q(n597) );
  AND2X1 U591 ( .IN1(n723), .IN2(n724), .Q(n722) );
  MUX21X1 U592 ( .IN1(n992), .IN2(n1056), .S(n725), .Q(n596) );
  MUX21X1 U593 ( .IN1(n1024), .IN2(n1057), .S(n725), .Q(n595) );
  AND2X1 U594 ( .IN1(n726), .IN2(n724), .Q(n725) );
  MUX21X1 U595 ( .IN1(n1000), .IN2(n1056), .S(n727), .Q(n594) );
  MUX21X1 U596 ( .IN1(n1032), .IN2(n1057), .S(n727), .Q(n593) );
  AND2X1 U597 ( .IN1(n728), .IN2(n724), .Q(n727) );
  MUX21X1 U598 ( .IN1(n1008), .IN2(n1056), .S(n729), .Q(n592) );
  MUX21X1 U599 ( .IN1(n1040), .IN2(n1057), .S(n729), .Q(n591) );
  AND2X1 U600 ( .IN1(n730), .IN2(n724), .Q(n729) );
  MUX21X1 U601 ( .IN1(n985), .IN2(n1056), .S(n731), .Q(n590) );
  MUX21X1 U602 ( .IN1(n1017), .IN2(n1057), .S(n731), .Q(n589) );
  AND2X1 U603 ( .IN1(n732), .IN2(n723), .Q(n731) );
  MUX21X1 U604 ( .IN1(n993), .IN2(n1056), .S(n733), .Q(n588) );
  MUX21X1 U605 ( .IN1(n1025), .IN2(n1057), .S(n733), .Q(n587) );
  AND2X1 U606 ( .IN1(n732), .IN2(n726), .Q(n733) );
  MUX21X1 U607 ( .IN1(n1001), .IN2(n1056), .S(n734), .Q(n586) );
  MUX21X1 U608 ( .IN1(n1033), .IN2(n1057), .S(n734), .Q(n585) );
  AND2X1 U609 ( .IN1(n732), .IN2(n728), .Q(n734) );
  MUX21X1 U610 ( .IN1(n1009), .IN2(n1056), .S(n735), .Q(n584) );
  MUX21X1 U611 ( .IN1(n1041), .IN2(n1057), .S(n735), .Q(n583) );
  AND2X1 U612 ( .IN1(n732), .IN2(n730), .Q(n735) );
  MUX21X1 U613 ( .IN1(n978), .IN2(n1056), .S(n736), .Q(n582) );
  MUX21X1 U614 ( .IN1(n1010), .IN2(n1057), .S(n736), .Q(n581) );
  AND2X1 U615 ( .IN1(n737), .IN2(n723), .Q(n736) );
  MUX21X1 U616 ( .IN1(n986), .IN2(n1056), .S(n738), .Q(n580) );
  MUX21X1 U617 ( .IN1(n1018), .IN2(n1057), .S(n738), .Q(n579) );
  AND2X1 U618 ( .IN1(n737), .IN2(n726), .Q(n738) );
  MUX21X1 U619 ( .IN1(n994), .IN2(n1056), .S(n739), .Q(n578) );
  MUX21X1 U620 ( .IN1(n1026), .IN2(n1057), .S(n739), .Q(n577) );
  AND2X1 U621 ( .IN1(n737), .IN2(n728), .Q(n739) );
  MUX21X1 U622 ( .IN1(n1002), .IN2(n1056), .S(n740), .Q(n576) );
  MUX21X1 U623 ( .IN1(n1034), .IN2(n1057), .S(n740), .Q(n575) );
  AND2X1 U624 ( .IN1(n737), .IN2(n730), .Q(n740) );
  MUX21X1 U625 ( .IN1(n979), .IN2(n1056), .S(n741), .Q(n574) );
  MUX21X1 U626 ( .IN1(n1011), .IN2(n1057), .S(n741), .Q(n573) );
  AND2X1 U627 ( .IN1(n742), .IN2(n723), .Q(n741) );
  MUX21X1 U628 ( .IN1(n987), .IN2(n1056), .S(n743), .Q(n572) );
  MUX21X1 U629 ( .IN1(n1019), .IN2(n1057), .S(n743), .Q(n571) );
  AND2X1 U630 ( .IN1(n742), .IN2(n726), .Q(n743) );
  MUX21X1 U631 ( .IN1(n995), .IN2(n1056), .S(n744), .Q(n570) );
  MUX21X1 U632 ( .IN1(n1027), .IN2(n1057), .S(n744), .Q(n569) );
  AND2X1 U633 ( .IN1(n742), .IN2(n728), .Q(n744) );
  MUX21X1 U634 ( .IN1(n1003), .IN2(n1056), .S(n745), .Q(n568) );
  MUX21X1 U635 ( .IN1(n1035), .IN2(n1057), .S(n745), .Q(n567) );
  AND2X1 U636 ( .IN1(n742), .IN2(n730), .Q(n745) );
  MUX21X1 U637 ( .IN1(n982), .IN2(n1056), .S(n746), .Q(n566) );
  MUX21X1 U638 ( .IN1(n1014), .IN2(n1057), .S(n746), .Q(n565) );
  AND2X1 U639 ( .IN1(n747), .IN2(n723), .Q(n746) );
  MUX21X1 U640 ( .IN1(n990), .IN2(n1056), .S(n748), .Q(n564) );
  MUX21X1 U641 ( .IN1(n1022), .IN2(n1057), .S(n748), .Q(n563) );
  AND2X1 U642 ( .IN1(n747), .IN2(n726), .Q(n748) );
  MUX21X1 U643 ( .IN1(n998), .IN2(n1056), .S(n749), .Q(n562) );
  MUX21X1 U644 ( .IN1(n1030), .IN2(n1057), .S(n749), .Q(n561) );
  AND2X1 U645 ( .IN1(n747), .IN2(n728), .Q(n749) );
  MUX21X1 U646 ( .IN1(n1006), .IN2(n1056), .S(n750), .Q(n560) );
  MUX21X1 U647 ( .IN1(n1038), .IN2(n1057), .S(n750), .Q(n559) );
  AND2X1 U648 ( .IN1(n747), .IN2(n730), .Q(n750) );
  MUX21X1 U649 ( .IN1(n983), .IN2(n1056), .S(n751), .Q(n558) );
  MUX21X1 U650 ( .IN1(n1015), .IN2(n1057), .S(n751), .Q(n557) );
  AND2X1 U651 ( .IN1(n752), .IN2(n723), .Q(n751) );
  MUX21X1 U652 ( .IN1(n991), .IN2(n1056), .S(n753), .Q(n556) );
  MUX21X1 U653 ( .IN1(n1023), .IN2(n1057), .S(n753), .Q(n555) );
  AND2X1 U654 ( .IN1(n752), .IN2(n726), .Q(n753) );
  MUX21X1 U655 ( .IN1(n999), .IN2(n1056), .S(n754), .Q(n554) );
  MUX21X1 U656 ( .IN1(n1031), .IN2(n1057), .S(n754), .Q(n553) );
  AND2X1 U657 ( .IN1(n752), .IN2(n728), .Q(n754) );
  MUX21X1 U658 ( .IN1(n1007), .IN2(n1056), .S(n755), .Q(n552) );
  MUX21X1 U659 ( .IN1(n1039), .IN2(n1057), .S(n755), .Q(n551) );
  AND2X1 U660 ( .IN1(n752), .IN2(n730), .Q(n755) );
  MUX21X1 U661 ( .IN1(n980), .IN2(n1056), .S(n756), .Q(n550) );
  MUX21X1 U662 ( .IN1(n1012), .IN2(n1057), .S(n756), .Q(n549) );
  AND2X1 U663 ( .IN1(n757), .IN2(n723), .Q(n756) );
  MUX21X1 U664 ( .IN1(n988), .IN2(n1056), .S(n758), .Q(n548) );
  MUX21X1 U665 ( .IN1(n1020), .IN2(n1057), .S(n758), .Q(n547) );
  AND2X1 U666 ( .IN1(n757), .IN2(n726), .Q(n758) );
  MUX21X1 U667 ( .IN1(n996), .IN2(n1056), .S(n759), .Q(n546) );
  MUX21X1 U668 ( .IN1(n1028), .IN2(n1057), .S(n759), .Q(n545) );
  AND2X1 U669 ( .IN1(n757), .IN2(n728), .Q(n759) );
  MUX21X1 U670 ( .IN1(n1004), .IN2(n1056), .S(n760), .Q(n544) );
  MUX21X1 U671 ( .IN1(n1036), .IN2(n1057), .S(n760), .Q(n543) );
  AND2X1 U672 ( .IN1(n757), .IN2(n730), .Q(n760) );
  MUX21X1 U673 ( .IN1(n981), .IN2(n1056), .S(n761), .Q(n542) );
  MUX21X1 U674 ( .IN1(n1013), .IN2(n1057), .S(n761), .Q(n541) );
  AND2X1 U675 ( .IN1(n762), .IN2(n723), .Q(n761) );
  AND2X1 U676 ( .IN1(n763), .IN2(n636), .Q(n723) );
  MUX21X1 U677 ( .IN1(n989), .IN2(n1056), .S(n764), .Q(n540) );
  MUX21X1 U678 ( .IN1(n1021), .IN2(n1057), .S(n764), .Q(n539) );
  AND2X1 U679 ( .IN1(n762), .IN2(n726), .Q(n764) );
  AND2X1 U680 ( .IN1(n765), .IN2(n636), .Q(n726) );
  MUX21X1 U681 ( .IN1(n997), .IN2(n1056), .S(n766), .Q(n538) );
  MUX21X1 U682 ( .IN1(n1029), .IN2(n1057), .S(n766), .Q(n537) );
  AND2X1 U683 ( .IN1(n762), .IN2(n728), .Q(n766) );
  AND2X1 U684 ( .IN1(n767), .IN2(n636), .Q(n728) );
  MUX21X1 U685 ( .IN1(n1005), .IN2(n1056), .S(n768), .Q(n536) );
  MUX21X1 U686 ( .IN1(n639), .IN2(n769), .S(n712), .Q(n535) );
  NAND4X0 U687 ( .IN1(n770), .IN2(n771), .IN3(n772), .IN4(n773), .QN(n769) );
  OR3X1 U688 ( .IN1(n774), .IN2(k[0]), .IN3(n716), .Q(n771) );
  NAND2X0 U689 ( .IN1(n1050), .IN2(n775), .QN(n770) );
  MUX21X1 U690 ( .IN1(n1037), .IN2(n1057), .S(n768), .Q(n534) );
  AND2X1 U691 ( .IN1(n762), .IN2(n730), .Q(n768) );
  AND2X1 U692 ( .IN1(n776), .IN2(n636), .Q(n730) );
  MUX21X1 U693 ( .IN1(n777), .IN2(nl[2]), .S(n778), .Q(n533) );
  AND3X1 U694 ( .IN1(n779), .IN2(n780), .IN3(n781), .Q(n778) );
  NAND3X0 U695 ( .IN1(n782), .IN2(n638), .IN3(n1049), .QN(n780) );
  NAND3X0 U696 ( .IN1(n1059), .IN2(n640), .IN3(n783), .QN(n779) );
  MUX21X1 U697 ( .IN1(n777), .IN2(nl[3]), .S(n784), .Q(n532) );
  AND3X1 U698 ( .IN1(n785), .IN2(n786), .IN3(n781), .Q(n784) );
  NAND3X0 U699 ( .IN1(n1050), .IN2(n782), .IN3(n1049), .QN(n786) );
  NAND3X0 U700 ( .IN1(n1058), .IN2(n1059), .IN3(n783), .QN(n785) );
  MUX21X1 U701 ( .IN1(n626), .IN2(n787), .S(n712), .Q(n531) );
  AO221X1 U702 ( .IN1(n661), .IN2(n788), .IN3(n687), .IN4(n789), .IN5(n782), 
        .Q(n712) );
  NAND2X0 U703 ( .IN1(n790), .IN2(n791), .QN(n788) );
  AO21X1 U704 ( .IN1(n1049), .IN2(n775), .IN3(n792), .Q(n787) );
  MUX21X1 U705 ( .IN1(n793), .IN2(n794), .S(n1048), .Q(n530) );
  NOR2X0 U706 ( .IN1(n795), .IN2(n642), .QN(n794) );
  NAND2X0 U707 ( .IN1(n796), .IN2(play), .QN(n793) );
  OA221X1 U708 ( .IN1(n1048), .IN2(n795), .IN3(n796), .IN4(speaker), .IN5(play), .Q(n529) );
  AND2X1 U709 ( .IN1(n797), .IN2(n798), .Q(n796) );
  MUX21X1 U710 ( .IN1(n799), .IN2(n800), .S(sound_2_), .Q(n797) );
  OA21X1 U711 ( .IN1(n801), .IN2(n639), .IN3(n1071), .Q(n800) );
  MUX21X1 U712 ( .IN1(n802), .IN2(n803), .S(n1071), .Q(n795) );
  NOR2X0 U713 ( .IN1(n801), .IN2(n804), .QN(n803) );
  MUX21X1 U714 ( .IN1(n799), .IN2(n805), .S(sound_2_), .Q(n804) );
  NOR2X0 U715 ( .IN1(n1070), .IN2(n806), .QN(n805) );
  NAND2X0 U716 ( .IN1(n631), .IN2(n807), .QN(n799) );
  INVX0 U717 ( .INP(n807), .ZN(n801) );
  NAND2X0 U718 ( .IN1(n798), .IN2(n646), .QN(n802) );
  MUX21X1 U719 ( .IN1(n777), .IN2(nl[0]), .S(n808), .Q(n528) );
  AND3X1 U720 ( .IN1(n809), .IN2(n810), .IN3(n781), .Q(n808) );
  NAND3X0 U721 ( .IN1(n638), .IN2(n628), .IN3(n782), .QN(n810) );
  NAND3X0 U722 ( .IN1(n630), .IN2(n640), .IN3(n783), .QN(n809) );
  MUX21X1 U723 ( .IN1(n777), .IN2(nl[1]), .S(n811), .Q(n527) );
  AND3X1 U724 ( .IN1(n812), .IN2(n813), .IN3(n781), .Q(n811) );
  OA21X1 U725 ( .IN1(n790), .IN2(n814), .IN3(n815), .Q(n781) );
  INVX0 U726 ( .INP(n715), .ZN(n815) );
  AO21X1 U727 ( .IN1(n661), .IN2(n816), .IN3(n817), .Q(n715) );
  NAND3X0 U728 ( .IN1(n818), .IN2(n819), .IN3(n820), .QN(n816) );
  NAND3X0 U729 ( .IN1(n782), .IN2(n628), .IN3(n1050), .QN(n813) );
  NAND2X0 U730 ( .IN1(n821), .IN2(n717), .QN(n782) );
  AO21X1 U731 ( .IN1(n822), .IN2(n773), .IN3(n814), .Q(n821) );
  NAND3X0 U732 ( .IN1(n1058), .IN2(n630), .IN3(n783), .QN(n812) );
  NAND4X0 U733 ( .IN1(n773), .IN2(n823), .IN3(n717), .IN4(n824), .QN(n777) );
  AND2X1 U734 ( .IN1(n772), .IN2(n818), .Q(n824) );
  AO221X1 U735 ( .IN1(n825), .IN2(n826), .IN3(nloss), .IN4(n695), .IN5(n827), 
        .Q(n526) );
  INVX0 U736 ( .INP(n817), .ZN(n695) );
  AO221X1 U737 ( .IN1(n828), .IN2(r271_SUM_5_), .IN3(n678), .IN4(timebase[5]), 
        .IN5(n817), .Q(n525) );
  AO22X1 U738 ( .IN1(timebase[4]), .IN2(n678), .IN3(r271_SUM_4_), .IN4(n828), 
        .Q(n524) );
  AO222X1 U739 ( .IN1(n829), .IN2(r271_SUM_4_), .IN3(n830), .IN4(timebase[4]), 
        .IN5(n1047), .IN6(n660), .Q(n523) );
  AO22X1 U740 ( .IN1(n678), .IN2(timebase[3]), .IN3(n828), .IN4(r271_SUM_3_), 
        .Q(n522) );
  AO22X1 U741 ( .IN1(timebase[2]), .IN2(n678), .IN3(r271_SUM_2_), .IN4(n828), 
        .Q(n521) );
  AO222X1 U742 ( .IN1(r271_SUM_2_), .IN2(n829), .IN3(timebase[2]), .IN4(n830), 
        .IN5(n1046), .IN6(n660), .Q(n520) );
  AO22X1 U743 ( .IN1(timebase[1]), .IN2(n678), .IN3(r271_SUM_1_), .IN4(n828), 
        .Q(n519) );
  AO222X1 U744 ( .IN1(r271_SUM_1_), .IN2(n829), .IN3(timebase[1]), .IN4(n830), 
        .IN5(n1045), .IN6(n660), .Q(n518) );
  AND2X1 U745 ( .IN1(n664), .IN2(n831), .Q(n830) );
  AND2X1 U746 ( .IN1(n668), .IN2(n831), .Q(n829) );
  AO221X1 U747 ( .IN1(n828), .IN2(n832), .IN3(timebase[0]), .IN4(n678), .IN5(
        n817), .Q(n517) );
  NOR2X0 U748 ( .IN1(n833), .IN2(n678), .QN(n828) );
  MUX21X1 U749 ( .IN1(n834), .IN2(n1044), .S(n660), .Q(n516) );
  INVX0 U750 ( .INP(n831), .ZN(n660) );
  NAND4X0 U751 ( .IN1(n835), .IN2(n836), .IN3(n820), .IN4(n837), .QN(n831) );
  AND3X1 U752 ( .IN1(n838), .IN2(n791), .IN3(n839), .Q(n837) );
  NAND2X0 U753 ( .IN1(n682), .IN2(n814), .QN(n839) );
  INVX0 U754 ( .INP(n840), .ZN(n682) );
  AO221X1 U755 ( .IN1(timebase[0]), .IN2(n664), .IN3(n668), .IN4(n832), .IN5(
        n669), .Q(n834) );
  INVX0 U756 ( .INP(U3_U3_Z_0), .ZN(n832) );
  NAND3X0 U757 ( .IN1(n841), .IN2(n840), .IN3(n665), .QN(n668) );
  AOI221X1 U758 ( .IN1(n792), .IN2(n842), .IN3(n814), .IN4(n843), .IN5(n708), 
        .QN(n665) );
  INVX0 U759 ( .INP(n844), .ZN(n708) );
  NAND2X0 U760 ( .IN1(n662), .IN2(n814), .QN(n841) );
  NAND3X0 U761 ( .IN1(n818), .IN2(n819), .IN3(n718), .QN(n662) );
  AO221X1 U762 ( .IN1(n825), .IN2(n789), .IN3(n661), .IN4(n843), .IN5(n775), 
        .Q(n664) );
  INVX0 U763 ( .INP(n717), .ZN(n775) );
  NAND2X0 U764 ( .IN1(n842), .IN2(n685), .QN(n789) );
  AO22X1 U765 ( .IN1(n845), .IN2(timebase[5]), .IN3(n1060), .IN4(n846), .Q(
        U3_U3_Z_5) );
  AO22X1 U766 ( .IN1(timebase[0]), .IN2(n845), .IN3(n1044), .IN4(n846), .Q(
        U3_U3_Z_0) );
  NAND4X0 U767 ( .IN1(n666), .IN2(n820), .IN3(n847), .IN4(n844), .QN(n846) );
  INVX0 U768 ( .INP(n843), .ZN(n820) );
  NAND2X0 U769 ( .IN1(n848), .IN2(n849), .QN(n843) );
  NOR2X0 U770 ( .IN1(n711), .IN2(n850), .QN(n666) );
  NAND4X0 U771 ( .IN1(n790), .IN2(n851), .IN3(n677), .IN4(n852), .QN(N890) );
  NOR2X0 U772 ( .IN1(n853), .IN2(n850), .QN(n852) );
  AOI21X1 U773 ( .IN1(n849), .IN2(n791), .IN3(n814), .QN(n853) );
  OR3X1 U774 ( .IN1(n833), .IN2(n854), .IN3(n681), .Q(n791) );
  NAND4X0 U775 ( .IN1(n855), .IN2(n823), .IN3(n856), .IN4(n857), .QN(N889) );
  OA221X1 U776 ( .IN1(n858), .IN2(n833), .IN3(n859), .IN4(n844), .IN5(n836), 
        .Q(n857) );
  AND3X1 U777 ( .IN1(n818), .IN2(n790), .IN3(n847), .Q(n836) );
  INVX0 U778 ( .INP(n860), .ZN(n859) );
  AO21X1 U779 ( .IN1(n861), .IN2(n862), .IN3(n863), .Q(n856) );
  NAND4X0 U780 ( .IN1(n848), .IN2(n864), .IN3(n835), .IN4(n865), .QN(N888) );
  NOR4X0 U781 ( .IN1(n783), .IN2(n866), .IN3(n867), .IN4(n868), .QN(n865) );
  NOR2X0 U782 ( .IN1(n634), .IN2(n694), .QN(n868) );
  INVX0 U783 ( .INP(n823), .ZN(n783) );
  AO21X1 U784 ( .IN1(n862), .IN2(n869), .IN3(n870), .Q(n823) );
  NAND2X0 U785 ( .IN1(n1064), .IN2(n627), .QN(n869) );
  AND4X1 U786 ( .IN1(n819), .IN2(n773), .IN3(n717), .IN4(n871), .Q(n835) );
  NOR2X0 U787 ( .IN1(n872), .IN2(n873), .QN(n871) );
  AOI21X1 U788 ( .IN1(n844), .IN2(n675), .IN3(n661), .QN(n873) );
  NAND4X0 U789 ( .IN1(n874), .IN2(n1064), .IN3(n875), .IN4(n861), .QN(n717) );
  MUX21X1 U790 ( .IN1(n635), .IN2(n627), .S(n1063), .Q(n874) );
  NAND2X0 U791 ( .IN1(n687), .IN2(n826), .QN(n864) );
  NAND2X0 U792 ( .IN1(n876), .IN2(n877), .QN(n826) );
  MUX21X1 U793 ( .IN1(n878), .IN2(n879), .S(n1050), .Q(n877) );
  NOR2X0 U794 ( .IN1(n774), .IN2(k[0]), .QN(n879) );
  AND2X1 U795 ( .IN1(k[2]), .IN2(n880), .Q(n774) );
  OR2X1 U796 ( .IN1(n691), .IN2(k[0]), .Q(n878) );
  OA21X1 U797 ( .IN1(n881), .IN2(k[2]), .IN3(n880), .Q(n691) );
  MUX21X1 U798 ( .IN1(n628), .IN2(n882), .S(n685), .Q(n876) );
  NAND2X0 U799 ( .IN1(n686), .IN2(n628), .QN(n882) );
  INVX0 U800 ( .INP(n842), .ZN(n686) );
  NOR2X0 U801 ( .IN1(k[3]), .IN2(k[2]), .QN(n842) );
  INVX0 U802 ( .INP(n883), .ZN(n848) );
  NAND4X0 U803 ( .IN1(n884), .IN2(n885), .IN3(n886), .IN4(n887), .QN(N887) );
  AND4X1 U804 ( .IN1(n888), .IN2(n819), .IN3(n889), .IN4(n773), .Q(n887) );
  AOI21X1 U805 ( .IN1(n890), .IN2(n891), .IN3(n687), .QN(n888) );
  INVX0 U806 ( .INP(n847), .ZN(n687) );
  NAND2X0 U807 ( .IN1(n825), .IN2(n814), .QN(n847) );
  OA221X1 U808 ( .IN1(n814), .IN2(n822), .IN3(n634), .IN4(n892), .IN5(n678), 
        .Q(n886) );
  NOR2X0 U809 ( .IN1(n817), .IN2(n845), .QN(n678) );
  AND2X1 U810 ( .IN1(n858), .IN2(n709), .Q(n845) );
  AND2X1 U811 ( .IN1(n860), .IN2(n854), .Q(n858) );
  NAND4X0 U812 ( .IN1(max[0]), .IN2(max[1]), .IN3(n893), .IN4(max[2]), .QN(
        n854) );
  NOR2X0 U813 ( .IN1(n509), .IN2(n508), .QN(n893) );
  NAND2X0 U814 ( .IN1(n894), .IN2(n895), .QN(n817) );
  NAND3X0 U815 ( .IN1(n625), .IN2(n645), .IN3(n890), .QN(n895) );
  INVX0 U816 ( .INP(n872), .ZN(n822) );
  NOR2X0 U817 ( .IN1(n772), .IN2(n681), .QN(n872) );
  INVX0 U818 ( .INP(n705), .ZN(n885) );
  AO21X1 U819 ( .IN1(n890), .IN2(n896), .IN3(n696), .Q(n705) );
  NOR2X0 U820 ( .IN1(n694), .IN2(n1064), .QN(n696) );
  NAND3X0 U821 ( .IN1(n627), .IN2(n625), .IN3(n897), .QN(n694) );
  NOR2X0 U822 ( .IN1(n861), .IN2(n634), .QN(n890) );
  NAND4X0 U823 ( .IN1(n898), .IN2(n884), .IN3(n899), .IN4(n900), .QN(N886) );
  NOR4X0 U824 ( .IN1(n866), .IN2(n669), .IN3(n827), .IN4(n901), .QN(n900) );
  AND2X1 U825 ( .IN1(n681), .IN2(n680), .Q(n901) );
  INVX0 U826 ( .INP(n677), .ZN(n827) );
  NAND2X0 U827 ( .IN1(n661), .IN2(n825), .QN(n677) );
  INVX0 U828 ( .INP(n838), .ZN(n669) );
  NAND3X0 U829 ( .IN1(n891), .IN2(n634), .IN3(n902), .QN(n838) );
  NOR2X0 U830 ( .IN1(n892), .IN2(n862), .QN(n866) );
  NAND3X0 U831 ( .IN1(n635), .IN2(n625), .IN3(n875), .QN(n892) );
  MUX21X1 U832 ( .IN1(n903), .IN2(n904), .S(n1049), .Q(n899) );
  NAND2X0 U833 ( .IN1(n792), .IN2(n905), .QN(n904) );
  MUX21X1 U834 ( .IN1(k[2]), .IN2(n906), .S(n1050), .Q(n905) );
  NOR2X0 U835 ( .IN1(k[2]), .IN2(n881), .QN(n906) );
  INVX0 U836 ( .INP(k[3]), .ZN(n881) );
  NOR2X0 U837 ( .IN1(n688), .IN2(n716), .QN(n792) );
  INVX0 U838 ( .INP(n685), .ZN(n688) );
  NOR2X0 U839 ( .IN1(k[0]), .IN2(k[1]), .QN(n685) );
  NAND2X0 U840 ( .IN1(n825), .IN2(n907), .QN(n903) );
  MUX21X1 U841 ( .IN1(k[0]), .IN2(n908), .S(n1050), .Q(n907) );
  NOR2X0 U842 ( .IN1(k[0]), .IN2(n880), .QN(n908) );
  INVX0 U843 ( .INP(k[1]), .ZN(n880) );
  INVX0 U844 ( .INP(n716), .ZN(n825) );
  NAND3X0 U845 ( .IN1(n627), .IN2(n634), .IN3(n909), .QN(n716) );
  AND3X1 U846 ( .IN1(n910), .IN2(n855), .IN3(n911), .Q(n884) );
  OA22X1 U847 ( .IN1(n860), .IN2(n844), .IN3(n862), .IN4(n870), .Q(n911) );
  NAND3X0 U848 ( .IN1(n897), .IN2(n625), .IN3(n912), .QN(n844) );
  NOR2X0 U849 ( .IN1(n814), .IN2(n681), .QN(n860) );
  NAND4X0 U850 ( .IN1(n913), .IN2(n914), .IN3(n915), .IN4(n916), .QN(n681) );
  XOR2X1 U851 ( .IN1(scan[1]), .IN2(n501), .Q(n916) );
  NOR2X0 U852 ( .IN1(n917), .IN2(n918), .QN(n915) );
  XOR2X1 U853 ( .IN1(scan[0]), .IN2(max[0]), .Q(n918) );
  XOR2X1 U854 ( .IN1(scan[2]), .IN2(max[2]), .Q(n917) );
  XOR2X1 U855 ( .IN1(scan[3]), .IN2(n508), .Q(n914) );
  XOR2X1 U856 ( .IN1(scan[4]), .IN2(n509), .Q(n913) );
  INVX0 U857 ( .INP(n661), .ZN(n814) );
  INVX0 U858 ( .INP(n867), .ZN(n855) );
  NOR2X0 U859 ( .IN1(n849), .IN2(n661), .QN(n867) );
  NAND2X0 U860 ( .IN1(n909), .IN2(n912), .QN(n849) );
  INVX0 U861 ( .INP(n870), .ZN(n909) );
  NAND2X0 U862 ( .IN1(n891), .IN2(n1065), .QN(n870) );
  NAND3X0 U863 ( .IN1(n627), .IN2(n634), .IN3(n897), .QN(n910) );
  MUX21X1 U864 ( .IN1(n718), .IN2(n919), .S(n661), .Q(n898) );
  OR3X1 U865 ( .IN1(n1046), .IN2(n1044), .IN3(n1045), .Q(n920) );
  NOR2X0 U866 ( .IN1(n883), .IN2(n850), .QN(n919) );
  NAND3X0 U867 ( .IN1(n819), .IN2(n840), .IN3(n818), .QN(n850) );
  NAND3X0 U868 ( .IN1(n1062), .IN2(n634), .IN3(n921), .QN(n818) );
  NAND3X0 U869 ( .IN1(n902), .IN2(n634), .IN3(n896), .QN(n840) );
  INVX0 U870 ( .INP(n861), .ZN(n902) );
  NAND3X0 U871 ( .IN1(n922), .IN2(n1063), .IN3(n897), .QN(n819) );
  NAND2X0 U872 ( .IN1(n889), .IN2(n923), .QN(n883) );
  NAND3X0 U873 ( .IN1(n924), .IN2(n635), .IN3(n922), .QN(n923) );
  NAND2X0 U874 ( .IN1(n863), .IN2(n851), .QN(n924) );
  NAND3X0 U875 ( .IN1(n922), .IN2(n625), .IN3(n897), .QN(n889) );
  INVX0 U876 ( .INP(n862), .ZN(n922) );
  NAND2X0 U877 ( .IN1(n1066), .IN2(n634), .QN(n862) );
  INVX0 U878 ( .INP(n711), .ZN(n718) );
  NAND3X0 U879 ( .IN1(n790), .IN2(n773), .IN3(n675), .QN(n711) );
  NOR2X0 U880 ( .IN1(n709), .IN2(n680), .QN(n675) );
  INVX0 U881 ( .INP(n772), .ZN(n680) );
  NAND3X0 U882 ( .IN1(n912), .IN2(n635), .IN3(n896), .QN(n772) );
  INVX0 U883 ( .INP(n851), .ZN(n896) );
  NAND2X0 U884 ( .IN1(n1063), .IN2(n875), .QN(n851) );
  INVX0 U885 ( .INP(n833), .ZN(n709) );
  NAND3X0 U886 ( .IN1(n891), .IN2(n635), .IN3(n912), .QN(n833) );
  INVX0 U887 ( .INP(n863), .ZN(n891) );
  NAND3X0 U888 ( .IN1(n625), .IN2(n894), .IN3(n1062), .QN(n863) );
  INVX0 U889 ( .INP(start), .ZN(n894) );
  NAND3X0 U890 ( .IN1(n897), .IN2(n1063), .IN3(n912), .QN(n773) );
  NOR2X0 U891 ( .IN1(n634), .IN2(n627), .QN(n912) );
  AND2X1 U892 ( .IN1(n1065), .IN2(n875), .Q(n897) );
  NOR2X0 U893 ( .IN1(n1062), .IN2(start), .QN(n875) );
  NAND2X0 U894 ( .IN1(n925), .IN2(n921), .QN(n790) );
  NOR3X0 U895 ( .IN1(n625), .IN2(start), .IN3(n861), .QN(n921) );
  NAND2X0 U896 ( .IN1(n635), .IN2(n627), .QN(n861) );
  INVX0 U897 ( .INP(n720), .ZN(n925) );
  NAND2X0 U898 ( .IN1(n1064), .IN2(n1062), .QN(n720) );
  AO221X1 U899 ( .IN1(n765), .IN2(n926), .IN3(n763), .IN4(n927), .IN5(n928), 
        .Q(N210) );
  AO22X1 U900 ( .IN1(n767), .IN2(n929), .IN3(n776), .IN4(n930), .Q(n928) );
  NAND4X0 U901 ( .IN1(n931), .IN2(n932), .IN3(n933), .IN4(n934), .QN(n930) );
  AOI22X1 U902 ( .IN1(n737), .IN2(n1002), .IN3(n742), .IN4(n1003), .QN(n934)
         );
  AOI22X1 U903 ( .IN1(n757), .IN2(n1004), .IN3(n762), .IN4(n1005), .QN(n933)
         );
  AOI22X1 U904 ( .IN1(n747), .IN2(n1006), .IN3(n752), .IN4(n1007), .QN(n932)
         );
  AOI22X1 U905 ( .IN1(n724), .IN2(n1008), .IN3(n732), .IN4(n1009), .QN(n931)
         );
  NAND4X0 U906 ( .IN1(n935), .IN2(n936), .IN3(n937), .IN4(n938), .QN(n929) );
  AOI22X1 U907 ( .IN1(n737), .IN2(n994), .IN3(n742), .IN4(n995), .QN(n938) );
  AOI22X1 U908 ( .IN1(n757), .IN2(n996), .IN3(n762), .IN4(n997), .QN(n937) );
  AOI22X1 U909 ( .IN1(n747), .IN2(n998), .IN3(n752), .IN4(n999), .QN(n936) );
  AOI22X1 U910 ( .IN1(n724), .IN2(n1000), .IN3(n732), .IN4(n1001), .QN(n935)
         );
  NAND4X0 U911 ( .IN1(n939), .IN2(n940), .IN3(n941), .IN4(n942), .QN(n927) );
  AOI22X1 U912 ( .IN1(n737), .IN2(n978), .IN3(n742), .IN4(n979), .QN(n942) );
  AOI22X1 U913 ( .IN1(n757), .IN2(n980), .IN3(n762), .IN4(n981), .QN(n941) );
  AOI22X1 U914 ( .IN1(n747), .IN2(n982), .IN3(n752), .IN4(n983), .QN(n940) );
  AOI22X1 U915 ( .IN1(n724), .IN2(n984), .IN3(n732), .IN4(n985), .QN(n939) );
  NAND4X0 U916 ( .IN1(n943), .IN2(n944), .IN3(n945), .IN4(n946), .QN(n926) );
  AOI22X1 U917 ( .IN1(n737), .IN2(n986), .IN3(n742), .IN4(n987), .QN(n946) );
  AOI22X1 U918 ( .IN1(n757), .IN2(n988), .IN3(n762), .IN4(n989), .QN(n945) );
  AOI22X1 U919 ( .IN1(n747), .IN2(n990), .IN3(n752), .IN4(n991), .QN(n944) );
  AOI22X1 U920 ( .IN1(n724), .IN2(n992), .IN3(n732), .IN4(n993), .QN(n943) );
  AO221X1 U921 ( .IN1(n765), .IN2(n947), .IN3(n763), .IN4(n948), .IN5(n949), 
        .Q(N209) );
  AO22X1 U922 ( .IN1(n767), .IN2(n950), .IN3(n776), .IN4(n951), .Q(n949) );
  NAND4X0 U923 ( .IN1(n952), .IN2(n953), .IN3(n954), .IN4(n955), .QN(n951) );
  AOI22X1 U924 ( .IN1(n737), .IN2(n1034), .IN3(n742), .IN4(n1035), .QN(n955)
         );
  AOI22X1 U925 ( .IN1(n757), .IN2(n1036), .IN3(n762), .IN4(n1037), .QN(n954)
         );
  AOI22X1 U926 ( .IN1(n747), .IN2(n1038), .IN3(n752), .IN4(n1039), .QN(n953)
         );
  AOI22X1 U927 ( .IN1(n724), .IN2(n1040), .IN3(n732), .IN4(n1041), .QN(n952)
         );
  NOR2X0 U928 ( .IN1(n1055), .IN2(n1053), .QN(n776) );
  NAND4X0 U929 ( .IN1(n956), .IN2(n957), .IN3(n958), .IN4(n959), .QN(n950) );
  AOI22X1 U930 ( .IN1(n737), .IN2(n1026), .IN3(n742), .IN4(n1027), .QN(n959)
         );
  AOI22X1 U931 ( .IN1(n757), .IN2(n1028), .IN3(n762), .IN4(n1029), .QN(n958)
         );
  AOI22X1 U932 ( .IN1(n747), .IN2(n1030), .IN3(n752), .IN4(n1031), .QN(n957)
         );
  AOI22X1 U933 ( .IN1(n724), .IN2(n1032), .IN3(n732), .IN4(n1033), .QN(n956)
         );
  NOR2X0 U934 ( .IN1(n641), .IN2(n1053), .QN(n767) );
  NAND4X0 U935 ( .IN1(n960), .IN2(n961), .IN3(n962), .IN4(n963), .QN(n948) );
  AOI22X1 U936 ( .IN1(n737), .IN2(n1010), .IN3(n742), .IN4(n1011), .QN(n963)
         );
  AOI22X1 U937 ( .IN1(n757), .IN2(n1012), .IN3(n762), .IN4(n1013), .QN(n962)
         );
  AOI22X1 U938 ( .IN1(n747), .IN2(n1014), .IN3(n752), .IN4(n1015), .QN(n961)
         );
  AOI22X1 U939 ( .IN1(n724), .IN2(n1016), .IN3(n732), .IN4(n1017), .QN(n960)
         );
  NOR2X0 U940 ( .IN1(n632), .IN2(n641), .QN(n763) );
  NAND4X0 U941 ( .IN1(n964), .IN2(n965), .IN3(n966), .IN4(n967), .QN(n947) );
  AOI22X1 U942 ( .IN1(n737), .IN2(n1018), .IN3(n742), .IN4(n1019), .QN(n967)
         );
  AND3X1 U943 ( .IN1(n1054), .IN2(n637), .IN3(n1051), .Q(n742) );
  AND3X1 U944 ( .IN1(n1052), .IN2(n1054), .IN3(n1051), .Q(n737) );
  AOI22X1 U945 ( .IN1(n757), .IN2(n1020), .IN3(n762), .IN4(n1021), .QN(n966)
         );
  NOR3X0 U946 ( .IN1(n629), .IN2(n1051), .IN3(n637), .QN(n762) );
  NOR3X0 U947 ( .IN1(n1052), .IN2(n1051), .IN3(n629), .QN(n757) );
  AOI22X1 U948 ( .IN1(n747), .IN2(n1022), .IN3(n752), .IN4(n1023), .QN(n965)
         );
  NOR3X0 U949 ( .IN1(n1052), .IN2(n1051), .IN3(n1054), .QN(n752) );
  NOR3X0 U950 ( .IN1(n1054), .IN2(n1051), .IN3(n637), .QN(n747) );
  AOI22X1 U951 ( .IN1(n724), .IN2(n1024), .IN3(n732), .IN4(n1025), .QN(n964)
         );
  AND3X1 U952 ( .IN1(n1052), .IN2(n629), .IN3(n1051), .Q(n732) );
  AND3X1 U953 ( .IN1(n629), .IN2(n637), .IN3(n1051), .Q(n724) );
  NOR2X0 U954 ( .IN1(n632), .IN2(n1055), .QN(n765) );
  XOR2X1 U955 ( .IN1(n1042), .IN2(n1043), .Q(N208) );
  OA21X1 U956 ( .IN1(n968), .IN2(n1067), .IN3(n969), .Q(N203) );
  INVX0 U957 ( .INP(n970), .ZN(n969) );
  NOR2X0 U958 ( .IN1(n633), .IN2(n643), .QN(n968) );
  MUX21X1 U959 ( .IN1(n971), .IN2(N201), .S(n1068), .Q(N202) );
  NOR2X0 U960 ( .IN1(n633), .IN2(n970), .QN(n971) );
  NOR2X0 U961 ( .IN1(n970), .IN2(n1069), .QN(N201) );
  AO21X1 U962 ( .IN1(n972), .IN2(n973), .IN3(n642), .Q(n970) );
  NAND3X0 U963 ( .IN1(n1071), .IN2(n807), .IN3(n974), .QN(n973) );
  MUX21X1 U964 ( .IN1(n631), .IN2(n1070), .S(sound_2_), .Q(n974) );
  NAND4X0 U965 ( .IN1(n1068), .IN2(n1069), .IN3(n1070), .IN4(n975), .QN(n807)
         );
  NAND2X0 U966 ( .IN1(sound_2_), .IN2(n631), .QN(n975) );
  AO21X1 U967 ( .IN1(sound_2_), .IN2(n626), .IN3(n798), .Q(n972) );
  INVX0 U968 ( .INP(n806), .ZN(n798) );
  AO22X1 U969 ( .IN1(n631), .IN2(n643), .IN3(n626), .IN4(n976), .Q(n806) );
  NAND2X0 U970 ( .IN1(n1067), .IN2(n977), .QN(n976) );
  AO21X1 U971 ( .IN1(n1069), .IN2(n1070), .IN3(n1068), .Q(n977) );
endmodule

