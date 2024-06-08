/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:08:05 2023
/////////////////////////////////////////////////////////////


module b04 ( RESTART, AVERAGE, ENABLE, DATA_IN, DATA_OUT, RESET, CLOCK );
  input [7:0] DATA_IN;
  output [7:0] DATA_OUT;
  input RESTART, AVERAGE, ENABLE, RESET, CLOCK;
  wire   N40, N83, N84, N85, N86, N87, N88, N118, N119, N120, N121, N122, N123,
         N124, N125, N170, N171, U3_U3_Z_0, U3_U3_Z_1, U3_U3_Z_2, U3_U3_Z_3,
         U3_U3_Z_4, U3_U3_Z_5, U3_U3_Z_6, U3_U3_Z_8, U3_U4_Z_0, U3_U4_Z_1,
         U3_U4_Z_2, U3_U4_Z_3, U3_U4_Z_4, U3_U4_Z_5, U3_U4_Z_6, U3_U4_Z_8,
         n196, n197, n198, n201, n202, n205, n206, n207, n208, n209, n210,
         n211, n212, n213, n214, n215, n216, n217, n218, n219, n220, n221,
         n222, n223, n224, n225, n226, n227, n228, n229, n230, n231, n232,
         n233, n234, n235, n236, n237, n238, n239, n240, n241, n242, n243,
         n244, n245, n246, n247, n248, n249, n250, n251, n252, n253, n254,
         n255, n256, n257, n258, n259, n260, n261, n262, n263, n264, n265,
         n266, n267, n268, n269, n270, n271, n272, n273, n274, n275,
         r180_carry_8_, n277, n278, n279, n280, n281, n282, n283, n284, n285,
         n286, n287, n288, n289, n290, n291, n292, n293, n294, n295, n296,
         n297, n298, n299, n300, n301, n302, n303, n304, n305, n306, n307,
         n308, n309, n310, n311, n312, n313, n314, n315, n316, n317, n318,
         n319, n320, n321, n322, n323, n324, n325, n326, n327, n328, n329,
         n330, n331, n332, n333, n334, n335, n336, n337, n338, n339, n340,
         n341, n342, n343, n344, n345, n346, n347, n348, n349, n350, n351,
         n352, n353, n354, n355, n356, n357, n358, n359, n360, n361, n362,
         n363, n364, n365, n366, n367, n368, n369, n370, n371, n372, n373,
         n374, n375, n376, n377, n378, n379, n380, n381, n382, n383, n384,
         n385, n386, n387, n388, n389, n390, n391, n392, n393, n394, n395,
         n396, n397, n398, n399, n400, n401, n402, n403, n404, n405, n406,
         n407, n408, n409, n410, n411, n412, n413, n414, n415, n416, n417,
         n418, n419, n420, n421, n422, n423, n424, n425, n426, n427;
  wire   [1:0] stato;
  wire   [6:2] r34_carry;

  DFFARX1 stato_reg_0_ ( .D(N170), .CLK(CLOCK), .RSTB(n427), .Q(stato[0]), 
        .QN(n284) );
  DFFARX1 stato_reg_1_ ( .D(N171), .CLK(CLOCK), .RSTB(n427), .Q(stato[1]) );
  DFFARX1 RMAX_reg_0_ ( .D(n275), .CLK(CLOCK), .RSTB(n427), .Q(n299), .QN(n211) );
  DFFARX1 RMAX_reg_1_ ( .D(n274), .CLK(CLOCK), .RSTB(n427), .Q(n296), .QN(n210) );
  DFFARX1 RMAX_reg_2_ ( .D(n273), .CLK(CLOCK), .RSTB(n427), .Q(n297), .QN(n209) );
  DFFARX1 RMAX_reg_3_ ( .D(n272), .CLK(CLOCK), .RSTB(n427), .Q(n294), .QN(n208) );
  DFFARX1 RMAX_reg_4_ ( .D(n271), .CLK(CLOCK), .RSTB(n427), .Q(n293), .QN(n207) );
  DFFARX1 RMAX_reg_5_ ( .D(n270), .CLK(CLOCK), .RSTB(n427), .Q(n295), .QN(n206) );
  DFFARX1 RMAX_reg_6_ ( .D(n269), .CLK(CLOCK), .RSTB(n427), .Q(n281), .QN(n205) );
  DFFARX1 RMAX_reg_7_ ( .D(n268), .CLK(CLOCK), .RSTB(n427), .Q(n280) );
  DFFARX1 RMIN_reg_0_ ( .D(n267), .CLK(CLOCK), .RSTB(n427), .Q(n300) );
  DFFARX1 RMIN_reg_7_ ( .D(n266), .CLK(CLOCK), .RSTB(n427), .Q(n285), .QN(n202) );
  DFFARX1 RMIN_reg_6_ ( .D(n265), .CLK(CLOCK), .RSTB(n427), .Q(n282), .QN(n201) );
  DFFARX1 RMIN_reg_5_ ( .D(n264), .CLK(CLOCK), .RSTB(n427), .Q(n277) );
  DFFARX1 RMIN_reg_4_ ( .D(n263), .CLK(CLOCK), .RSTB(n427), .Q(n278) );
  DFFARX1 RMIN_reg_3_ ( .D(n262), .CLK(CLOCK), .RSTB(n427), .Q(n283), .QN(n198) );
  DFFARX1 RMIN_reg_2_ ( .D(n261), .CLK(CLOCK), .RSTB(n427), .Q(n298), .QN(n197) );
  DFFARX1 RMIN_reg_1_ ( .D(n260), .CLK(CLOCK), .RSTB(n427), .Q(n292), .QN(n196) );
  DFFARX1 RLAST_reg_7_ ( .D(n259), .CLK(CLOCK), .RSTB(n427), .Q(n419) );
  DFFARX1 RLAST_reg_6_ ( .D(n258), .CLK(CLOCK), .RSTB(n427), .Q(n418) );
  DFFARX1 RLAST_reg_5_ ( .D(n257), .CLK(CLOCK), .RSTB(n427), .Q(n417) );
  DFFARX1 RLAST_reg_4_ ( .D(n256), .CLK(CLOCK), .RSTB(n427), .Q(n416) );
  DFFARX1 RLAST_reg_3_ ( .D(n255), .CLK(CLOCK), .RSTB(n427), .Q(n415) );
  DFFARX1 RLAST_reg_2_ ( .D(n254), .CLK(CLOCK), .RSTB(n427), .Q(n414) );
  DFFARX1 RLAST_reg_1_ ( .D(n253), .CLK(CLOCK), .RSTB(n427), .Q(n413) );
  DFFARX1 RLAST_reg_0_ ( .D(n252), .CLK(CLOCK), .RSTB(n427), .Q(n412) );
  DFFARX1 REG1_reg_7_ ( .D(n251), .CLK(CLOCK), .RSTB(n427), .Q(n411) );
  DFFARX1 REG1_reg_6_ ( .D(n250), .CLK(CLOCK), .RSTB(n427), .Q(n410) );
  DFFARX1 REG1_reg_5_ ( .D(n249), .CLK(CLOCK), .RSTB(n427), .Q(n409) );
  DFFARX1 REG1_reg_4_ ( .D(n248), .CLK(CLOCK), .RSTB(n427), .Q(n408) );
  DFFARX1 REG1_reg_3_ ( .D(n247), .CLK(CLOCK), .RSTB(n427), .Q(n407) );
  DFFARX1 REG1_reg_2_ ( .D(n246), .CLK(CLOCK), .RSTB(n427), .Q(n406) );
  DFFARX1 REG1_reg_1_ ( .D(n245), .CLK(CLOCK), .RSTB(n427), .Q(n405) );
  DFFARX1 REG1_reg_0_ ( .D(n244), .CLK(CLOCK), .RSTB(n427), .Q(n404) );
  DFFARX1 REG2_reg_7_ ( .D(n243), .CLK(CLOCK), .RSTB(n427), .Q(n403) );
  DFFARX1 REG2_reg_6_ ( .D(n242), .CLK(CLOCK), .RSTB(n427), .Q(n402) );
  DFFARX1 REG2_reg_5_ ( .D(n241), .CLK(CLOCK), .RSTB(n427), .Q(n401) );
  DFFARX1 REG2_reg_4_ ( .D(n240), .CLK(CLOCK), .RSTB(n427), .Q(n400) );
  DFFARX1 REG2_reg_3_ ( .D(n239), .CLK(CLOCK), .RSTB(n427), .Q(n399) );
  DFFARX1 REG2_reg_2_ ( .D(n238), .CLK(CLOCK), .RSTB(n427), .Q(n398) );
  DFFARX1 REG2_reg_1_ ( .D(n237), .CLK(CLOCK), .RSTB(n427), .Q(n397) );
  DFFARX1 REG2_reg_0_ ( .D(n236), .CLK(CLOCK), .RSTB(n427), .Q(n396) );
  DFFARX1 REG3_reg_7_ ( .D(n235), .CLK(CLOCK), .RSTB(n427), .Q(n395) );
  DFFARX1 REG3_reg_6_ ( .D(n234), .CLK(CLOCK), .RSTB(n427), .Q(n394) );
  DFFARX1 REG3_reg_5_ ( .D(n233), .CLK(CLOCK), .RSTB(n427), .Q(n393) );
  DFFARX1 REG3_reg_4_ ( .D(n232), .CLK(CLOCK), .RSTB(n427), .Q(n392) );
  DFFARX1 REG3_reg_3_ ( .D(n231), .CLK(CLOCK), .RSTB(n427), .Q(n391) );
  DFFARX1 REG3_reg_2_ ( .D(n230), .CLK(CLOCK), .RSTB(n427), .Q(n390) );
  DFFARX1 REG3_reg_1_ ( .D(n229), .CLK(CLOCK), .RSTB(n427), .Q(n389) );
  DFFARX1 REG3_reg_0_ ( .D(n228), .CLK(CLOCK), .RSTB(n427), .Q(n388) );
  DFFARX1 REG4_reg_7_ ( .D(n227), .CLK(CLOCK), .RSTB(n427), .Q(n387) );
  DFFARX1 REG4_reg_6_ ( .D(n226), .CLK(CLOCK), .RSTB(n427), .Q(n386) );
  DFFARX1 REG4_reg_5_ ( .D(n225), .CLK(CLOCK), .RSTB(n427), .Q(n385) );
  DFFARX1 REG4_reg_4_ ( .D(n224), .CLK(CLOCK), .RSTB(n427), .Q(n384) );
  DFFARX1 REG4_reg_3_ ( .D(n223), .CLK(CLOCK), .RSTB(n427), .Q(n383) );
  DFFARX1 REG4_reg_2_ ( .D(n222), .CLK(CLOCK), .RSTB(n427), .Q(n382) );
  DFFARX1 REG4_reg_1_ ( .D(n221), .CLK(CLOCK), .RSTB(n427), .Q(n381) );
  DFFARX1 REG4_reg_0_ ( .D(n220), .CLK(CLOCK), .RSTB(n427), .Q(n380) );
  DFFARX1 DATA_OUT_reg_7_ ( .D(n219), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[7]) );
  DFFARX1 DATA_OUT_reg_6_ ( .D(n218), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[6]) );
  DFFARX1 DATA_OUT_reg_5_ ( .D(n217), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[5]) );
  DFFARX1 DATA_OUT_reg_4_ ( .D(n216), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[4]) );
  DFFARX1 DATA_OUT_reg_3_ ( .D(n215), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[3]) );
  DFFARX1 DATA_OUT_reg_2_ ( .D(n214), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[2]) );
  DFFARX1 DATA_OUT_reg_1_ ( .D(n213), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[1]) );
  DFFARX1 DATA_OUT_reg_0_ ( .D(n212), .CLK(CLOCK), .RSTB(n427), .Q(DATA_OUT[0]) );
  FADDX1 r180_U1_8 ( .A(U3_U3_Z_8), .B(U3_U4_Z_8), .CI(r180_carry_8_), .S(N40)
         );
  FADDX1 r34_U1_1 ( .A(U3_U3_Z_1), .B(U3_U4_Z_1), .CI(n313), .CO(r34_carry[2]), 
        .S(N83) );
  FADDX1 r34_U1_2 ( .A(U3_U3_Z_2), .B(U3_U4_Z_2), .CI(r34_carry[2]), .CO(
        r34_carry[3]), .S(N84) );
  FADDX1 r34_U1_3 ( .A(U3_U3_Z_3), .B(U3_U4_Z_3), .CI(r34_carry[3]), .CO(
        r34_carry[4]), .S(N85) );
  FADDX1 r34_U1_4 ( .A(U3_U3_Z_4), .B(U3_U4_Z_4), .CI(r34_carry[4]), .CO(
        r34_carry[5]), .S(N86) );
  FADDX1 r34_U1_5 ( .A(U3_U3_Z_5), .B(U3_U4_Z_5), .CI(r34_carry[5]), .CO(
        r34_carry[6]), .S(N87) );
  FADDX1 r34_U1_6 ( .A(U3_U3_Z_6), .B(U3_U4_Z_6), .CI(r34_carry[6]), .S(N88)
         );
  XNOR2X1 U181 ( .IN1(U3_U4_Z_0), .IN2(U3_U3_Z_0), .Q(n279) );
  XNOR2X1 U182 ( .IN1(n426), .IN2(n279), .Q(n286) );
  XNOR2X1 U183 ( .IN1(n425), .IN2(n311), .Q(n287) );
  XNOR2X1 U184 ( .IN1(n424), .IN2(n306), .Q(n288) );
  XNOR2X1 U185 ( .IN1(n423), .IN2(n307), .Q(n289) );
  XNOR2X1 U186 ( .IN1(n422), .IN2(n308), .Q(n290) );
  XNOR2X1 U187 ( .IN1(n421), .IN2(n309), .Q(n291) );
  INVX4 U188 ( .INP(RESET), .ZN(n427) );
  INVX2 U189 ( .INP(n363), .ZN(n337) );
  NAND2X1 U190 ( .IN1(stato[1]), .IN2(n284), .QN(n363) );
  INVX2 U191 ( .INP(N171), .ZN(n333) );
  AND2X1 U192 ( .IN1(n288), .IN2(n310), .Q(n301) );
  AND2X1 U193 ( .IN1(n289), .IN2(n301), .Q(n302) );
  AND2X1 U194 ( .IN1(n290), .IN2(n302), .Q(n303) );
  AND2X1 U195 ( .IN1(n291), .IN2(n303), .Q(n304) );
  NAND2X1 U196 ( .IN1(n305), .IN2(n304), .QN(n314) );
  AND2X1 U197 ( .IN1(n421), .IN2(n309), .Q(n305) );
  AND2X1 U198 ( .IN1(n425), .IN2(n311), .Q(n306) );
  AND2X1 U199 ( .IN1(n424), .IN2(n306), .Q(n307) );
  AND2X1 U200 ( .IN1(n423), .IN2(n307), .Q(n308) );
  AND2X1 U201 ( .IN1(n422), .IN2(n308), .Q(n309) );
  AND2X1 U202 ( .IN1(n287), .IN2(n312), .Q(n310) );
  AND2X1 U203 ( .IN1(n426), .IN2(n279), .Q(n311) );
  AND2X1 U204 ( .IN1(n286), .IN2(n420), .Q(n312) );
  AND2X1 U205 ( .IN1(U3_U4_Z_0), .IN2(U3_U3_Z_0), .Q(n313) );
  XNOR2X1 U206 ( .IN1(n305), .IN2(n314), .Q(N125) );
  XOR2X1 U207 ( .IN1(n286), .IN2(n420), .Q(N118) );
  XOR2X1 U208 ( .IN1(n287), .IN2(n312), .Q(N119) );
  XOR2X1 U209 ( .IN1(n288), .IN2(n310), .Q(N120) );
  XOR2X1 U210 ( .IN1(n289), .IN2(n301), .Q(N121) );
  XOR2X1 U211 ( .IN1(n290), .IN2(n302), .Q(N122) );
  XOR2X1 U212 ( .IN1(n291), .IN2(n303), .Q(N123) );
  XOR2X1 U213 ( .IN1(n305), .IN2(n304), .Q(N124) );
  AO22X1 U214 ( .IN1(n315), .IN2(U3_U3_Z_8), .IN3(n316), .IN4(U3_U4_Z_8), .Q(
        r180_carry_8_) );
  OR2X1 U215 ( .IN1(n315), .IN2(U3_U3_Z_8), .Q(n316) );
  AO22X1 U216 ( .IN1(n317), .IN2(U3_U3_Z_6), .IN3(n318), .IN4(U3_U4_Z_6), .Q(
        n315) );
  OR2X1 U217 ( .IN1(n317), .IN2(U3_U3_Z_6), .Q(n318) );
  AO22X1 U218 ( .IN1(n319), .IN2(U3_U3_Z_5), .IN3(n320), .IN4(U3_U4_Z_5), .Q(
        n317) );
  OR2X1 U219 ( .IN1(n319), .IN2(U3_U3_Z_5), .Q(n320) );
  AO22X1 U220 ( .IN1(n321), .IN2(U3_U3_Z_4), .IN3(n322), .IN4(U3_U4_Z_4), .Q(
        n319) );
  OR2X1 U221 ( .IN1(n321), .IN2(U3_U3_Z_4), .Q(n322) );
  AO22X1 U222 ( .IN1(n323), .IN2(U3_U3_Z_3), .IN3(n324), .IN4(U3_U4_Z_3), .Q(
        n321) );
  OR2X1 U223 ( .IN1(n323), .IN2(U3_U3_Z_3), .Q(n324) );
  AO22X1 U224 ( .IN1(n325), .IN2(U3_U3_Z_2), .IN3(n326), .IN4(U3_U4_Z_2), .Q(
        n323) );
  OR2X1 U225 ( .IN1(n325), .IN2(U3_U3_Z_2), .Q(n326) );
  AO22X1 U226 ( .IN1(U3_U3_Z_1), .IN2(U3_U4_Z_1), .IN3(n327), .IN4(n328), .Q(
        n325) );
  OR2X1 U227 ( .IN1(U3_U3_Z_1), .IN2(U3_U4_Z_1), .Q(n328) );
  AND2X1 U228 ( .IN1(U3_U4_Z_0), .IN2(U3_U3_Z_0), .Q(n327) );
  OR3X1 U229 ( .IN1(n305), .IN2(n279), .IN3(n329), .Q(n420) );
  INVX0 U230 ( .INP(N88), .ZN(n421) );
  INVX0 U231 ( .INP(N87), .ZN(n422) );
  INVX0 U232 ( .INP(N86), .ZN(n423) );
  INVX0 U233 ( .INP(N85), .ZN(n424) );
  INVX0 U234 ( .INP(N84), .ZN(n425) );
  INVX0 U235 ( .INP(N83), .ZN(n426) );
  MUX21X1 U236 ( .IN1(n299), .IN2(DATA_IN[0]), .S(n330), .Q(n275) );
  MUX21X1 U237 ( .IN1(n296), .IN2(DATA_IN[1]), .S(n330), .Q(n274) );
  MUX21X1 U238 ( .IN1(n297), .IN2(DATA_IN[2]), .S(n330), .Q(n273) );
  MUX21X1 U239 ( .IN1(n294), .IN2(DATA_IN[3]), .S(n330), .Q(n272) );
  MUX21X1 U240 ( .IN1(n293), .IN2(DATA_IN[4]), .S(n330), .Q(n271) );
  MUX21X1 U241 ( .IN1(n295), .IN2(DATA_IN[5]), .S(n330), .Q(n270) );
  MUX21X1 U242 ( .IN1(n281), .IN2(DATA_IN[6]), .S(n330), .Q(n269) );
  MUX21X1 U243 ( .IN1(n280), .IN2(DATA_IN[7]), .S(n330), .Q(n268) );
  AOI21X1 U244 ( .IN1(n331), .IN2(n332), .IN3(n333), .QN(n330) );
  MUX21X1 U245 ( .IN1(DATA_IN[0]), .IN2(n300), .S(n334), .Q(n267) );
  MUX21X1 U246 ( .IN1(DATA_IN[7]), .IN2(n285), .S(n334), .Q(n266) );
  MUX21X1 U247 ( .IN1(DATA_IN[6]), .IN2(n282), .S(n334), .Q(n265) );
  MUX21X1 U248 ( .IN1(DATA_IN[5]), .IN2(n277), .S(n334), .Q(n264) );
  MUX21X1 U249 ( .IN1(DATA_IN[4]), .IN2(n278), .S(n334), .Q(n263) );
  MUX21X1 U250 ( .IN1(DATA_IN[3]), .IN2(n283), .S(n334), .Q(n262) );
  MUX21X1 U251 ( .IN1(DATA_IN[2]), .IN2(n298), .S(n334), .Q(n261) );
  MUX21X1 U252 ( .IN1(DATA_IN[1]), .IN2(n292), .S(n334), .Q(n260) );
  AND2X1 U253 ( .IN1(n332), .IN2(n335), .Q(n334) );
  NAND3X0 U254 ( .IN1(n331), .IN2(n336), .IN3(n337), .QN(n335) );
  AO21X1 U255 ( .IN1(n202), .IN2(DATA_IN[7]), .IN3(n338), .Q(n336) );
  OA22X1 U256 ( .IN1(DATA_IN[7]), .IN2(n202), .IN3(n339), .IN4(n340), .Q(n338)
         );
  NOR2X0 U257 ( .IN1(n201), .IN2(DATA_IN[6]), .QN(n340) );
  OA221X1 U258 ( .IN1(n341), .IN2(n277), .IN3(n342), .IN4(n282), .IN5(n343), 
        .Q(n339) );
  AO221X1 U259 ( .IN1(n344), .IN2(n278), .IN3(n341), .IN4(n277), .IN5(n345), 
        .Q(n343) );
  OA221X1 U260 ( .IN1(n346), .IN2(n283), .IN3(n344), .IN4(n278), .IN5(n347), 
        .Q(n345) );
  OAI222X1 U261 ( .IN1(n348), .IN2(DATA_IN[2]), .IN3(n349), .IN4(n197), .IN5(
        DATA_IN[3]), .IN6(n198), .QN(n347) );
  AND2X1 U262 ( .IN1(n348), .IN2(DATA_IN[2]), .Q(n349) );
  OA22X1 U263 ( .IN1(DATA_IN[1]), .IN2(n196), .IN3(n350), .IN4(DATA_IN[0]), 
        .Q(n348) );
  AND2X1 U264 ( .IN1(n196), .IN2(DATA_IN[1]), .Q(n350) );
  INVX0 U265 ( .INP(DATA_IN[3]), .ZN(n346) );
  INVX0 U266 ( .INP(DATA_IN[4]), .ZN(n344) );
  INVX0 U267 ( .INP(DATA_IN[5]), .ZN(n341) );
  AOI21X1 U268 ( .IN1(n351), .IN2(n280), .IN3(n352), .QN(n331) );
  OA22X1 U269 ( .IN1(n351), .IN2(n280), .IN3(n353), .IN4(n354), .Q(n352) );
  NOR2X0 U270 ( .IN1(n342), .IN2(n281), .QN(n354) );
  INVX0 U271 ( .INP(DATA_IN[6]), .ZN(n342) );
  OA221X1 U272 ( .IN1(DATA_IN[6]), .IN2(n205), .IN3(DATA_IN[5]), .IN4(n206), 
        .IN5(n355), .Q(n353) );
  AO221X1 U273 ( .IN1(n207), .IN2(DATA_IN[4]), .IN3(n206), .IN4(DATA_IN[5]), 
        .IN5(n356), .Q(n355) );
  OA221X1 U274 ( .IN1(DATA_IN[3]), .IN2(n208), .IN3(DATA_IN[4]), .IN4(n207), 
        .IN5(n357), .Q(n356) );
  AO222X1 U275 ( .IN1(DATA_IN[2]), .IN2(n358), .IN3(n209), .IN4(n359), .IN5(
        n208), .IN6(DATA_IN[3]), .Q(n357) );
  OR2X1 U276 ( .IN1(n358), .IN2(DATA_IN[2]), .Q(n359) );
  AO22X1 U277 ( .IN1(n210), .IN2(DATA_IN[1]), .IN3(n360), .IN4(n211), .Q(n358)
         );
  OA21X1 U278 ( .IN1(DATA_IN[1]), .IN2(n210), .IN3(DATA_IN[0]), .Q(n360) );
  INVX0 U279 ( .INP(DATA_IN[7]), .ZN(n351) );
  AO22X1 U280 ( .IN1(n419), .IN2(n361), .IN3(n362), .IN4(DATA_IN[7]), .Q(n259)
         );
  AO22X1 U281 ( .IN1(n418), .IN2(n361), .IN3(n362), .IN4(DATA_IN[6]), .Q(n258)
         );
  AO22X1 U282 ( .IN1(n417), .IN2(n361), .IN3(n362), .IN4(DATA_IN[5]), .Q(n257)
         );
  AO22X1 U283 ( .IN1(n416), .IN2(n361), .IN3(n362), .IN4(DATA_IN[4]), .Q(n256)
         );
  AO22X1 U284 ( .IN1(n415), .IN2(n361), .IN3(n362), .IN4(DATA_IN[3]), .Q(n255)
         );
  AO22X1 U285 ( .IN1(n414), .IN2(n361), .IN3(n362), .IN4(DATA_IN[2]), .Q(n254)
         );
  AO22X1 U286 ( .IN1(n413), .IN2(n361), .IN3(n362), .IN4(DATA_IN[1]), .Q(n253)
         );
  AO22X1 U287 ( .IN1(n412), .IN2(n361), .IN3(n362), .IN4(DATA_IN[0]), .Q(n252)
         );
  NOR2X0 U288 ( .IN1(n361), .IN2(n363), .QN(n362) );
  AO21X1 U289 ( .IN1(n332), .IN2(n364), .IN3(n333), .Q(n361) );
  AO22X1 U290 ( .IN1(n411), .IN2(n333), .IN3(n337), .IN4(DATA_IN[7]), .Q(n251)
         );
  AO22X1 U291 ( .IN1(n410), .IN2(n333), .IN3(n337), .IN4(DATA_IN[6]), .Q(n250)
         );
  AO22X1 U292 ( .IN1(n409), .IN2(n333), .IN3(n337), .IN4(DATA_IN[5]), .Q(n249)
         );
  AO22X1 U293 ( .IN1(n408), .IN2(n333), .IN3(n337), .IN4(DATA_IN[4]), .Q(n248)
         );
  AO22X1 U294 ( .IN1(n407), .IN2(n333), .IN3(n337), .IN4(DATA_IN[3]), .Q(n247)
         );
  AO22X1 U295 ( .IN1(n406), .IN2(n333), .IN3(n337), .IN4(DATA_IN[2]), .Q(n246)
         );
  AO22X1 U296 ( .IN1(n405), .IN2(n333), .IN3(n337), .IN4(DATA_IN[1]), .Q(n245)
         );
  AO22X1 U297 ( .IN1(n404), .IN2(n333), .IN3(n337), .IN4(DATA_IN[0]), .Q(n244)
         );
  AO22X1 U298 ( .IN1(n403), .IN2(n333), .IN3(n411), .IN4(n337), .Q(n243) );
  AO22X1 U299 ( .IN1(n402), .IN2(n333), .IN3(n410), .IN4(n337), .Q(n242) );
  AO22X1 U300 ( .IN1(n401), .IN2(n333), .IN3(n409), .IN4(n337), .Q(n241) );
  AO22X1 U301 ( .IN1(n400), .IN2(n333), .IN3(n408), .IN4(n337), .Q(n240) );
  AO22X1 U302 ( .IN1(n399), .IN2(n333), .IN3(n407), .IN4(n337), .Q(n239) );
  AO22X1 U303 ( .IN1(n398), .IN2(n333), .IN3(n406), .IN4(n337), .Q(n238) );
  AO22X1 U304 ( .IN1(n397), .IN2(n333), .IN3(n405), .IN4(n337), .Q(n237) );
  AO22X1 U305 ( .IN1(n396), .IN2(n333), .IN3(n404), .IN4(n337), .Q(n236) );
  AO22X1 U306 ( .IN1(n395), .IN2(n333), .IN3(n403), .IN4(n337), .Q(n235) );
  AO22X1 U307 ( .IN1(n394), .IN2(n333), .IN3(n402), .IN4(n337), .Q(n234) );
  AO22X1 U308 ( .IN1(n393), .IN2(n333), .IN3(n401), .IN4(n337), .Q(n233) );
  AO22X1 U309 ( .IN1(n392), .IN2(n333), .IN3(n400), .IN4(n337), .Q(n232) );
  AO22X1 U310 ( .IN1(n391), .IN2(n333), .IN3(n399), .IN4(n337), .Q(n231) );
  AO22X1 U311 ( .IN1(n390), .IN2(n333), .IN3(n398), .IN4(n337), .Q(n230) );
  AO22X1 U312 ( .IN1(n389), .IN2(n333), .IN3(n397), .IN4(n337), .Q(n229) );
  AO22X1 U313 ( .IN1(n388), .IN2(n333), .IN3(n396), .IN4(n337), .Q(n228) );
  AO22X1 U314 ( .IN1(n387), .IN2(n333), .IN3(n395), .IN4(n337), .Q(n227) );
  AO22X1 U315 ( .IN1(n386), .IN2(n333), .IN3(n394), .IN4(n337), .Q(n226) );
  AO22X1 U316 ( .IN1(n385), .IN2(n333), .IN3(n393), .IN4(n337), .Q(n225) );
  AO22X1 U317 ( .IN1(n384), .IN2(n333), .IN3(n392), .IN4(n337), .Q(n224) );
  AO22X1 U318 ( .IN1(n383), .IN2(n333), .IN3(n391), .IN4(n337), .Q(n223) );
  AO22X1 U319 ( .IN1(n382), .IN2(n333), .IN3(n390), .IN4(n337), .Q(n222) );
  AO22X1 U320 ( .IN1(n381), .IN2(n333), .IN3(n389), .IN4(n337), .Q(n221) );
  AO22X1 U321 ( .IN1(n380), .IN2(n333), .IN3(n388), .IN4(n337), .Q(n220) );
  AO221X1 U322 ( .IN1(n365), .IN2(n387), .IN3(n333), .IN4(DATA_OUT[7]), .IN5(
        n366), .Q(n219) );
  AO22X1 U323 ( .IN1(N125), .IN2(n367), .IN3(n368), .IN4(n419), .Q(n366) );
  AO221X1 U324 ( .IN1(n365), .IN2(n386), .IN3(n333), .IN4(DATA_OUT[6]), .IN5(
        n369), .Q(n218) );
  AO22X1 U325 ( .IN1(N124), .IN2(n367), .IN3(n368), .IN4(n418), .Q(n369) );
  AO221X1 U326 ( .IN1(n365), .IN2(n385), .IN3(n333), .IN4(DATA_OUT[5]), .IN5(
        n370), .Q(n217) );
  AO222X1 U327 ( .IN1(n371), .IN2(N88), .IN3(n368), .IN4(n417), .IN5(N123), 
        .IN6(n367), .Q(n370) );
  AO221X1 U328 ( .IN1(n365), .IN2(n384), .IN3(n333), .IN4(DATA_OUT[4]), .IN5(
        n372), .Q(n216) );
  AO222X1 U329 ( .IN1(n371), .IN2(N87), .IN3(n368), .IN4(n416), .IN5(N122), 
        .IN6(n367), .Q(n372) );
  AO221X1 U330 ( .IN1(n365), .IN2(n383), .IN3(n333), .IN4(DATA_OUT[3]), .IN5(
        n373), .Q(n215) );
  AO222X1 U331 ( .IN1(n371), .IN2(N86), .IN3(n368), .IN4(n415), .IN5(N121), 
        .IN6(n367), .Q(n373) );
  AO221X1 U332 ( .IN1(n365), .IN2(n382), .IN3(n333), .IN4(DATA_OUT[2]), .IN5(
        n374), .Q(n214) );
  AO222X1 U333 ( .IN1(n371), .IN2(N85), .IN3(n368), .IN4(n414), .IN5(N120), 
        .IN6(n367), .Q(n374) );
  AO221X1 U334 ( .IN1(n365), .IN2(n381), .IN3(n333), .IN4(DATA_OUT[1]), .IN5(
        n375), .Q(n213) );
  AO222X1 U335 ( .IN1(n371), .IN2(N84), .IN3(n368), .IN4(n413), .IN5(N119), 
        .IN6(n367), .Q(n375) );
  AO221X1 U336 ( .IN1(n365), .IN2(n380), .IN3(n333), .IN4(DATA_OUT[0]), .IN5(
        n376), .Q(n212) );
  AO222X1 U337 ( .IN1(n371), .IN2(N83), .IN3(n368), .IN4(n412), .IN5(N118), 
        .IN6(n367), .Q(n376) );
  NOR3X0 U338 ( .IN1(n363), .IN2(n329), .IN3(n377), .QN(n367) );
  INVX0 U339 ( .INP(N40), .ZN(n377) );
  NOR3X0 U340 ( .IN1(ENABLE), .IN2(RESTART), .IN3(n363), .QN(n368) );
  NOR3X0 U341 ( .IN1(n329), .IN2(N40), .IN3(n363), .QN(n371) );
  OA21X1 U342 ( .IN1(n364), .IN2(AVERAGE), .IN3(n378), .Q(n329) );
  AND4X1 U343 ( .IN1(AVERAGE), .IN2(n337), .IN3(ENABLE), .IN4(n378), .Q(n365)
         );
  INVX0 U344 ( .INP(RESTART), .ZN(n378) );
  AO22X1 U345 ( .IN1(RESTART), .IN2(n285), .IN3(n387), .IN4(n379), .Q(
        U3_U4_Z_8) );
  AO22X1 U346 ( .IN1(RESTART), .IN2(n282), .IN3(n386), .IN4(n379), .Q(
        U3_U4_Z_6) );
  AO22X1 U347 ( .IN1(RESTART), .IN2(n277), .IN3(n385), .IN4(n379), .Q(
        U3_U4_Z_5) );
  AO22X1 U348 ( .IN1(RESTART), .IN2(n278), .IN3(n384), .IN4(n379), .Q(
        U3_U4_Z_4) );
  AO22X1 U349 ( .IN1(RESTART), .IN2(n283), .IN3(n383), .IN4(n379), .Q(
        U3_U4_Z_3) );
  AO22X1 U350 ( .IN1(RESTART), .IN2(n298), .IN3(n382), .IN4(n379), .Q(
        U3_U4_Z_2) );
  AO22X1 U351 ( .IN1(RESTART), .IN2(n292), .IN3(n381), .IN4(n379), .Q(
        U3_U4_Z_1) );
  AO22X1 U352 ( .IN1(RESTART), .IN2(n300), .IN3(n380), .IN4(n379), .Q(
        U3_U4_Z_0) );
  AO22X1 U353 ( .IN1(RESTART), .IN2(n280), .IN3(DATA_IN[7]), .IN4(n379), .Q(
        U3_U3_Z_8) );
  AO22X1 U354 ( .IN1(RESTART), .IN2(n281), .IN3(DATA_IN[6]), .IN4(n379), .Q(
        U3_U3_Z_6) );
  AO22X1 U355 ( .IN1(RESTART), .IN2(n295), .IN3(DATA_IN[5]), .IN4(n379), .Q(
        U3_U3_Z_5) );
  AO22X1 U356 ( .IN1(RESTART), .IN2(n293), .IN3(DATA_IN[4]), .IN4(n379), .Q(
        U3_U3_Z_4) );
  AO22X1 U357 ( .IN1(RESTART), .IN2(n294), .IN3(DATA_IN[3]), .IN4(n379), .Q(
        U3_U3_Z_3) );
  AO22X1 U358 ( .IN1(RESTART), .IN2(n297), .IN3(DATA_IN[2]), .IN4(n379), .Q(
        U3_U3_Z_2) );
  AO22X1 U359 ( .IN1(RESTART), .IN2(n296), .IN3(DATA_IN[1]), .IN4(n379), .Q(
        U3_U3_Z_1) );
  AO22X1 U360 ( .IN1(RESTART), .IN2(n299), .IN3(DATA_IN[0]), .IN4(n379), .Q(
        U3_U3_Z_0) );
  NOR3X0 U361 ( .IN1(AVERAGE), .IN2(RESTART), .IN3(n364), .QN(n379) );
  INVX0 U362 ( .INP(ENABLE), .ZN(n364) );
  NAND2X0 U363 ( .IN1(n363), .IN2(n332), .QN(N171) );
  OR2X1 U364 ( .IN1(n284), .IN2(stato[1]), .Q(n332) );
  NOR2X0 U365 ( .IN1(stato[1]), .IN2(stato[0]), .QN(N170) );
endmodule

