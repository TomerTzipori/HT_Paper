/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:08:21 2023
/////////////////////////////////////////////////////////////


module b11 ( x_in, stbi, clock, reset, x_out );
  input [5:0] x_in;
  output [5:0] x_out;
  input stbi, clock, reset;
  wire   N70, N71, N72, N73, N242, N243, N244, N245, U3_U5_Z_0, U3_U5_Z_1,
         U3_U5_Z_2, U3_U5_Z_3, U3_U5_Z_4, U3_U5_Z_5, U3_U5_Z_6, U3_U5_Z_7,
         U3_U5_Z_8, U3_U7_Z_0, n170, n176, n182, n183, n184, n185, n186, n187,
         n188, n189, n190, n191, n192, n193, n194, n195, n196, n197, n198,
         n199, n200, n201, n202, n203, n204, n205, n206, n207, n209,
         r147_carry_1_, r147_carry_2_, r147_carry_3_, r147_carry_4_,
         r147_carry_5_, r147_carry_6_, r147_carry_7_, r147_carry_8_,
         r147_SUM_0_, r147_SUM_1_, r147_SUM_2_, r147_SUM_3_, r147_SUM_4_,
         r147_SUM_5_, r147_SUM_6_, r147_SUM_7_, r147_SUM_8_, n211, n212, n213,
         n214, n215, n216, n217, n218, n219, n220, n221, n222, n223, n224,
         n225, n226, n227, n228, n229, n230, n231, n232, n233, n234, n235,
         n236, n237, n238, n239, n240, n241, n242, n243, n244, n245, n246,
         n247, n248, n249, n250, n251, n252, n253, n254, n255, n256, n257,
         n258, n259, n260, n261, n262, n263, n264, n265, n266, n267, n268,
         n269, n270, n271, n272, n273, n274, n275, n276, n277, n278, n279,
         n280, n281, n282, n283, n284, n285, n286, n287, n288, n289, n290,
         n291, n292, n293, n294, n295, n296, n297, n298, n299, n300, n301,
         n302, n303, n304, n305, n306, n307, n308, n309, n310, n311, n312,
         n313, n314, n315, n316, n317, n318, n319, n320, n321, n322, n323,
         n324, n325, n326, n327, n328, n329, n330, n331;
  wire   [5:0] r_in;
  wire   [4:0] cont;
  wire   [8:0] r147_B_AS;
  wire   [5:2] add_53_carry;

  DFFARX1 cont1_reg_0_ ( .D(n209), .CLK(clock), .RSTB(n331), .Q(n322), .QN(
        n224) );
  DFFARX1 cont1_reg_8_ ( .D(n207), .CLK(clock), .RSTB(n331), .Q(n324), .QN(
        n223) );
  DFFARX1 stato_reg_1_ ( .D(N243), .CLK(clock), .RSTB(n331), .Q(n327), .QN(
        n211) );
  DFFARX1 r_in_reg_0_ ( .D(n206), .CLK(clock), .RSTB(n331), .Q(r_in[0]), .QN(
        n217) );
  DFFARX1 r_in_reg_5_ ( .D(n205), .CLK(clock), .RSTB(n331), .Q(r_in[5]) );
  DFFARX1 r_in_reg_4_ ( .D(n204), .CLK(clock), .RSTB(n331), .Q(r_in[4]), .QN(
        n225) );
  DFFARX1 r_in_reg_3_ ( .D(n203), .CLK(clock), .RSTB(n331), .Q(r_in[3]), .QN(
        n213) );
  DFFARX1 r_in_reg_2_ ( .D(n202), .CLK(clock), .RSTB(n331), .Q(r_in[2]), .QN(
        n215) );
  DFFARX1 r_in_reg_1_ ( .D(n201), .CLK(clock), .RSTB(n331), .Q(r_in[1]), .QN(
        n220) );
  DFFARX1 stato_reg_0_ ( .D(N242), .CLK(clock), .RSTB(n331), .Q(n330), .QN(
        n212) );
  DFFARX1 stato_reg_2_ ( .D(N244), .CLK(clock), .RSTB(n331), .Q(n328), .QN(
        n214) );
  DFFARX1 stato_reg_3_ ( .D(N245), .CLK(clock), .RSTB(n331), .Q(n329), .QN(
        n219) );
  DFFARX1 cont1_reg_7_ ( .D(n200), .CLK(clock), .RSTB(n331), .Q(n325), .QN(
        n221) );
  DFFARX1 cont_reg_5_ ( .D(n199), .CLK(clock), .RSTB(n331), .Q(n216) );
  DFFARX1 cont_reg_0_ ( .D(n198), .CLK(clock), .RSTB(n331), .Q(cont[0]), .QN(
        n176) );
  DFFARX1 cont_reg_1_ ( .D(n197), .CLK(clock), .RSTB(n331), .Q(cont[1]) );
  DFFARX1 cont1_reg_1_ ( .D(n196), .CLK(clock), .RSTB(n331), .Q(n321) );
  DFFARX1 cont_reg_2_ ( .D(n195), .CLK(clock), .RSTB(n331), .Q(cont[2]), .QN(
        n170) );
  DFFARX1 cont1_reg_2_ ( .D(n194), .CLK(clock), .RSTB(n331), .Q(n320), .QN(
        n222) );
  DFFARX1 cont_reg_3_ ( .D(n193), .CLK(clock), .RSTB(n331), .Q(cont[3]) );
  DFFARX1 cont1_reg_3_ ( .D(n192), .CLK(clock), .RSTB(n331), .Q(n319) );
  DFFARX1 cont_reg_4_ ( .D(n191), .CLK(clock), .RSTB(n331), .Q(cont[4]) );
  DFFARX1 cont1_reg_4_ ( .D(n190), .CLK(clock), .RSTB(n331), .Q(n318) );
  DFFARX1 cont1_reg_5_ ( .D(n189), .CLK(clock), .RSTB(n331), .Q(n323) );
  DFFARX1 cont1_reg_6_ ( .D(n188), .CLK(clock), .RSTB(n331), .Q(n326), .QN(
        n218) );
  DFFARX1 x_out_reg_5_ ( .D(n187), .CLK(clock), .RSTB(n331), .Q(x_out[5]) );
  DFFARX1 x_out_reg_4_ ( .D(n186), .CLK(clock), .RSTB(n331), .Q(x_out[4]) );
  DFFARX1 x_out_reg_3_ ( .D(n185), .CLK(clock), .RSTB(n331), .Q(x_out[3]) );
  DFFARX1 x_out_reg_2_ ( .D(n184), .CLK(clock), .RSTB(n331), .Q(x_out[2]) );
  DFFARX1 x_out_reg_1_ ( .D(n183), .CLK(clock), .RSTB(n331), .Q(x_out[1]) );
  DFFARX1 x_out_reg_0_ ( .D(n182), .CLK(clock), .RSTB(n331), .Q(x_out[0]) );
  FADDX1 r147_U1_0 ( .A(U3_U5_Z_0), .B(r147_B_AS[0]), .CI(U3_U7_Z_0), .CO(
        r147_carry_1_), .S(r147_SUM_0_) );
  FADDX1 r147_U1_1 ( .A(U3_U5_Z_1), .B(r147_B_AS[1]), .CI(r147_carry_1_), .CO(
        r147_carry_2_), .S(r147_SUM_1_) );
  FADDX1 r147_U1_2 ( .A(U3_U5_Z_2), .B(r147_B_AS[2]), .CI(r147_carry_2_), .CO(
        r147_carry_3_), .S(r147_SUM_2_) );
  FADDX1 r147_U1_3 ( .A(U3_U5_Z_3), .B(r147_B_AS[3]), .CI(r147_carry_3_), .CO(
        r147_carry_4_), .S(r147_SUM_3_) );
  FADDX1 r147_U1_4 ( .A(U3_U5_Z_4), .B(r147_B_AS[4]), .CI(r147_carry_4_), .CO(
        r147_carry_5_), .S(r147_SUM_4_) );
  FADDX1 r147_U1_5 ( .A(U3_U5_Z_5), .B(r147_B_AS[5]), .CI(r147_carry_5_), .CO(
        r147_carry_6_), .S(r147_SUM_5_) );
  FADDX1 r147_U1_6 ( .A(U3_U5_Z_6), .B(r147_B_AS[6]), .CI(r147_carry_6_), .CO(
        r147_carry_7_), .S(r147_SUM_6_) );
  FADDX1 r147_U1_7 ( .A(U3_U5_Z_7), .B(r147_B_AS[7]), .CI(r147_carry_7_), .CO(
        r147_carry_8_), .S(r147_SUM_7_) );
  FADDX1 r147_U1_8 ( .A(U3_U5_Z_8), .B(r147_B_AS[8]), .CI(r147_carry_8_), .S(
        r147_SUM_8_) );
  HADDX1 add_53_U1_1_1 ( .A0(cont[1]), .B0(cont[0]), .C1(add_53_carry[2]), 
        .SO(N70) );
  HADDX1 add_53_U1_1_2 ( .A0(cont[2]), .B0(add_53_carry[2]), .C1(
        add_53_carry[3]), .SO(N71) );
  HADDX1 add_53_U1_1_3 ( .A0(cont[3]), .B0(add_53_carry[3]), .C1(
        add_53_carry[4]), .SO(N72) );
  HADDX1 add_53_U1_1_4 ( .A0(cont[4]), .B0(add_53_carry[4]), .C1(
        add_53_carry[5]), .SO(N73) );
  INVX2 U164 ( .INP(reset), .ZN(n331) );
  XOR2X1 U165 ( .IN1(n226), .IN2(n227), .Q(r147_B_AS[8]) );
  OA21X1 U166 ( .IN1(n228), .IN2(n223), .IN3(n229), .Q(n227) );
  XOR2X1 U167 ( .IN1(n226), .IN2(n230), .Q(r147_B_AS[7]) );
  OA21X1 U168 ( .IN1(n228), .IN2(n221), .IN3(n229), .Q(n230) );
  XOR2X1 U169 ( .IN1(n226), .IN2(n231), .Q(r147_B_AS[6]) );
  OA21X1 U170 ( .IN1(n228), .IN2(n218), .IN3(n229), .Q(n231) );
  NAND2X0 U171 ( .IN1(n232), .IN2(n323), .QN(n229) );
  XOR2X1 U172 ( .IN1(n226), .IN2(n233), .Q(r147_B_AS[5]) );
  AOI22X1 U173 ( .IN1(n234), .IN2(n323), .IN3(n213), .IN4(n235), .QN(n233) );
  XOR2X1 U174 ( .IN1(U3_U7_Z_0), .IN2(n236), .Q(r147_B_AS[4]) );
  AO222X1 U175 ( .IN1(n318), .IN2(n234), .IN3(n237), .IN4(n213), .IN5(n238), 
        .IN6(n239), .Q(n236) );
  XOR2X1 U176 ( .IN1(n226), .IN2(n240), .Q(r147_B_AS[3]) );
  AOI21X1 U177 ( .IN1(n234), .IN2(n319), .IN3(n241), .QN(n240) );
  XOR2X1 U178 ( .IN1(n226), .IN2(n242), .Q(r147_B_AS[2]) );
  OA21X1 U179 ( .IN1(n243), .IN2(n222), .IN3(n244), .Q(n242) );
  XOR2X1 U180 ( .IN1(U3_U7_Z_0), .IN2(n245), .Q(r147_B_AS[1]) );
  AO221X1 U181 ( .IN1(n321), .IN2(n234), .IN3(n235), .IN4(n215), .IN5(n241), 
        .Q(n245) );
  XOR2X1 U182 ( .IN1(n226), .IN2(n246), .Q(r147_B_AS[0]) );
  OA21X1 U183 ( .IN1(n243), .IN2(n224), .IN3(n244), .Q(n246) );
  INVX0 U184 ( .INP(n234), .ZN(n243) );
  NAND2X0 U185 ( .IN1(n228), .IN2(n247), .QN(n234) );
  NOR2X0 U186 ( .IN1(n248), .IN2(n249), .QN(n228) );
  INVX0 U187 ( .INP(U3_U7_Z_0), .ZN(n226) );
  AO221X1 U188 ( .IN1(r147_SUM_0_), .IN2(n250), .IN3(n251), .IN4(n322), .IN5(
        n252), .Q(n209) );
  AO22X1 U189 ( .IN1(n253), .IN2(cont[0]), .IN3(n254), .IN4(r_in[0]), .Q(n252)
         );
  AO22X1 U190 ( .IN1(n251), .IN2(n324), .IN3(r147_SUM_8_), .IN4(n250), .Q(n207) );
  MUX21X1 U191 ( .IN1(x_in[0]), .IN2(r_in[0]), .S(n255), .Q(n206) );
  MUX21X1 U192 ( .IN1(x_in[5]), .IN2(r_in[5]), .S(n255), .Q(n205) );
  MUX21X1 U193 ( .IN1(x_in[4]), .IN2(r_in[4]), .S(n255), .Q(n204) );
  MUX21X1 U194 ( .IN1(x_in[3]), .IN2(r_in[3]), .S(n255), .Q(n203) );
  MUX21X1 U195 ( .IN1(x_in[2]), .IN2(r_in[2]), .S(n255), .Q(n202) );
  MUX21X1 U196 ( .IN1(x_in[1]), .IN2(r_in[1]), .S(n255), .Q(n201) );
  AO22X1 U197 ( .IN1(n251), .IN2(n325), .IN3(r147_SUM_7_), .IN4(n250), .Q(n200) );
  AO22X1 U198 ( .IN1(n256), .IN2(n216), .IN3(add_53_carry[5]), .IN4(n257), .Q(
        n199) );
  MUX21X1 U199 ( .IN1(n256), .IN2(n257), .S(n176), .Q(n198) );
  AO22X1 U200 ( .IN1(n256), .IN2(cont[1]), .IN3(N70), .IN4(n257), .Q(n197) );
  AO221X1 U201 ( .IN1(r147_SUM_1_), .IN2(n250), .IN3(n251), .IN4(n321), .IN5(
        n258), .Q(n196) );
  AO222X1 U202 ( .IN1(n253), .IN2(cont[1]), .IN3(n254), .IN4(r_in[1]), .IN5(
        n259), .IN6(cont[0]), .Q(n258) );
  AO22X1 U203 ( .IN1(n256), .IN2(cont[2]), .IN3(N71), .IN4(n257), .Q(n195) );
  AO221X1 U204 ( .IN1(r147_SUM_2_), .IN2(n250), .IN3(n251), .IN4(n320), .IN5(
        n260), .Q(n194) );
  AO222X1 U205 ( .IN1(n253), .IN2(cont[2]), .IN3(n254), .IN4(r_in[2]), .IN5(
        n259), .IN6(cont[1]), .Q(n260) );
  AO22X1 U206 ( .IN1(n256), .IN2(cont[3]), .IN3(N72), .IN4(n257), .Q(n193) );
  AO221X1 U207 ( .IN1(r147_SUM_3_), .IN2(n250), .IN3(n251), .IN4(n319), .IN5(
        n261), .Q(n192) );
  AO222X1 U208 ( .IN1(n253), .IN2(cont[3]), .IN3(n254), .IN4(r_in[3]), .IN5(
        n259), .IN6(cont[2]), .Q(n261) );
  AO22X1 U209 ( .IN1(n256), .IN2(cont[4]), .IN3(N73), .IN4(n257), .Q(n191) );
  NOR2X0 U210 ( .IN1(n256), .IN2(n239), .QN(n257) );
  NAND4X0 U211 ( .IN1(n262), .IN2(n212), .IN3(n219), .IN4(n214), .QN(n256) );
  NAND2X0 U212 ( .IN1(n263), .IN2(n327), .QN(n262) );
  AO221X1 U213 ( .IN1(r147_SUM_4_), .IN2(n250), .IN3(n251), .IN4(n318), .IN5(
        n264), .Q(n190) );
  AO222X1 U214 ( .IN1(n253), .IN2(cont[4]), .IN3(n254), .IN4(r_in[4]), .IN5(
        n259), .IN6(cont[3]), .Q(n264) );
  AO221X1 U215 ( .IN1(r147_SUM_5_), .IN2(n250), .IN3(n251), .IN4(n323), .IN5(
        n265), .Q(n189) );
  AO222X1 U216 ( .IN1(n253), .IN2(n216), .IN3(n254), .IN4(r_in[5]), .IN5(n259), 
        .IN6(cont[4]), .Q(n265) );
  AND2X1 U217 ( .IN1(n266), .IN2(n267), .Q(n254) );
  NOR2X0 U218 ( .IN1(n268), .IN2(r_in[0]), .QN(n253) );
  AO222X1 U219 ( .IN1(r147_SUM_6_), .IN2(n250), .IN3(n259), .IN4(n216), .IN5(
        n251), .IN6(n326), .Q(n188) );
  INVX0 U220 ( .INP(n267), .ZN(n251) );
  NOR2X0 U221 ( .IN1(n268), .IN2(n217), .QN(n259) );
  AND2X1 U222 ( .IN1(n269), .IN2(n267), .Q(n250) );
  AO222X1 U223 ( .IN1(n239), .IN2(n269), .IN3(n270), .IN4(n330), .IN5(n271), 
        .IN6(n272), .Q(n267) );
  NOR2X0 U224 ( .IN1(n329), .IN2(n211), .QN(n270) );
  AO222X1 U225 ( .IN1(r147_SUM_5_), .IN2(n232), .IN3(n273), .IN4(n323), .IN5(
        x_out[5]), .IN6(n274), .Q(n187) );
  AO222X1 U226 ( .IN1(r147_SUM_4_), .IN2(n232), .IN3(n273), .IN4(n318), .IN5(
        x_out[4]), .IN6(n274), .Q(n186) );
  AO222X1 U227 ( .IN1(r147_SUM_3_), .IN2(n232), .IN3(n273), .IN4(n319), .IN5(
        x_out[3]), .IN6(n274), .Q(n185) );
  AO222X1 U228 ( .IN1(r147_SUM_2_), .IN2(n232), .IN3(n273), .IN4(n320), .IN5(
        x_out[2]), .IN6(n274), .Q(n184) );
  AO222X1 U229 ( .IN1(r147_SUM_1_), .IN2(n232), .IN3(n273), .IN4(n321), .IN5(
        x_out[1]), .IN6(n274), .Q(n183) );
  AO222X1 U230 ( .IN1(r147_SUM_0_), .IN2(n232), .IN3(n273), .IN4(n322), .IN5(
        x_out[0]), .IN6(n274), .Q(n182) );
  AND2X1 U231 ( .IN1(n239), .IN2(n275), .Q(n273) );
  INVX0 U232 ( .INP(n247), .ZN(n232) );
  AO221X1 U233 ( .IN1(n276), .IN2(n213), .IN3(n248), .IN4(n220), .IN5(n277), 
        .Q(U3_U7_Z_0) );
  OAI21X1 U234 ( .IN1(n278), .IN2(n279), .IN3(n247), .QN(n277) );
  NAND2X0 U235 ( .IN1(n275), .IN2(n279), .QN(n247) );
  NOR2X0 U236 ( .IN1(n280), .IN2(n223), .QN(U3_U5_Z_8) );
  NOR2X0 U237 ( .IN1(n280), .IN2(n221), .QN(U3_U5_Z_7) );
  NOR2X0 U238 ( .IN1(n280), .IN2(n218), .QN(U3_U5_Z_6) );
  AO22X1 U239 ( .IN1(n248), .IN2(r_in[5]), .IN3(n323), .IN4(n281), .Q(
        U3_U5_Z_5) );
  AO221X1 U240 ( .IN1(n318), .IN2(n281), .IN3(n248), .IN4(r_in[4]), .IN5(n249), 
        .Q(U3_U5_Z_4) );
  AO221X1 U241 ( .IN1(n319), .IN2(n281), .IN3(n248), .IN4(r_in[3]), .IN5(n249), 
        .Q(U3_U5_Z_3) );
  AO221X1 U242 ( .IN1(n320), .IN2(n281), .IN3(n248), .IN4(r_in[2]), .IN5(n249), 
        .Q(U3_U5_Z_2) );
  AND3X1 U243 ( .IN1(r_in[3]), .IN2(r_in[2]), .IN3(n235), .Q(n249) );
  AO22X1 U244 ( .IN1(n248), .IN2(r_in[1]), .IN3(n321), .IN4(n281), .Q(
        U3_U5_Z_1) );
  AO22X1 U245 ( .IN1(n248), .IN2(r_in[0]), .IN3(n322), .IN4(n281), .Q(
        U3_U5_Z_0) );
  INVX0 U246 ( .INP(n280), .ZN(n281) );
  NOR2X0 U247 ( .IN1(n241), .IN2(n237), .QN(n280) );
  INVX0 U248 ( .INP(n244), .ZN(n237) );
  NAND2X0 U249 ( .IN1(n276), .IN2(n215), .QN(n244) );
  AO22X1 U250 ( .IN1(n238), .IN2(n239), .IN3(n235), .IN4(n213), .Q(n241) );
  OA21X1 U251 ( .IN1(r_in[3]), .IN2(r_in[2]), .IN3(n276), .Q(n235) );
  INVX0 U252 ( .INP(n282), .ZN(n276) );
  NAND2X0 U253 ( .IN1(n283), .IN2(n282), .QN(N245) );
  NAND3X0 U254 ( .IN1(n330), .IN2(n327), .IN3(n269), .QN(n282) );
  NAND3X0 U255 ( .IN1(n284), .IN2(n268), .IN3(n285), .QN(N244) );
  NAND2X0 U256 ( .IN1(n271), .IN2(n330), .QN(n268) );
  AO221X1 U257 ( .IN1(n279), .IN2(n286), .IN3(n248), .IN4(n220), .IN5(n287), 
        .Q(N243) );
  NAND2X0 U258 ( .IN1(n288), .IN2(n289), .QN(n287) );
  OR3X1 U259 ( .IN1(n212), .IN2(stbi), .IN3(n255), .Q(n288) );
  NAND3X0 U260 ( .IN1(n290), .IN2(n291), .IN3(n292), .QN(N242) );
  OA221X1 U261 ( .IN1(n255), .IN2(n293), .IN3(n239), .IN4(n289), .IN5(n274), 
        .Q(n292) );
  INVX0 U262 ( .INP(n279), .ZN(n239) );
  AO22X1 U263 ( .IN1(n294), .IN2(n295), .IN3(n324), .IN4(n296), .Q(n279) );
  OA222X1 U264 ( .IN1(n297), .IN2(n298), .IN3(n299), .IN4(n221), .IN5(n299), 
        .IN6(n218), .Q(n295) );
  NAND2X0 U265 ( .IN1(n300), .IN2(n301), .QN(n298) );
  AO222X1 U266 ( .IN1(n302), .IN2(cont[4]), .IN3(n318), .IN4(n296), .IN5(n303), 
        .IN6(r_in[4]), .Q(n301) );
  AO222X1 U267 ( .IN1(n302), .IN2(cont[3]), .IN3(n319), .IN4(n296), .IN5(n303), 
        .IN6(r_in[3]), .Q(n300) );
  OA22X1 U268 ( .IN1(n304), .IN2(n305), .IN3(n306), .IN4(n307), .Q(n297) );
  OA221X1 U269 ( .IN1(n308), .IN2(n309), .IN3(n215), .IN4(n310), .IN5(n311), 
        .Q(n307) );
  OA22X1 U270 ( .IN1(n299), .IN2(n222), .IN3(n170), .IN4(n283), .Q(n311) );
  NAND2X0 U271 ( .IN1(n285), .IN2(n310), .QN(n305) );
  AND2X1 U272 ( .IN1(n309), .IN2(n308), .Q(n304) );
  AOI222X1 U273 ( .IN1(n302), .IN2(cont[1]), .IN3(n321), .IN4(n296), .IN5(n303), .IN6(r_in[1]), .QN(n308) );
  OA222X1 U274 ( .IN1(n283), .IN2(n176), .IN3(n224), .IN4(n299), .IN5(n310), 
        .IN6(n217), .Q(n309) );
  OA22X1 U275 ( .IN1(n238), .IN2(n266), .IN3(n312), .IN4(n306), .Q(n294) );
  INVX0 U276 ( .INP(n289), .ZN(n306) );
  NAND3X0 U277 ( .IN1(n327), .IN2(n212), .IN3(n269), .QN(n289) );
  AOI222X1 U278 ( .IN1(r_in[5]), .IN2(n303), .IN3(n296), .IN4(n323), .IN5(n216), .IN6(n302), .QN(n312) );
  INVX0 U279 ( .INP(n283), .ZN(n302) );
  NAND2X0 U280 ( .IN1(n266), .IN2(n272), .QN(n283) );
  INVX0 U281 ( .INP(n263), .ZN(n272) );
  INVX0 U282 ( .INP(n299), .ZN(n296) );
  NOR2X0 U283 ( .IN1(n238), .IN2(n275), .QN(n299) );
  NOR2X0 U284 ( .IN1(n219), .IN2(n274), .QN(n275) );
  NAND3X0 U285 ( .IN1(n214), .IN2(n211), .IN3(n212), .QN(n274) );
  INVX0 U286 ( .INP(n310), .ZN(n303) );
  INVX0 U287 ( .INP(n285), .ZN(n238) );
  NAND2X0 U288 ( .IN1(n269), .IN2(n313), .QN(n285) );
  XOR2X1 U289 ( .IN1(n327), .IN2(n330), .Q(n313) );
  INVX0 U290 ( .INP(stbi), .ZN(n293) );
  NAND3X0 U291 ( .IN1(n214), .IN2(n211), .IN3(n219), .QN(n255) );
  NAND2X0 U292 ( .IN1(n248), .IN2(r_in[1]), .QN(n291) );
  INVX0 U293 ( .INP(n284), .ZN(n248) );
  NAND3X0 U294 ( .IN1(n212), .IN2(n211), .IN3(n269), .QN(n284) );
  INVX0 U295 ( .INP(n286), .ZN(n290) );
  NAND2X0 U296 ( .IN1(n310), .IN2(n278), .QN(n286) );
  NAND3X0 U297 ( .IN1(n330), .IN2(n211), .IN3(n269), .QN(n278) );
  NOR2X0 U298 ( .IN1(n214), .IN2(n329), .QN(n269) );
  NAND2X0 U299 ( .IN1(n263), .IN2(n266), .QN(n310) );
  AND2X1 U300 ( .IN1(n271), .IN2(n212), .Q(n266) );
  NOR3X0 U301 ( .IN1(n329), .IN2(n328), .IN3(n211), .QN(n271) );
  MUX21X1 U302 ( .IN1(n314), .IN2(n315), .S(r_in[5]), .Q(n263) );
  NAND4X0 U303 ( .IN1(r_in[4]), .IN2(r_in[3]), .IN3(n316), .IN4(r_in[2]), .QN(
        n315) );
  NOR2X0 U304 ( .IN1(n217), .IN2(n220), .QN(n316) );
  NAND4X0 U305 ( .IN1(n213), .IN2(n225), .IN3(n215), .IN4(n317), .QN(n314) );
  NOR2X0 U306 ( .IN1(r_in[1]), .IN2(r_in[0]), .QN(n317) );
endmodule

