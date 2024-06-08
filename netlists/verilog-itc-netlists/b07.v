/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:09:13 2023
/////////////////////////////////////////////////////////////


module b07 ( punti_retta, start, reset, clock );
  output [7:0] punti_retta;
  input start, reset, clock;
  wire   N136, N137, N138, N139, N140, N141, N142, N143, N159, N160, N161,
         N162, N163, N164, N165, N166, N171, N172, N173, U3_U1_Z_0, U3_U1_Z_1,
         U3_U1_Z_2, U3_U1_Z_3, U3_U1_Z_4, U3_U1_Z_5, U3_U1_Z_6, U3_U1_Z_7,
         U3_U2_Z_0, U3_U2_Z_1, U3_U2_Z_2, U3_U2_Z_3, U3_U2_Z_4, U3_U2_Z_5,
         U3_U2_Z_6, U3_U2_Z_7, n109, n110, n111, n112, n113, n114, n115, n116,
         n117, n118, n119, n120, n121, n122, n123, n124, n125, n126, n127,
         n128, n129, n130, n131, n132, n133, n134, n135, n136, n137, n138,
         n139, n140, n141, n142, n143, n144, n145, n146, n147, n148, n149,
         n151, add_79_aco_B_0_, n154, n155, n156, n157, n158, n159, n160, n161,
         n162, n163, n164, n165, n166, n167, n168, n169, n170, n171, n172,
         n173, n174, n175, n176, n177, n178, n179, n180, n181, n182, n183,
         n184, n185, n186, n187, n188, n189, n190, n191, n192, n193, n194,
         n195, n196, n197, n198, n199, n200, n201, n202, n203, n204, n205,
         n206, n207, n208, n209, n210, n211, n212, n213, n214, n215, n216,
         n217, n218, n219, n220, n221, n222, n223, n224, n225, n226, n227,
         n228, n229, n230, n231, n232, n233, n234, n235, n236;
  wire   [2:0] stato;
  wire   [7:0] cont;
  wire   [6:0] x;
  wire   [6:0] y;
  wire   [6:0] t;
  wire   [7:2] r143_carry;

  DFFARX1 mar_reg_0_ ( .D(n151), .CLK(clock), .RSTB(n236), .Q(n234), .QN(n157)
         );
  DFFARX1 mar_reg_1_ ( .D(n149), .CLK(clock), .RSTB(n236), .Q(n232), .QN(n161)
         );
  DFFARX1 stato_reg_1_ ( .D(N172), .CLK(clock), .RSTB(n236), .Q(n231), .QN(
        n155) );
  DFFARX1 stato_reg_0_ ( .D(N171), .CLK(clock), .RSTB(n236), .Q(stato[0]), 
        .QN(n158) );
  DFFARX1 stato_reg_2_ ( .D(N173), .CLK(clock), .RSTB(n236), .Q(stato[2]), 
        .QN(n154) );
  DFFARX1 mar_reg_2_ ( .D(n148), .CLK(clock), .RSTB(n236), .Q(n233), .QN(n160)
         );
  DFFARX1 mar_reg_3_ ( .D(n147), .CLK(clock), .RSTB(n236), .Q(n235), .QN(n156)
         );
  DFFARX1 y_reg_6_ ( .D(n146), .CLK(clock), .RSTB(n236), .Q(y[6]) );
  DFFARX1 y_reg_5_ ( .D(n145), .CLK(clock), .RSTB(n236), .Q(y[5]) );
  DFFARX1 y_reg_4_ ( .D(n144), .CLK(clock), .RSTB(n236), .Q(y[4]) );
  DFFARX1 y_reg_3_ ( .D(n143), .CLK(clock), .RSTB(n236), .Q(y[3]) );
  DFFARX1 y_reg_2_ ( .D(n142), .CLK(clock), .RSTB(n236), .Q(y[2]) );
  DFFARX1 y_reg_1_ ( .D(n141), .CLK(clock), .RSTB(n236), .Q(y[1]) );
  DFFARX1 y_reg_0_ ( .D(n140), .CLK(clock), .RSTB(n236), .Q(y[0]) );
  DFFARX1 t_reg_0_ ( .D(n139), .CLK(clock), .RSTB(n236), .Q(t[0]) );
  DFFARX1 x_reg_0_ ( .D(n138), .CLK(clock), .RSTB(n236), .Q(x[0]) );
  DFFARX1 t_reg_1_ ( .D(n137), .CLK(clock), .RSTB(n236), .Q(t[1]) );
  DFFARX1 x_reg_1_ ( .D(n136), .CLK(clock), .RSTB(n236), .Q(x[1]) );
  DFFARX1 t_reg_2_ ( .D(n135), .CLK(clock), .RSTB(n236), .Q(t[2]) );
  DFFARX1 x_reg_2_ ( .D(n134), .CLK(clock), .RSTB(n236), .Q(x[2]) );
  DFFARX1 t_reg_3_ ( .D(n133), .CLK(clock), .RSTB(n236), .Q(t[3]) );
  DFFARX1 x_reg_3_ ( .D(n132), .CLK(clock), .RSTB(n236), .Q(x[3]) );
  DFFARX1 t_reg_4_ ( .D(n131), .CLK(clock), .RSTB(n236), .Q(t[4]) );
  DFFARX1 x_reg_4_ ( .D(n130), .CLK(clock), .RSTB(n236), .Q(x[4]) );
  DFFARX1 t_reg_5_ ( .D(n129), .CLK(clock), .RSTB(n236), .Q(t[5]) );
  DFFARX1 x_reg_5_ ( .D(n128), .CLK(clock), .RSTB(n236), .Q(x[5]) );
  DFFARX1 t_reg_6_ ( .D(n127), .CLK(clock), .RSTB(n236), .Q(t[6]) );
  DFFARX1 x_reg_6_ ( .D(n126), .CLK(clock), .RSTB(n236), .Q(x[6]) );
  DFFARX1 x_reg_7_ ( .D(n125), .CLK(clock), .RSTB(n236), .Q(n230), .QN(n162)
         );
  DFFARX1 cont_reg_0_ ( .D(n124), .CLK(clock), .RSTB(n236), .Q(cont[0]) );
  DFFARX1 cont_reg_7_ ( .D(n123), .CLK(clock), .RSTB(n236), .Q(cont[7]) );
  DFFARX1 cont_reg_6_ ( .D(n122), .CLK(clock), .RSTB(n236), .Q(cont[6]) );
  DFFARX1 cont_reg_5_ ( .D(n121), .CLK(clock), .RSTB(n236), .Q(cont[5]) );
  DFFARX1 cont_reg_4_ ( .D(n120), .CLK(clock), .RSTB(n236), .Q(cont[4]) );
  DFFARX1 cont_reg_3_ ( .D(n119), .CLK(clock), .RSTB(n236), .Q(cont[3]) );
  DFFARX1 cont_reg_2_ ( .D(n118), .CLK(clock), .RSTB(n236), .Q(cont[2]) );
  DFFARX1 cont_reg_1_ ( .D(n117), .CLK(clock), .RSTB(n236), .Q(cont[1]) );
  DFFARX1 punti_retta_reg_0_ ( .D(n116), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[0]) );
  DFFARX1 punti_retta_reg_1_ ( .D(n115), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[1]) );
  DFFARX1 punti_retta_reg_2_ ( .D(n114), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[2]) );
  DFFARX1 punti_retta_reg_3_ ( .D(n113), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[3]) );
  DFFARX1 punti_retta_reg_4_ ( .D(n112), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[4]) );
  DFFARX1 punti_retta_reg_5_ ( .D(n111), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[5]) );
  DFFARX1 punti_retta_reg_6_ ( .D(n110), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[6]) );
  DFFARX1 punti_retta_reg_7_ ( .D(n109), .CLK(clock), .RSTB(n236), .Q(
        punti_retta[7]) );
  FADDX1 r143_U1_1 ( .A(U3_U1_Z_1), .B(U3_U2_Z_1), .CI(n159), .CO(
        r143_carry[2]), .S(N137) );
  FADDX1 r143_U1_2 ( .A(U3_U1_Z_2), .B(U3_U2_Z_2), .CI(r143_carry[2]), .CO(
        r143_carry[3]), .S(N138) );
  FADDX1 r143_U1_3 ( .A(U3_U1_Z_3), .B(U3_U2_Z_3), .CI(r143_carry[3]), .CO(
        r143_carry[4]), .S(N139) );
  FADDX1 r143_U1_4 ( .A(U3_U1_Z_4), .B(U3_U2_Z_4), .CI(r143_carry[4]), .CO(
        r143_carry[5]), .S(N140) );
  FADDX1 r143_U1_5 ( .A(U3_U1_Z_5), .B(U3_U2_Z_5), .CI(r143_carry[5]), .CO(
        r143_carry[6]), .S(N141) );
  FADDX1 r143_U1_6 ( .A(U3_U1_Z_6), .B(U3_U2_Z_6), .CI(r143_carry[6]), .CO(
        r143_carry[7]), .S(N142) );
  FADDX1 r143_U1_7 ( .A(U3_U1_Z_7), .B(U3_U2_Z_7), .CI(r143_carry[7]), .S(N143) );
  AND2X1 U141 ( .IN1(U3_U2_Z_0), .IN2(U3_U1_Z_0), .Q(n159) );
  INVX2 U142 ( .INP(reset), .ZN(n236) );
  AND2X1 U143 ( .IN1(add_79_aco_B_0_), .IN2(cont[0]), .Q(n163) );
  AND2X1 U144 ( .IN1(n163), .IN2(cont[1]), .Q(n164) );
  AND2X1 U145 ( .IN1(n164), .IN2(cont[2]), .Q(n165) );
  AND2X1 U146 ( .IN1(n165), .IN2(cont[3]), .Q(n166) );
  AND2X1 U147 ( .IN1(n166), .IN2(cont[4]), .Q(n167) );
  AND2X1 U148 ( .IN1(n167), .IN2(cont[5]), .Q(n168) );
  XNOR2X1 U149 ( .IN1(n169), .IN2(cont[7]), .Q(N166) );
  NAND2X1 U150 ( .IN1(n168), .IN2(cont[6]), .QN(n169) );
  XOR2X1 U151 ( .IN1(U3_U2_Z_0), .IN2(U3_U1_Z_0), .Q(N136) );
  XOR2X1 U152 ( .IN1(add_79_aco_B_0_), .IN2(cont[0]), .Q(N159) );
  XOR2X1 U153 ( .IN1(n163), .IN2(cont[1]), .Q(N160) );
  XOR2X1 U154 ( .IN1(n164), .IN2(cont[2]), .Q(N161) );
  XOR2X1 U155 ( .IN1(n165), .IN2(cont[3]), .Q(N162) );
  XOR2X1 U156 ( .IN1(n166), .IN2(cont[4]), .Q(N163) );
  XOR2X1 U157 ( .IN1(n167), .IN2(cont[5]), .Q(N164) );
  XOR2X1 U158 ( .IN1(n168), .IN2(cont[6]), .Q(N165) );
  MUX21X1 U159 ( .IN1(n170), .IN2(n171), .S(n157), .Q(n151) );
  AO22X1 U160 ( .IN1(n172), .IN2(n171), .IN3(n232), .IN4(n173), .Q(n149) );
  AO21X1 U161 ( .IN1(n171), .IN2(n157), .IN3(n170), .Q(n173) );
  NOR2X0 U162 ( .IN1(n174), .IN2(n175), .QN(n148) );
  OA21X1 U163 ( .IN1(n176), .IN2(n160), .IN3(n177), .Q(n175) );
  NAND4X0 U164 ( .IN1(n232), .IN2(n234), .IN3(n178), .IN4(n179), .QN(n177) );
  NOR2X0 U165 ( .IN1(n174), .IN2(n180), .QN(n147) );
  OA22X1 U166 ( .IN1(n181), .IN2(n156), .IN3(n182), .IN4(n183), .Q(n180) );
  INVX0 U167 ( .INP(n176), .ZN(n183) );
  NOR2X0 U168 ( .IN1(n179), .IN2(n170), .QN(n176) );
  NOR2X0 U169 ( .IN1(n170), .IN2(n184), .QN(n181) );
  NOR2X0 U170 ( .IN1(n171), .IN2(n170), .QN(n174) );
  INVX0 U171 ( .INP(n178), .ZN(n170) );
  AO221X1 U172 ( .IN1(n185), .IN2(start), .IN3(n186), .IN4(n184), .IN5(n187), 
        .Q(n178) );
  NOR2X0 U173 ( .IN1(stato[2]), .IN2(n158), .QN(n185) );
  NAND2X0 U174 ( .IN1(n188), .IN2(n189), .QN(n171) );
  AO21X1 U175 ( .IN1(y[6]), .IN2(n190), .IN3(n191), .Q(n146) );
  AO21X1 U176 ( .IN1(y[5]), .IN2(n190), .IN3(n191), .Q(n145) );
  AO21X1 U177 ( .IN1(y[4]), .IN2(n190), .IN3(n191), .Q(n144) );
  AO21X1 U178 ( .IN1(y[3]), .IN2(n190), .IN3(n191), .Q(n143) );
  AND2X1 U179 ( .IN1(n192), .IN2(n193), .Q(n191) );
  AO22X1 U180 ( .IN1(y[2]), .IN2(n190), .IN3(n192), .IN4(n194), .Q(n142) );
  MUX21X1 U181 ( .IN1(n195), .IN2(y[1]), .S(n190), .Q(n141) );
  AO22X1 U182 ( .IN1(y[0]), .IN2(n190), .IN3(n192), .IN4(n196), .Q(n140) );
  NOR2X0 U183 ( .IN1(n190), .IN2(n233), .QN(n192) );
  MUX21X1 U184 ( .IN1(N136), .IN2(t[0]), .S(n189), .Q(n139) );
  AO222X1 U185 ( .IN1(x[0]), .IN2(n197), .IN3(n198), .IN4(n196), .IN5(N136), 
        .IN6(n199), .Q(n138) );
  AO21X1 U186 ( .IN1(n156), .IN2(n161), .IN3(n200), .Q(n196) );
  MUX21X1 U187 ( .IN1(N137), .IN2(t[1]), .S(n189), .Q(n137) );
  AO222X1 U188 ( .IN1(x[1]), .IN2(n197), .IN3(n201), .IN4(n195), .IN5(N137), 
        .IN6(n199), .Q(n136) );
  NAND2X0 U189 ( .IN1(n202), .IN2(n203), .QN(n195) );
  MUX21X1 U190 ( .IN1(n204), .IN2(n205), .S(n160), .Q(n202) );
  NAND2X0 U191 ( .IN1(n200), .IN2(n157), .QN(n205) );
  NAND2X0 U192 ( .IN1(n235), .IN2(n234), .QN(n204) );
  MUX21X1 U193 ( .IN1(N138), .IN2(t[2]), .S(n189), .Q(n135) );
  AO222X1 U194 ( .IN1(x[2]), .IN2(n197), .IN3(n198), .IN4(n194), .IN5(N138), 
        .IN6(n199), .Q(n134) );
  OR2X1 U195 ( .IN1(n206), .IN2(n200), .Q(n194) );
  MUX21X1 U196 ( .IN1(N139), .IN2(t[3]), .S(n189), .Q(n133) );
  AO221X1 U197 ( .IN1(x[3]), .IN2(n197), .IN3(N139), .IN4(n199), .IN5(n207), 
        .Q(n132) );
  MUX21X1 U198 ( .IN1(N140), .IN2(t[4]), .S(n189), .Q(n131) );
  AO221X1 U199 ( .IN1(x[4]), .IN2(n197), .IN3(N140), .IN4(n199), .IN5(n207), 
        .Q(n130) );
  MUX21X1 U200 ( .IN1(N141), .IN2(t[5]), .S(n189), .Q(n129) );
  AO221X1 U201 ( .IN1(x[5]), .IN2(n197), .IN3(N141), .IN4(n199), .IN5(n207), 
        .Q(n128) );
  MUX21X1 U202 ( .IN1(N142), .IN2(t[6]), .S(n189), .Q(n127) );
  AO221X1 U203 ( .IN1(x[6]), .IN2(n197), .IN3(N142), .IN4(n199), .IN5(n207), 
        .Q(n126) );
  AO221X1 U204 ( .IN1(n230), .IN2(n197), .IN3(N143), .IN4(n199), .IN5(n207), 
        .Q(n125) );
  AND2X1 U205 ( .IN1(n198), .IN2(n193), .Q(n207) );
  AO21X1 U206 ( .IN1(n200), .IN2(n157), .IN3(n206), .Q(n193) );
  NOR2X0 U207 ( .IN1(n203), .IN2(n235), .QN(n206) );
  INVX0 U208 ( .INP(n172), .ZN(n203) );
  NOR2X0 U209 ( .IN1(n157), .IN2(n232), .QN(n172) );
  NOR2X0 U210 ( .IN1(n156), .IN2(n161), .QN(n200) );
  NOR2X0 U211 ( .IN1(n208), .IN2(n233), .QN(n198) );
  NOR2X0 U212 ( .IN1(n199), .IN2(n201), .QN(n197) );
  INVX0 U213 ( .INP(n208), .ZN(n201) );
  AO22X1 U214 ( .IN1(n209), .IN2(cont[0]), .IN3(n210), .IN4(N136), .Q(n124) );
  AO22X1 U215 ( .IN1(n209), .IN2(cont[7]), .IN3(n210), .IN4(N143), .Q(n123) );
  AO22X1 U216 ( .IN1(n209), .IN2(cont[6]), .IN3(n210), .IN4(N142), .Q(n122) );
  AO22X1 U217 ( .IN1(n209), .IN2(cont[5]), .IN3(n210), .IN4(N141), .Q(n121) );
  AO22X1 U218 ( .IN1(n209), .IN2(cont[4]), .IN3(n210), .IN4(N140), .Q(n120) );
  AO22X1 U219 ( .IN1(n209), .IN2(cont[3]), .IN3(n210), .IN4(N139), .Q(n119) );
  AO22X1 U220 ( .IN1(n209), .IN2(cont[2]), .IN3(n210), .IN4(N138), .Q(n118) );
  AO22X1 U221 ( .IN1(n209), .IN2(cont[1]), .IN3(n210), .IN4(N137), .Q(n117) );
  NOR2X0 U222 ( .IN1(n188), .IN2(n209), .QN(n210) );
  AND2X1 U223 ( .IN1(n211), .IN2(n212), .Q(n209) );
  AO22X1 U224 ( .IN1(punti_retta[0]), .IN2(n213), .IN3(N159), .IN4(n214), .Q(
        n116) );
  AO22X1 U225 ( .IN1(punti_retta[1]), .IN2(n213), .IN3(N160), .IN4(n214), .Q(
        n115) );
  AO22X1 U226 ( .IN1(punti_retta[2]), .IN2(n213), .IN3(N161), .IN4(n214), .Q(
        n114) );
  AO22X1 U227 ( .IN1(punti_retta[3]), .IN2(n213), .IN3(N162), .IN4(n214), .Q(
        n113) );
  AO22X1 U228 ( .IN1(punti_retta[4]), .IN2(n213), .IN3(N163), .IN4(n214), .Q(
        n112) );
  AO22X1 U229 ( .IN1(punti_retta[5]), .IN2(n213), .IN3(N164), .IN4(n214), .Q(
        n111) );
  AO22X1 U230 ( .IN1(punti_retta[6]), .IN2(n213), .IN3(N165), .IN4(n214), .Q(
        n110) );
  AO22X1 U231 ( .IN1(punti_retta[7]), .IN2(n213), .IN3(N166), .IN4(n214), .Q(
        n109) );
  NOR2X0 U232 ( .IN1(n213), .IN2(n188), .QN(n214) );
  AO221X1 U233 ( .IN1(t[6]), .IN2(n215), .IN3(n216), .IN4(y[6]), .IN5(
        U3_U2_Z_7), .Q(U3_U2_Z_6) );
  AO222X1 U234 ( .IN1(t[5]), .IN2(n215), .IN3(x[5]), .IN4(n187), .IN5(n216), 
        .IN6(y[5]), .Q(U3_U2_Z_5) );
  AO222X1 U235 ( .IN1(t[4]), .IN2(n215), .IN3(x[4]), .IN4(n187), .IN5(n216), 
        .IN6(y[4]), .Q(U3_U2_Z_4) );
  AO222X1 U236 ( .IN1(t[3]), .IN2(n215), .IN3(x[3]), .IN4(n187), .IN5(n216), 
        .IN6(y[3]), .Q(U3_U2_Z_3) );
  AO222X1 U237 ( .IN1(t[2]), .IN2(n215), .IN3(x[2]), .IN4(n187), .IN5(n216), 
        .IN6(y[2]), .Q(U3_U2_Z_2) );
  AO222X1 U238 ( .IN1(t[1]), .IN2(n215), .IN3(x[1]), .IN4(n187), .IN5(n216), 
        .IN6(y[1]), .Q(U3_U2_Z_1) );
  AO221X1 U239 ( .IN1(t[0]), .IN2(n215), .IN3(n216), .IN4(y[0]), .IN5(n217), 
        .Q(U3_U2_Z_0) );
  AO21X1 U240 ( .IN1(x[0]), .IN2(n187), .IN3(n218), .Q(n217) );
  INVX0 U241 ( .INP(n219), .ZN(n216) );
  INVX0 U242 ( .INP(n190), .ZN(n215) );
  AO21X1 U243 ( .IN1(n218), .IN2(cont[7]), .IN3(U3_U2_Z_7), .Q(U3_U1_Z_7) );
  AND2X1 U244 ( .IN1(x[6]), .IN2(n187), .Q(U3_U2_Z_7) );
  AO22X1 U245 ( .IN1(x[6]), .IN2(n220), .IN3(n218), .IN4(cont[6]), .Q(
        U3_U1_Z_6) );
  AO22X1 U246 ( .IN1(x[5]), .IN2(n220), .IN3(n218), .IN4(cont[5]), .Q(
        U3_U1_Z_5) );
  AO22X1 U247 ( .IN1(x[4]), .IN2(n220), .IN3(n218), .IN4(cont[4]), .Q(
        U3_U1_Z_4) );
  AO22X1 U248 ( .IN1(x[3]), .IN2(n220), .IN3(n218), .IN4(cont[3]), .Q(
        U3_U1_Z_3) );
  AO22X1 U249 ( .IN1(x[2]), .IN2(n220), .IN3(n218), .IN4(cont[2]), .Q(
        U3_U1_Z_2) );
  AO22X1 U250 ( .IN1(x[1]), .IN2(n220), .IN3(n218), .IN4(cont[1]), .Q(
        U3_U1_Z_1) );
  AO22X1 U251 ( .IN1(x[0]), .IN2(n220), .IN3(n218), .IN4(cont[0]), .Q(
        U3_U1_Z_0) );
  INVX0 U252 ( .INP(n212), .ZN(n218) );
  NAND3X0 U253 ( .IN1(n186), .IN2(n184), .IN3(add_79_aco_B_0_), .QN(n212) );
  AND4X1 U254 ( .IN1(x[1]), .IN2(n162), .IN3(n221), .IN4(n222), .Q(
        add_79_aco_B_0_) );
  NOR4X0 U255 ( .IN1(x[6]), .IN2(x[5]), .IN3(x[4]), .IN4(x[3]), .QN(n222) );
  NOR2X0 U256 ( .IN1(x[2]), .IN2(x[0]), .QN(n221) );
  INVX0 U257 ( .INP(n182), .ZN(n184) );
  INVX0 U258 ( .INP(n223), .ZN(n220) );
  NAND2X0 U259 ( .IN1(n223), .IN2(n224), .QN(N173) );
  NAND3X0 U260 ( .IN1(n186), .IN2(start), .IN3(n182), .QN(n224) );
  NOR2X0 U261 ( .IN1(n199), .IN2(n187), .QN(n223) );
  INVX0 U262 ( .INP(n189), .ZN(n187) );
  NAND3X0 U263 ( .IN1(n231), .IN2(n154), .IN3(stato[0]), .QN(n189) );
  NAND2X0 U264 ( .IN1(n219), .IN2(n190), .QN(n199) );
  NAND3X0 U265 ( .IN1(n155), .IN2(n158), .IN3(stato[2]), .QN(n190) );
  NAND4X0 U266 ( .IN1(n225), .IN2(n211), .IN3(n219), .IN4(n208), .QN(N172) );
  NAND3X0 U267 ( .IN1(n158), .IN2(n154), .IN3(n231), .QN(n208) );
  NAND3X0 U268 ( .IN1(stato[0]), .IN2(n155), .IN3(stato[2]), .QN(n219) );
  OR2X1 U269 ( .IN1(n226), .IN2(n227), .Q(n211) );
  AO21X1 U270 ( .IN1(n182), .IN2(n227), .IN3(n188), .Q(n225) );
  INVX0 U271 ( .INP(start), .ZN(n227) );
  OAI21X1 U272 ( .IN1(n228), .IN2(stato[0]), .IN3(n213), .QN(N171) );
  AO21X1 U273 ( .IN1(n229), .IN2(n226), .IN3(start), .Q(n213) );
  NAND3X0 U274 ( .IN1(n155), .IN2(n154), .IN3(stato[0]), .QN(n226) );
  NAND2X0 U275 ( .IN1(n182), .IN2(n186), .QN(n229) );
  INVX0 U276 ( .INP(n188), .ZN(n186) );
  NAND2X0 U277 ( .IN1(n228), .IN2(n158), .QN(n188) );
  NOR2X0 U278 ( .IN1(n156), .IN2(n179), .QN(n182) );
  NAND3X0 U279 ( .IN1(n233), .IN2(n234), .IN3(n232), .QN(n179) );
  NOR2X0 U280 ( .IN1(n154), .IN2(n155), .QN(n228) );
endmodule

