/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:10:25 2023
/////////////////////////////////////////////////////////////


module b13 ( reset, eoc, soc, load_dato, add_mpx2, canale, mux_en, clock, 
        data_in, dsr, error, data_out );
  output [3:0] canale;
  input [7:0] data_in;
  input reset, eoc, clock, dsr;
  output soc, load_dato, add_mpx2, mux_en, error, data_out;
  wire   send_data, rdy, N89, N90, N91, N115, N116, next_bit_0_, N209, N210,
         N211, N212, N213, N214, N215, N216, N230, N231, n113, n114, n115,
         n116, n117, n118, n119, n120, n121, n122, n123, n124, n125, n126,
         n127, n128, n129, n130, n131, n132, n133, n134, n135, n136, n137,
         n139, n140, n141, n142, n143, n144, n145, n146, n147, n148, n149,
         n150, n151, n152, n153, n154, n155, n156, n158, n161, n162, n163,
         n164, n165, n166, n167, n168, n169, n170, n171, n172, n173, n174,
         n175, n176, n177, n178, n179, n180, n181, n182, n183, n184, n185,
         n186, n187, n188, n189, n190, n191, n192, n193, n194, n195, n196,
         n197, n198, n199, n200, n201, n202, n203, n204, n205, n206, n207,
         n208, n209, n210, n211, n212, n213, n214, n215, n216, n217, n218,
         n219, n220, n221, n222, n223, n224, n225, n226, n227, n228, n229,
         n230, n231, n232, n233, n234, n235, n236, n237, n238, n239, n240,
         n241, n242, n243, n244, n245, n246, n247, n248, n249, n250, n251,
         n252, n253, n254, n255, n256, n257, n258, n259, n260, n261, n262,
         n263, n264, n265, n266, n267, n268, n269, n270, n271, n272, n273,
         n274;
  wire   [7:0] out_reg;
  wire   [9:0] tx_conta;
  wire   [9:2] add_291_carry;
  assign canale[3] = 1'b0;

  DFFARX1 next_bit_reg_0_ ( .D(n158), .CLK(clock), .RSTB(n274), .Q(next_bit_0_), .QN(n163) );
  DFFARX1 next_bit_reg_1_ ( .D(n156), .CLK(clock), .RSTB(n274), .Q(n272), .QN(
        n172) );
  DFFARX1 next_bit_reg_3_ ( .D(n155), .CLK(clock), .RSTB(n274), .Q(n270), .QN(
        n170) );
  DFFARX1 tx_end_reg ( .D(N230), .CLK(clock), .RSTB(n274), .Q(n267), .QN(n177)
         );
  DFFARX1 confirm_reg ( .D(n154), .CLK(clock), .RSTB(n274), .Q(n273), .QN(n175) );
  DFFARX1 mpx_reg ( .D(n153), .CLK(clock), .RSTB(n274), .Q(n265), .QN(n179) );
  DFFARX1 rdy_reg ( .D(n152), .CLK(clock), .RSTB(n274), .Q(rdy), .QN(n178) );
  DFFARX1 S1_reg_1_ ( .D(N90), .CLK(clock), .RSTB(n274), .Q(n269), .QN(n161)
         );
  DFFARX1 S1_reg_2_ ( .D(N91), .CLK(clock), .RSTB(n274), .Q(n262), .QN(n168)
         );
  DFFARX1 S1_reg_0_ ( .D(N89), .CLK(clock), .RSTB(n274), .Q(n268), .QN(n173)
         );
  DFFARX1 send_data_reg ( .D(n151), .CLK(clock), .RSTB(n274), .Q(send_data) );
  DFFARX1 S2_reg_0_ ( .D(N115), .CLK(clock), .RSTB(n274), .Q(n255), .QN(n164)
         );
  DFFARX1 shot_reg ( .D(n150), .CLK(clock), .RSTB(n274), .Q(n264) );
  DFFARX1 itfc_state_reg_0_ ( .D(n149), .CLK(clock), .RSTB(n274), .Q(n263), 
        .QN(n162) );
  DFFARX1 itfc_state_reg_1_ ( .D(n148), .CLK(clock), .RSTB(n274), .Q(n266), 
        .QN(n171) );
  DFFARX1 send_reg ( .D(n147), .CLK(clock), .RSTB(n274), .Q(n257) );
  DFFARX1 load_reg ( .D(n146), .CLK(clock), .RSTB(n274), .Q(n254), .QN(n165)
         );
  DFFARX1 S2_reg_1_ ( .D(N116), .CLK(clock), .RSTB(n274), .Q(n253), .QN(n169)
         );
  DFFARX1 mux_en_reg ( .D(n145), .CLK(clock), .RSTB(n274), .Q(mux_en) );
  DFFARX1 conta_tmp_reg_0_ ( .D(n144), .CLK(clock), .RSTB(n274), .Q(n260) );
  DFFARX1 canale_reg_0_ ( .D(n143), .CLK(clock), .RSTB(n274), .Q(canale[0]) );
  DFFARX1 conta_tmp_reg_1_ ( .D(n142), .CLK(clock), .RSTB(n274), .Q(n261), 
        .QN(n174) );
  DFFARX1 canale_reg_1_ ( .D(n141), .CLK(clock), .RSTB(n274), .Q(canale[1]) );
  DFFARX1 conta_tmp_reg_2_ ( .D(n140), .CLK(clock), .RSTB(n274), .Q(n259), 
        .QN(n176) );
  DFFARX1 canale_reg_2_ ( .D(n139), .CLK(clock), .RSTB(n274), .Q(canale[2]), 
        .QN(n252) );
  DFFARX1 load_dato_reg ( .D(n137), .CLK(clock), .RSTB(n274), .Q(load_dato) );
  DFFARX1 soc_reg ( .D(n136), .CLK(clock), .RSTB(n274), .Q(soc) );
  DFFARX1 add_mpx2_reg ( .D(n135), .CLK(clock), .RSTB(n274), .Q(add_mpx2) );
  DFFARX1 tre_reg ( .D(n134), .CLK(clock), .RSTB(n274), .Q(n258) );
  DFFARX1 error_reg ( .D(n133), .CLK(clock), .RSTB(n274), .Q(error) );
  DFFARX1 out_reg_reg_0_ ( .D(n132), .CLK(clock), .RSTB(n274), .Q(out_reg[0]), 
        .QN(n180) );
  DFFARX1 out_reg_reg_1_ ( .D(n131), .CLK(clock), .RSTB(n274), .Q(out_reg[1])
         );
  DFFARX1 out_reg_reg_2_ ( .D(n130), .CLK(clock), .RSTB(n274), .Q(out_reg[2])
         );
  DFFARX1 out_reg_reg_3_ ( .D(n129), .CLK(clock), .RSTB(n274), .Q(out_reg[3])
         );
  DFFARX1 out_reg_reg_4_ ( .D(n128), .CLK(clock), .RSTB(n274), .Q(out_reg[4]), 
        .QN(n166) );
  DFFARX1 out_reg_reg_5_ ( .D(n127), .CLK(clock), .RSTB(n274), .Q(out_reg[5])
         );
  DFFARX1 out_reg_reg_6_ ( .D(n126), .CLK(clock), .RSTB(n274), .Q(out_reg[6]), 
        .QN(n181) );
  DFFARX1 out_reg_reg_7_ ( .D(n125), .CLK(clock), .RSTB(n274), .Q(out_reg[7])
         );
  DFFARX1 send_en_reg ( .D(n124), .CLK(clock), .RSTB(n274), .Q(n256), .QN(n167) );
  DFFARX1 tx_conta_reg_1_ ( .D(n123), .CLK(clock), .RSTB(n274), .Q(tx_conta[1]) );
  DFFARX1 tx_conta_reg_9_ ( .D(n122), .CLK(clock), .RSTB(n274), .Q(tx_conta[9]) );
  DFFARX1 tx_conta_reg_0_ ( .D(n121), .CLK(clock), .RSTB(n274), .Q(tx_conta[0]) );
  DFFARX1 tx_conta_reg_2_ ( .D(n120), .CLK(clock), .RSTB(n274), .Q(tx_conta[2]) );
  DFFARX1 tx_conta_reg_3_ ( .D(n119), .CLK(clock), .RSTB(n274), .Q(tx_conta[3]) );
  DFFARX1 tx_conta_reg_4_ ( .D(n118), .CLK(clock), .RSTB(n274), .Q(tx_conta[4]) );
  DFFARX1 tx_conta_reg_5_ ( .D(n117), .CLK(clock), .RSTB(n274), .Q(tx_conta[5]) );
  DFFARX1 tx_conta_reg_6_ ( .D(n116), .CLK(clock), .RSTB(n274), .Q(tx_conta[6]) );
  DFFARX1 tx_conta_reg_7_ ( .D(n115), .CLK(clock), .RSTB(n274), .Q(tx_conta[7]) );
  DFFARX1 tx_conta_reg_8_ ( .D(n114), .CLK(clock), .RSTB(n274), .Q(tx_conta[8]) );
  DFFARX1 next_bit_reg_2_ ( .D(n113), .CLK(clock), .RSTB(n274), .Q(n271) );
  DFFARX1 data_out_reg ( .D(N231), .CLK(clock), .RSTB(n274), .Q(data_out) );
  HADDX1 add_291_U1_1_1 ( .A0(tx_conta[1]), .B0(tx_conta[0]), .C1(
        add_291_carry[2]), .SO(N209) );
  HADDX1 add_291_U1_1_2 ( .A0(tx_conta[2]), .B0(add_291_carry[2]), .C1(
        add_291_carry[3]), .SO(N210) );
  HADDX1 add_291_U1_1_3 ( .A0(tx_conta[3]), .B0(add_291_carry[3]), .C1(
        add_291_carry[4]), .SO(N211) );
  HADDX1 add_291_U1_1_4 ( .A0(tx_conta[4]), .B0(add_291_carry[4]), .C1(
        add_291_carry[5]), .SO(N212) );
  HADDX1 add_291_U1_1_5 ( .A0(tx_conta[5]), .B0(add_291_carry[5]), .C1(
        add_291_carry[6]), .SO(N213) );
  HADDX1 add_291_U1_1_6 ( .A0(tx_conta[6]), .B0(add_291_carry[6]), .C1(
        add_291_carry[7]), .SO(N214) );
  HADDX1 add_291_U1_1_7 ( .A0(tx_conta[7]), .B0(add_291_carry[7]), .C1(
        add_291_carry[8]), .SO(N215) );
  HADDX1 add_291_U1_1_8 ( .A0(tx_conta[8]), .B0(add_291_carry[8]), .C1(
        add_291_carry[9]), .SO(N216) );
  INVX2 U144 ( .INP(reset), .ZN(n274) );
  AO21X1 U146 ( .IN1(next_bit_0_), .IN2(n182), .IN3(n183), .Q(n158) );
  MUX21X1 U147 ( .IN1(n184), .IN2(n185), .S(n186), .Q(n183) );
  AND2X1 U148 ( .IN1(n163), .IN2(n187), .Q(n185) );
  NOR2X0 U149 ( .IN1(n182), .IN2(n170), .QN(n184) );
  AO222X1 U150 ( .IN1(n188), .IN2(n189), .IN3(n190), .IN4(n187), .IN5(n272), 
        .IN6(n191), .Q(n156) );
  NOR2X0 U151 ( .IN1(next_bit_0_), .IN2(n271), .QN(n190) );
  AO221X1 U152 ( .IN1(n192), .IN2(n188), .IN3(n270), .IN4(n182), .IN5(n193), 
        .Q(n155) );
  INVX0 U153 ( .INP(n194), .ZN(n188) );
  AO22X1 U154 ( .IN1(n273), .IN2(n266), .IN3(n263), .IN4(n195), .Q(n154) );
  AO21X1 U155 ( .IN1(n266), .IN2(n267), .IN3(n273), .Q(n195) );
  XNOR2X1 U156 ( .IN1(n265), .IN2(n196), .Q(n153) );
  NAND2X0 U157 ( .IN1(n197), .IN2(n198), .QN(n152) );
  AO21X1 U158 ( .IN1(n199), .IN2(n265), .IN3(n178), .Q(n198) );
  NAND2X0 U159 ( .IN1(n200), .IN2(n201), .QN(n151) );
  NAND3X0 U160 ( .IN1(n268), .IN2(n269), .IN3(n262), .QN(n201) );
  OAI21X1 U161 ( .IN1(n202), .IN2(n173), .IN3(send_data), .QN(n200) );
  AO22X1 U162 ( .IN1(n255), .IN2(n169), .IN3(n264), .IN4(n203), .Q(n150) );
  NAND3X0 U163 ( .IN1(n175), .IN2(n164), .IN3(n253), .QN(n203) );
  AO22X1 U164 ( .IN1(n266), .IN2(n177), .IN3(n204), .IN4(n162), .Q(n149) );
  OR2X1 U165 ( .IN1(n264), .IN2(n266), .Q(n204) );
  MUX21X1 U166 ( .IN1(n205), .IN2(n266), .S(n162), .Q(n148) );
  NAND2X0 U167 ( .IN1(n266), .IN2(n267), .QN(n205) );
  AO22X1 U168 ( .IN1(n263), .IN2(n171), .IN3(n257), .IN4(n206), .Q(n147) );
  NAND2X0 U169 ( .IN1(n266), .IN2(n162), .QN(n206) );
  NAND2X0 U170 ( .IN1(n207), .IN2(n208), .QN(n146) );
  NAND3X0 U171 ( .IN1(n171), .IN2(n162), .IN3(n264), .QN(n208) );
  AO21X1 U172 ( .IN1(n263), .IN2(n171), .IN3(n165), .Q(n207) );
  AO22X1 U173 ( .IN1(n209), .IN2(mux_en), .IN3(n210), .IN4(n161), .Q(n145) );
  NOR2X0 U174 ( .IN1(n262), .IN2(n268), .QN(n210) );
  AO21X1 U175 ( .IN1(n260), .IN2(n211), .IN3(n212), .Q(n144) );
  AO21X1 U176 ( .IN1(n211), .IN2(canale[0]), .IN3(n212), .Q(n143) );
  NAND2X0 U177 ( .IN1(n213), .IN2(n214), .QN(n142) );
  AO21X1 U178 ( .IN1(n215), .IN2(n260), .IN3(n174), .Q(n214) );
  AO221X1 U179 ( .IN1(n212), .IN2(n261), .IN3(n211), .IN4(canale[1]), .IN5(
        n216), .Q(n141) );
  INVX0 U180 ( .INP(n213), .ZN(n216) );
  NAND3X0 U181 ( .IN1(n260), .IN2(n174), .IN3(n215), .QN(n213) );
  NOR2X0 U182 ( .IN1(n211), .IN2(n260), .QN(n212) );
  NAND2X0 U183 ( .IN1(n217), .IN2(n218), .QN(n140) );
  AO21X1 U184 ( .IN1(n219), .IN2(n215), .IN3(n176), .Q(n218) );
  NAND2X0 U185 ( .IN1(n220), .IN2(n217), .QN(n139) );
  NAND3X0 U186 ( .IN1(n215), .IN2(n176), .IN3(n219), .QN(n217) );
  INVX0 U187 ( .INP(n221), .ZN(n219) );
  MUX21X1 U188 ( .IN1(n222), .IN2(n252), .S(n211), .Q(n220) );
  NAND2X0 U189 ( .IN1(n259), .IN2(n221), .QN(n222) );
  NAND2X0 U190 ( .IN1(n261), .IN2(n260), .QN(n221) );
  INVX0 U191 ( .INP(n211), .ZN(n215) );
  AO21X1 U192 ( .IN1(n211), .IN2(load_dato), .IN3(n223), .Q(n137) );
  INVX0 U193 ( .INP(n209), .ZN(n223) );
  NAND4X0 U194 ( .IN1(n262), .IN2(n268), .IN3(n224), .IN4(n161), .QN(n209) );
  NAND2X0 U195 ( .IN1(n225), .IN2(n262), .QN(n211) );
  MUX21X1 U196 ( .IN1(soc), .IN2(n168), .S(n225), .Q(n136) );
  AO21X1 U197 ( .IN1(n199), .IN2(n179), .IN3(add_mpx2), .Q(n135) );
  INVX0 U198 ( .INP(n196), .ZN(n199) );
  OR3X1 U199 ( .IN1(n258), .IN2(n254), .IN3(n267), .Q(n134) );
  MUX21X1 U200 ( .IN1(n226), .IN2(n227), .S(n257), .Q(n133) );
  NAND2X0 U201 ( .IN1(dsr), .IN2(n258), .QN(n227) );
  MUX21X1 U202 ( .IN1(n258), .IN2(error), .S(n165), .Q(n226) );
  MUX21X1 U203 ( .IN1(out_reg[0]), .IN2(data_in[0]), .S(n228), .Q(n132) );
  MUX21X1 U204 ( .IN1(out_reg[1]), .IN2(data_in[1]), .S(n228), .Q(n131) );
  MUX21X1 U205 ( .IN1(out_reg[2]), .IN2(data_in[2]), .S(n228), .Q(n130) );
  MUX21X1 U206 ( .IN1(out_reg[3]), .IN2(data_in[3]), .S(n228), .Q(n129) );
  MUX21X1 U207 ( .IN1(out_reg[4]), .IN2(data_in[4]), .S(n228), .Q(n128) );
  MUX21X1 U208 ( .IN1(out_reg[5]), .IN2(data_in[5]), .S(n228), .Q(n127) );
  MUX21X1 U209 ( .IN1(out_reg[6]), .IN2(data_in[6]), .S(n228), .Q(n126) );
  MUX21X1 U210 ( .IN1(out_reg[7]), .IN2(data_in[7]), .S(n228), .Q(n125) );
  NOR2X0 U211 ( .IN1(n165), .IN2(n258), .QN(n228) );
  AO22X1 U212 ( .IN1(n256), .IN2(n177), .IN3(n229), .IN4(dsr), .Q(n124) );
  AND2X1 U213 ( .IN1(n257), .IN2(n258), .Q(n229) );
  AO22X1 U214 ( .IN1(tx_conta[1]), .IN2(n167), .IN3(N209), .IN4(n230), .Q(n123) );
  AO22X1 U215 ( .IN1(tx_conta[9]), .IN2(n167), .IN3(add_291_carry[9]), .IN4(
        n230), .Q(n122) );
  MUX21X1 U216 ( .IN1(n230), .IN2(n167), .S(tx_conta[0]), .Q(n121) );
  AO22X1 U217 ( .IN1(tx_conta[2]), .IN2(n167), .IN3(N210), .IN4(n230), .Q(n120) );
  AO22X1 U218 ( .IN1(tx_conta[3]), .IN2(n167), .IN3(N211), .IN4(n230), .Q(n119) );
  AO22X1 U219 ( .IN1(tx_conta[4]), .IN2(n167), .IN3(N212), .IN4(n230), .Q(n118) );
  AO22X1 U220 ( .IN1(tx_conta[5]), .IN2(n167), .IN3(N213), .IN4(n230), .Q(n117) );
  AO22X1 U221 ( .IN1(tx_conta[6]), .IN2(n167), .IN3(N214), .IN4(n230), .Q(n116) );
  AO22X1 U222 ( .IN1(tx_conta[7]), .IN2(n167), .IN3(N215), .IN4(n230), .Q(n115) );
  AO22X1 U223 ( .IN1(tx_conta[8]), .IN2(n167), .IN3(N216), .IN4(n230), .Q(n114) );
  AND2X1 U224 ( .IN1(n231), .IN2(n256), .Q(n230) );
  AO21X1 U225 ( .IN1(n189), .IN2(n170), .IN3(n232), .Q(n113) );
  MUX21X1 U226 ( .IN1(n233), .IN2(n191), .S(n271), .Q(n232) );
  AO21X1 U227 ( .IN1(n170), .IN2(n163), .IN3(n182), .Q(n191) );
  NOR2X0 U228 ( .IN1(n194), .IN2(n172), .QN(n233) );
  AO21X1 U229 ( .IN1(n262), .IN2(n268), .IN3(n225), .Q(N91) );
  NOR2X0 U230 ( .IN1(n161), .IN2(n268), .QN(n225) );
  MUX21X1 U231 ( .IN1(n234), .IN2(n262), .S(n173), .Q(N90) );
  AO22X1 U232 ( .IN1(n224), .IN2(n161), .IN3(n202), .IN4(n168), .Q(n234) );
  NAND2X0 U233 ( .IN1(n269), .IN2(rdy), .QN(n202) );
  INVX0 U234 ( .INP(eoc), .ZN(n224) );
  NAND2X0 U235 ( .IN1(n235), .IN2(n268), .QN(N89) );
  MUX21X1 U236 ( .IN1(n236), .IN2(n237), .S(n168), .Q(n235) );
  NAND2X0 U237 ( .IN1(n269), .IN2(n178), .QN(n237) );
  NAND2X0 U238 ( .IN1(eoc), .IN2(n161), .QN(n236) );
  NAND3X0 U239 ( .IN1(n238), .IN2(n239), .IN3(n240), .QN(N231) );
  MUX21X1 U240 ( .IN1(n241), .IN2(n242), .S(n163), .Q(n240) );
  NAND2X0 U241 ( .IN1(n243), .IN2(n170), .QN(n242) );
  AO222X1 U242 ( .IN1(out_reg[5]), .IN2(n189), .IN3(n244), .IN4(out_reg[7]), 
        .IN5(out_reg[3]), .IN6(n192), .Q(n243) );
  NOR2X0 U243 ( .IN1(n271), .IN2(n172), .QN(n244) );
  AND2X1 U244 ( .IN1(n271), .IN2(n172), .Q(n189) );
  OA22X1 U245 ( .IN1(n270), .IN2(n245), .IN3(n186), .IN4(n180), .Q(n241) );
  OA221X1 U246 ( .IN1(n271), .IN2(n181), .IN3(n272), .IN4(n166), .IN5(n246), 
        .Q(n245) );
  AOI21X1 U247 ( .IN1(n192), .IN2(out_reg[2]), .IN3(n247), .QN(n246) );
  AND2X1 U248 ( .IN1(n272), .IN2(n271), .Q(n192) );
  NAND2X0 U249 ( .IN1(n193), .IN2(out_reg[1]), .QN(n239) );
  AND3X1 U250 ( .IN1(n247), .IN2(n163), .IN3(n270), .Q(n193) );
  NOR2X0 U251 ( .IN1(n186), .IN2(n194), .QN(N230) );
  NAND2X0 U252 ( .IN1(next_bit_0_), .IN2(n187), .QN(n194) );
  NOR2X0 U253 ( .IN1(n182), .IN2(n270), .QN(n187) );
  INVX0 U254 ( .INP(n238), .ZN(n182) );
  NOR2X0 U255 ( .IN1(n167), .IN2(n231), .QN(n238) );
  NOR4X0 U256 ( .IN1(n248), .IN2(tx_conta[7]), .IN3(tx_conta[8]), .IN4(
        tx_conta[9]), .QN(n231) );
  AND3X1 U257 ( .IN1(tx_conta[5]), .IN2(n249), .IN3(tx_conta[6]), .Q(n248) );
  AO21X1 U258 ( .IN1(tx_conta[3]), .IN2(n250), .IN3(tx_conta[4]), .Q(n249) );
  OR3X1 U259 ( .IN1(tx_conta[1]), .IN2(tx_conta[2]), .IN3(tx_conta[0]), .Q(
        n250) );
  INVX0 U260 ( .INP(n247), .ZN(n186) );
  NOR2X0 U261 ( .IN1(n271), .IN2(n272), .QN(n247) );
  MUX21X1 U262 ( .IN1(n251), .IN2(n255), .S(n169), .Q(N116) );
  OA21X1 U263 ( .IN1(n265), .IN2(n175), .IN3(n164), .Q(n251) );
  NAND2X0 U264 ( .IN1(n196), .IN2(n197), .QN(N115) );
  NAND3X0 U265 ( .IN1(n164), .IN2(n169), .IN3(send_data), .QN(n197) );
  NAND3X0 U266 ( .IN1(n273), .IN2(n164), .IN3(n253), .QN(n196) );
endmodule

