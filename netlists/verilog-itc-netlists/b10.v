/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:09:10 2023
/////////////////////////////////////////////////////////////


module b10 ( r_button, g_button, key, start, reset, test, cts, ctr, rts, rtr, 
        clock, v_in, v_out );
  input [3:0] v_in;
  output [3:0] v_out;
  input r_button, g_button, key, start, reset, test, rts, rtr, clock;
  output cts, ctr;
  wire   n102, n103, n104, n105, n106, n107, n108, n109, n110, n111, n112,
         n113, n114, n115, n116, n117, n118, n119, n120, n121, n122, n123,
         n124, n125, n126, n127, n128, n129, n130, n131, n132, n133, n134,
         n135, n136, n137, n138, n139, n140, n141, n142, n143, n144, n145,
         n146, n147, n148, n149, n150, n151, n152, n153, n154, n155, n156,
         n157, n158, n159, n160, n161, n162, n163, n164, n165, n166, n167,
         n168, n169, n170, n171, n172, n173, n174, n175, n176, n177, n178,
         n179, n180, n181, n182, n183, n184, n185, n186, n187, n188, n189,
         n190, n191, n192, n193, n194, n195, n196, n197, n198, n199, n200,
         n201, n202, n203, n204, n205, n206, n207, n208, n209, n210, n211,
         n212, n213, n214;

  DFFARX1 last_r_reg ( .D(n119), .CLK(clock), .RSTB(n118), .Q(n210) );
  DFFARX1 voto2_reg ( .D(n117), .CLK(clock), .RSTB(n118), .Q(n125), .QN(n209)
         );
  DFFARX1 stato_reg_0_ ( .D(n116), .CLK(clock), .RSTB(n118), .Q(n120), .QN(
        n212) );
  DFFARX1 stato_reg_3_ ( .D(n115), .CLK(clock), .RSTB(n118), .Q(n123), .QN(
        n211) );
  DFFARX1 stato_reg_1_ ( .D(n114), .CLK(clock), .RSTB(n118), .Q(n121), .QN(
        n214) );
  DFFARX1 stato_reg_2_ ( .D(n113), .CLK(clock), .RSTB(n118), .Q(n124), .QN(
        n213) );
  DFFARX1 sign_reg_3_ ( .D(n112), .CLK(clock), .RSTB(n118), .QN(n205) );
  DFFARX1 last_g_reg ( .D(n111), .CLK(clock), .RSTB(n118), .Q(n204) );
  DFFARX1 voto1_reg ( .D(n110), .CLK(clock), .RSTB(n118), .Q(n122), .QN(n206)
         );
  DFFARX1 voto0_reg ( .D(n109), .CLK(clock), .RSTB(n118), .Q(n126), .QN(n208)
         );
  DFFARX1 voto3_reg ( .D(n108), .CLK(clock), .RSTB(n118), .Q(n127), .QN(n207)
         );
  DFFARX1 v_out_reg_3_ ( .D(n107), .CLK(clock), .RSTB(n118), .Q(v_out[3]) );
  DFFARX1 v_out_reg_2_ ( .D(n106), .CLK(clock), .RSTB(n118), .Q(v_out[2]) );
  DFFARX1 v_out_reg_1_ ( .D(n105), .CLK(clock), .RSTB(n118), .Q(v_out[1]) );
  DFFARX1 v_out_reg_0_ ( .D(n104), .CLK(clock), .RSTB(n118), .Q(v_out[0]) );
  DFFARX1 cts_reg ( .D(n103), .CLK(clock), .RSTB(n118), .Q(cts) );
  DFFARX1 ctr_reg ( .D(n102), .CLK(clock), .RSTB(n118), .Q(ctr) );
  INVX2 U111 ( .INP(reset), .ZN(n118) );
  MUX21X1 U112 ( .IN1(n210), .IN2(r_button), .S(n128), .Q(n119) );
  MUX21X1 U113 ( .IN1(n125), .IN2(n129), .S(n130), .Q(n117) );
  OA21X1 U114 ( .IN1(n131), .IN2(n132), .IN3(n133), .Q(n130) );
  NOR2X0 U115 ( .IN1(n210), .IN2(n134), .QN(n131) );
  INVX0 U116 ( .INP(r_button), .ZN(n134) );
  AO22X1 U117 ( .IN1(v_in[2]), .IN2(n135), .IN3(n209), .IN4(n136), .Q(n129) );
  MUX21X1 U118 ( .IN1(n137), .IN2(n120), .S(n138), .Q(n116) );
  AO22X1 U119 ( .IN1(n139), .IN2(n212), .IN3(n140), .IN4(n141), .Q(n137) );
  NAND2X0 U120 ( .IN1(n142), .IN2(n143), .QN(n115) );
  MUX21X1 U121 ( .IN1(n144), .IN2(n211), .S(n138), .Q(n142) );
  NAND2X0 U122 ( .IN1(n140), .IN2(n145), .QN(n144) );
  OAI22X1 U123 ( .IN1(n146), .IN2(test), .IN3(n141), .IN4(n213), .QN(n145) );
  OAI22X1 U124 ( .IN1(n214), .IN2(n147), .IN3(n148), .IN4(n138), .QN(n114) );
  OA21X1 U125 ( .IN1(n121), .IN2(n149), .IN3(n150), .Q(n148) );
  NOR2X0 U126 ( .IN1(n140), .IN2(n138), .QN(n147) );
  NOR2X0 U127 ( .IN1(n139), .IN2(n151), .QN(n113) );
  OA22X1 U128 ( .IN1(n140), .IN2(n138), .IN3(n213), .IN4(n152), .Q(n151) );
  NOR2X0 U129 ( .IN1(n138), .IN2(n141), .QN(n152) );
  NAND4X0 U130 ( .IN1(n207), .IN2(n208), .IN3(n153), .IN4(n214), .QN(n141) );
  NOR2X0 U131 ( .IN1(n206), .IN2(n209), .QN(n153) );
  NAND4X0 U132 ( .IN1(n154), .IN2(n155), .IN3(n156), .IN4(n157), .QN(n138) );
  NAND3X0 U133 ( .IN1(n158), .IN2(n214), .IN3(rtr), .QN(n157) );
  AO21X1 U134 ( .IN1(n159), .IN2(n211), .IN3(n213), .Q(n156) );
  MUX41X1 U135 ( .IN1(rts), .IN3(n160), .IN2(rtr), .IN4(n161), .S0(n120), .S1(
        n214), .Q(n159) );
  NAND3X0 U136 ( .IN1(n140), .IN2(n162), .IN3(start), .QN(n155) );
  MUX21X1 U137 ( .IN1(n163), .IN2(n164), .S(n123), .Q(n154) );
  NAND2X0 U138 ( .IN1(n165), .IN2(n120), .QN(n164) );
  NAND4X0 U139 ( .IN1(v_in[2]), .IN2(n214), .IN3(v_in[0]), .IN4(n166), .QN(
        n165) );
  AND2X1 U140 ( .IN1(v_in[1]), .IN2(v_in[3]), .Q(n166) );
  NAND2X0 U141 ( .IN1(n167), .IN2(n168), .QN(n163) );
  OAI21X1 U142 ( .IN1(n169), .IN2(n205), .IN3(n143), .QN(n112) );
  NOR3X0 U143 ( .IN1(n170), .IN2(test), .IN3(n146), .QN(n169) );
  MUX21X1 U144 ( .IN1(n204), .IN2(g_button), .S(n128), .Q(n111) );
  AND2X1 U145 ( .IN1(start), .IN2(n136), .Q(n128) );
  MUX21X1 U146 ( .IN1(n122), .IN2(n171), .S(n172), .Q(n110) );
  OA21X1 U147 ( .IN1(n173), .IN2(n132), .IN3(n133), .Q(n172) );
  AOI221X1 U148 ( .IN1(n174), .IN2(n168), .IN3(n121), .IN4(n120), .IN5(n175), 
        .QN(n133) );
  NAND2X0 U149 ( .IN1(n176), .IN2(n149), .QN(n174) );
  NOR2X0 U150 ( .IN1(n204), .IN2(n177), .QN(n173) );
  INVX0 U151 ( .INP(g_button), .ZN(n177) );
  AO22X1 U152 ( .IN1(v_in[1]), .IN2(n135), .IN3(n206), .IN4(n136), .Q(n171) );
  MUX21X1 U153 ( .IN1(n178), .IN2(n126), .S(n179), .Q(n109) );
  OA221X1 U154 ( .IN1(n170), .IN2(n146), .IN3(n161), .IN4(n180), .IN5(n181), 
        .Q(n179) );
  OA221X1 U155 ( .IN1(n182), .IN2(n176), .IN3(n168), .IN4(n183), .IN5(n143), 
        .Q(n181) );
  NAND2X0 U156 ( .IN1(n213), .IN2(n211), .QN(n183) );
  AND2X1 U157 ( .IN1(n184), .IN2(n149), .Q(n182) );
  AO221X1 U158 ( .IN1(n205), .IN2(n158), .IN3(v_in[0]), .IN4(n135), .IN5(n136), 
        .Q(n178) );
  NOR2X0 U159 ( .IN1(n170), .IN2(n132), .QN(n136) );
  NAND2X0 U160 ( .IN1(key), .IN2(n162), .QN(n132) );
  INVX0 U161 ( .INP(n140), .ZN(n170) );
  MUX21X1 U162 ( .IN1(n127), .IN2(n185), .S(n186), .Q(n108) );
  NOR3X0 U163 ( .IN1(n187), .IN2(n175), .IN3(n188), .QN(n186) );
  MUX21X1 U164 ( .IN1(n189), .IN2(n123), .S(n121), .Q(n188) );
  NOR2X0 U165 ( .IN1(start), .IN2(n149), .QN(n189) );
  NAND3X0 U166 ( .IN1(n184), .IN2(n190), .IN3(n191), .QN(n175) );
  NAND2X0 U167 ( .IN1(n124), .IN2(n161), .QN(n191) );
  INVX0 U168 ( .INP(n158), .ZN(n184) );
  NOR2X0 U169 ( .IN1(n120), .IN2(n211), .QN(n158) );
  MUX21X1 U170 ( .IN1(n192), .IN2(n124), .S(n120), .Q(n187) );
  OA21X1 U171 ( .IN1(key), .IN2(n168), .IN3(n162), .Q(n192) );
  INVX0 U172 ( .INP(start), .ZN(n168) );
  AO22X1 U173 ( .IN1(v_in[3]), .IN2(n135), .IN3(n193), .IN4(n194), .Q(n185) );
  XNOR3X1 U174 ( .IN1(n122), .IN2(n208), .IN3(n125), .Q(n194) );
  NOR2X0 U175 ( .IN1(n176), .IN2(n149), .QN(n193) );
  NAND2X0 U176 ( .IN1(n143), .IN2(n180), .QN(n135) );
  NAND3X0 U177 ( .IN1(n121), .IN2(n124), .IN3(n140), .QN(n180) );
  NAND2X0 U178 ( .IN1(n167), .IN2(n123), .QN(n143) );
  MUX21X1 U179 ( .IN1(v_out[3]), .IN2(n127), .S(n195), .Q(n107) );
  MUX21X1 U180 ( .IN1(v_out[2]), .IN2(n125), .S(n195), .Q(n106) );
  MUX21X1 U181 ( .IN1(v_out[1]), .IN2(n122), .S(n195), .Q(n105) );
  MUX21X1 U182 ( .IN1(v_out[0]), .IN2(n126), .S(n195), .Q(n104) );
  AO221X1 U183 ( .IN1(n196), .IN2(rtr), .IN3(n197), .IN4(cts), .IN5(n195), .Q(
        n103) );
  AND3X1 U184 ( .IN1(rtr), .IN2(n140), .IN3(n198), .Q(n195) );
  NAND4X0 U185 ( .IN1(n199), .IN2(n200), .IN3(n190), .IN4(n176), .QN(n197) );
  AO21X1 U186 ( .IN1(n214), .IN2(n211), .IN3(n160), .Q(n200) );
  INVX0 U187 ( .INP(rtr), .ZN(n160) );
  MUX21X1 U188 ( .IN1(n214), .IN2(n211), .S(n120), .Q(n199) );
  NOR2X0 U189 ( .IN1(n123), .IN2(n150), .QN(n196) );
  INVX0 U190 ( .INP(n167), .ZN(n150) );
  NOR2X0 U191 ( .IN1(n146), .IN2(n212), .QN(n167) );
  INVX0 U192 ( .INP(n139), .ZN(n146) );
  NOR2X0 U193 ( .IN1(n121), .IN2(n124), .QN(n139) );
  AO22X1 U194 ( .IN1(n201), .IN2(ctr), .IN3(n202), .IN4(n198), .Q(n102) );
  NOR2X0 U195 ( .IN1(rts), .IN2(n149), .QN(n202) );
  NAND2X0 U196 ( .IN1(n211), .IN2(n120), .QN(n149) );
  NAND4X0 U197 ( .IN1(n140), .IN2(n203), .IN3(n190), .IN4(n176), .QN(n201) );
  INVX0 U198 ( .INP(n162), .ZN(n176) );
  NOR2X0 U199 ( .IN1(n124), .IN2(n214), .QN(n162) );
  INVX0 U200 ( .INP(n198), .ZN(n190) );
  NOR2X0 U201 ( .IN1(n121), .IN2(n213), .QN(n198) );
  NAND2X0 U202 ( .IN1(n121), .IN2(n161), .QN(n203) );
  INVX0 U203 ( .INP(rts), .ZN(n161) );
  NOR2X0 U204 ( .IN1(n123), .IN2(n120), .QN(n140) );
endmodule

