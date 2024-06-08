/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:08:07 2023
/////////////////////////////////////////////////////////////


module b08 ( CLOCK, RESET, START, I, O );
  input [7:0] I;
  output [3:0] O;
  input CLOCK, RESET, START;
  wire   n76, n77, n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89,
         n90, n91, n92, n93, n94, n95, n96, n97, n98, n99, n100, n101, n102,
         n103, n104, n105, n106, n107, n108, n109, n110, n111, n112, n113,
         n114, n115, n116, n117, n118, n119, n120, n121, n122, n123, n124,
         n125, n126, n127, n128, n129, n130, n131, n132, n133, n134, n135,
         n136, n137, n138, n139, n140, n141, n142, n143, n144, n145, n146,
         n147, n148, n149, n150, n151, n152, n153, n154, n155, n156, n157,
         n158, n159, n160, n161, n162, n163, n164, n165;

  DFFARX1 MAR_reg_0_ ( .D(n97), .CLK(CLOCK), .RSTB(n96), .Q(n165), .QN(n100)
         );
  DFFARX1 MAR_reg_1_ ( .D(n95), .CLK(CLOCK), .RSTB(n96), .Q(n161), .QN(n99) );
  DFFARX1 MAR_reg_2_ ( .D(n94), .CLK(CLOCK), .RSTB(n96), .Q(n162), .QN(n101)
         );
  DFFARX1 STATO_reg_0_ ( .D(n93), .CLK(CLOCK), .RSTB(n96), .Q(n164), .QN(n98)
         );
  DFFARX1 STATO_reg_1_ ( .D(n92), .CLK(CLOCK), .RSTB(n96), .Q(n163), .QN(n103)
         );
  DFFARX1 IN_R_reg_7_ ( .D(n91), .CLK(CLOCK), .RSTB(n96), .Q(n160) );
  DFFARX1 IN_R_reg_6_ ( .D(n90), .CLK(CLOCK), .RSTB(n96), .Q(n159), .QN(n106)
         );
  DFFARX1 IN_R_reg_5_ ( .D(n89), .CLK(CLOCK), .RSTB(n96), .Q(n158), .QN(n102)
         );
  DFFARX1 IN_R_reg_4_ ( .D(n88), .CLK(CLOCK), .RSTB(n96), .Q(n157), .QN(n104)
         );
  DFFARX1 IN_R_reg_3_ ( .D(n87), .CLK(CLOCK), .RSTB(n96), .Q(n156), .QN(n105)
         );
  DFFARX1 IN_R_reg_2_ ( .D(n86), .CLK(CLOCK), .RSTB(n96), .Q(n155), .QN(n107)
         );
  DFFARX1 IN_R_reg_1_ ( .D(n85), .CLK(CLOCK), .RSTB(n96), .Q(n154) );
  DFFARX1 IN_R_reg_0_ ( .D(n84), .CLK(CLOCK), .RSTB(n96), .Q(n153) );
  DFFARX1 OUT_R_reg_3_ ( .D(n83), .CLK(CLOCK), .RSTB(n96), .Q(n152) );
  DFFARX1 O_reg_3_ ( .D(n82), .CLK(CLOCK), .RSTB(n96), .Q(O[3]) );
  DFFARX1 OUT_R_reg_2_ ( .D(n81), .CLK(CLOCK), .RSTB(n96), .Q(n151) );
  DFFARX1 O_reg_2_ ( .D(n80), .CLK(CLOCK), .RSTB(n96), .Q(O[2]) );
  DFFARX1 OUT_R_reg_1_ ( .D(n79), .CLK(CLOCK), .RSTB(n96), .Q(n150) );
  DFFARX1 O_reg_1_ ( .D(n78), .CLK(CLOCK), .RSTB(n96), .Q(O[1]) );
  DFFARX1 OUT_R_reg_0_ ( .D(n77), .CLK(CLOCK), .RSTB(n96), .Q(n149) );
  DFFARX1 O_reg_0_ ( .D(n76), .CLK(CLOCK), .RSTB(n96), .Q(O[0]) );
  INVX2 U83 ( .INP(RESET), .ZN(n96) );
  MUX21X1 U84 ( .IN1(n108), .IN2(n165), .S(n98), .Q(n97) );
  NOR2X0 U85 ( .IN1(n109), .IN2(n103), .QN(n108) );
  AO22X1 U86 ( .IN1(n161), .IN2(n98), .IN3(n163), .IN4(n110), .Q(n95) );
  NAND2X0 U87 ( .IN1(n111), .IN2(n112), .QN(n110) );
  MUX21X1 U88 ( .IN1(n165), .IN2(n113), .S(n99), .Q(n111) );
  NAND2X0 U89 ( .IN1(n164), .IN2(n165), .QN(n113) );
  AO21X1 U90 ( .IN1(n162), .IN2(n114), .IN3(n115), .Q(n94) );
  NOR4X0 U91 ( .IN1(n99), .IN2(n100), .IN3(n98), .IN4(n103), .QN(n115) );
  AO22X1 U92 ( .IN1(START), .IN2(n98), .IN3(n163), .IN4(n116), .Q(n93) );
  AO21X1 U93 ( .IN1(START), .IN2(n117), .IN3(n98), .Q(n116) );
  AO21X1 U94 ( .IN1(n163), .IN2(n118), .IN3(n119), .Q(n92) );
  MUX21X1 U95 ( .IN1(I[7]), .IN2(n160), .S(n114), .Q(n91) );
  MUX21X1 U96 ( .IN1(I[6]), .IN2(n159), .S(n114), .Q(n90) );
  MUX21X1 U97 ( .IN1(I[5]), .IN2(n158), .S(n114), .Q(n89) );
  MUX21X1 U98 ( .IN1(I[4]), .IN2(n157), .S(n114), .Q(n88) );
  MUX21X1 U99 ( .IN1(I[3]), .IN2(n156), .S(n114), .Q(n87) );
  MUX21X1 U100 ( .IN1(I[2]), .IN2(n155), .S(n114), .Q(n86) );
  MUX21X1 U101 ( .IN1(I[1]), .IN2(n154), .S(n114), .Q(n85) );
  MUX21X1 U102 ( .IN1(I[0]), .IN2(n153), .S(n114), .Q(n84) );
  OA21X1 U103 ( .IN1(n120), .IN2(n152), .IN3(n114), .Q(n83) );
  NOR3X0 U104 ( .IN1(n121), .IN2(n117), .IN3(n122), .QN(n120) );
  NOR2X0 U105 ( .IN1(n100), .IN2(n123), .QN(n122) );
  MUX21X1 U106 ( .IN1(O[3]), .IN2(n152), .S(n124), .Q(n82) );
  AO222X1 U107 ( .IN1(n125), .IN2(n117), .IN3(n126), .IN4(n127), .IN5(n151), 
        .IN6(n114), .Q(n81) );
  NOR2X0 U108 ( .IN1(n128), .IN2(n129), .QN(n126) );
  MUX21X1 U109 ( .IN1(O[2]), .IN2(n151), .S(n124), .Q(n80) );
  AO221X1 U110 ( .IN1(n125), .IN2(n101), .IN3(n150), .IN4(n114), .IN5(n127), 
        .Q(n79) );
  INVX0 U111 ( .INP(n121), .ZN(n125) );
  MUX21X1 U112 ( .IN1(O[1]), .IN2(n150), .S(n124), .Q(n78) );
  AO22X1 U113 ( .IN1(n149), .IN2(n114), .IN3(n127), .IN4(n161), .Q(n77) );
  NOR2X0 U114 ( .IN1(n121), .IN2(n165), .QN(n127) );
  NAND4X0 U115 ( .IN1(n163), .IN2(n130), .IN3(n131), .IN4(n132), .QN(n121) );
  OA221X1 U116 ( .IN1(n133), .IN2(n101), .IN3(n134), .IN4(n99), .IN5(n98), .Q(
        n132) );
  OA222X1 U117 ( .IN1(n135), .IN2(n165), .IN3(n160), .IN4(n104), .IN5(n162), 
        .IN6(n136), .Q(n134) );
  NOR4X0 U118 ( .IN1(n153), .IN2(n155), .IN3(n157), .IN4(n137), .QN(n135) );
  MUX21X1 U119 ( .IN1(n162), .IN2(n138), .S(n154), .Q(n137) );
  NAND3X0 U120 ( .IN1(n162), .IN2(n102), .IN3(n160), .QN(n138) );
  OA22X1 U121 ( .IN1(n165), .IN2(n156), .IN3(n161), .IN4(n153), .Q(n133) );
  MUX21X1 U122 ( .IN1(n139), .IN2(n140), .S(n123), .Q(n131) );
  NAND2X0 U123 ( .IN1(n109), .IN2(n107), .QN(n140) );
  NOR4X0 U124 ( .IN1(n160), .IN2(n141), .IN3(n102), .IN4(n105), .QN(n139) );
  NOR2X0 U125 ( .IN1(n165), .IN2(n159), .QN(n141) );
  NAND2X0 U126 ( .IN1(n165), .IN2(n142), .QN(n130) );
  NAND4X0 U127 ( .IN1(n143), .IN2(n144), .IN3(n145), .IN4(n136), .QN(n142) );
  AO21X1 U128 ( .IN1(n109), .IN2(n123), .IN3(n106), .Q(n136) );
  INVX0 U129 ( .INP(n129), .ZN(n123) );
  NOR2X0 U130 ( .IN1(n100), .IN2(n128), .QN(n109) );
  OAI21X1 U131 ( .IN1(n155), .IN2(n154), .IN3(n129), .QN(n145) );
  NOR2X0 U132 ( .IN1(n161), .IN2(n162), .QN(n129) );
  NAND2X0 U133 ( .IN1(n160), .IN2(n128), .QN(n144) );
  MUX21X1 U134 ( .IN1(n146), .IN2(n147), .S(n99), .Q(n143) );
  OA21X1 U135 ( .IN1(n159), .IN2(n101), .IN3(n153), .Q(n147) );
  NAND2X0 U136 ( .IN1(n101), .IN2(n104), .QN(n146) );
  INVX0 U137 ( .INP(n119), .ZN(n114) );
  NOR2X0 U138 ( .IN1(n98), .IN2(n163), .QN(n119) );
  MUX21X1 U139 ( .IN1(O[0]), .IN2(n149), .S(n124), .Q(n76) );
  NOR2X0 U140 ( .IN1(n118), .IN2(n103), .QN(n124) );
  NAND3X0 U141 ( .IN1(n164), .IN2(n148), .IN3(n117), .QN(n118) );
  NOR2X0 U142 ( .IN1(n112), .IN2(n100), .QN(n117) );
  INVX0 U143 ( .INP(n128), .ZN(n112) );
  NOR2X0 U144 ( .IN1(n99), .IN2(n101), .QN(n128) );
  INVX0 U145 ( .INP(START), .ZN(n148) );
endmodule

