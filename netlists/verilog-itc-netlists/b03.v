/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:08:06 2023
/////////////////////////////////////////////////////////////


module b03 ( clock, reset, request1, request2, request3, request4, grant_o );
  output [3:0] grant_o;
  input clock, reset, request1, request2, request3, request4;
  wire   N70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83,
         n84, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97,
         n98, n99, n100, n101, n102, n103, n104, n105, n106, n107, n108, n109,
         n110, n111, n112, n113, n114, n115, n116, n117, n118, n119, n120,
         n121, n122, n123, n124, n125, n126, n127, n128, n129, n130, n131,
         n132, n133, n134, n135, n136, n137, n138, n139, n140, n141, n142,
         n143, n144, n145, n146, n147, n148, n149, n150, n151;

  DFFARX1 stato_reg_0_ ( .D(n101), .CLK(clock), .RSTB(n100), .Q(n71), .QN(n101) );
  DFFARX1 stato_reg_1_ ( .D(N70), .CLK(clock), .RSTB(n100), .Q(n106) );
  DFFARX1 ru1_reg ( .D(n99), .CLK(clock), .RSTB(n100), .Q(n102), .QN(n151) );
  DFFARX1 fu1_reg ( .D(n98), .CLK(clock), .RSTB(n100), .Q(n150) );
  DFFARX1 ru2_reg ( .D(n97), .CLK(clock), .RSTB(n100), .Q(n104), .QN(n149) );
  DFFARX1 fu2_reg ( .D(n96), .CLK(clock), .RSTB(n100), .Q(n148) );
  DFFARX1 ru3_reg ( .D(n95), .CLK(clock), .RSTB(n100), .Q(n105), .QN(n147) );
  DFFARX1 fu3_reg ( .D(n94), .CLK(clock), .RSTB(n100), .Q(n146) );
  DFFARX1 ru4_reg ( .D(n93), .CLK(clock), .RSTB(n100), .Q(n107), .QN(n145) );
  DFFARX1 fu4_reg ( .D(n92), .CLK(clock), .RSTB(n100), .Q(n144) );
  DFFARX1 coda0_reg_0_ ( .D(n91), .CLK(clock), .RSTB(n100), .Q(n142), .QN(n108) );
  DFFARX1 coda1_reg_0_ ( .D(n90), .CLK(clock), .RSTB(n100), .Q(n143) );
  DFFARX1 coda2_reg_0_ ( .D(n89), .CLK(clock), .RSTB(n100), .Q(n141) );
  DFFARX1 coda3_reg_0_ ( .D(n88), .CLK(clock), .RSTB(n100), .Q(n140) );
  DFFARX1 coda0_reg_1_ ( .D(n87), .CLK(clock), .RSTB(n100), .Q(n138), .QN(n103) );
  DFFARX1 coda1_reg_1_ ( .D(n86), .CLK(clock), .RSTB(n100), .Q(n139) );
  DFFARX1 coda2_reg_1_ ( .D(n85), .CLK(clock), .RSTB(n100), .Q(n137) );
  DFFARX1 coda3_reg_1_ ( .D(n84), .CLK(clock), .RSTB(n100), .Q(n136) );
  DFFARX1 coda0_reg_2_ ( .D(n83), .CLK(clock), .RSTB(n100), .Q(n134) );
  DFFARX1 coda1_reg_2_ ( .D(n82), .CLK(clock), .RSTB(n100), .Q(n135) );
  DFFARX1 coda2_reg_2_ ( .D(n81), .CLK(clock), .RSTB(n100), .Q(n133) );
  DFFARX1 coda3_reg_2_ ( .D(n80), .CLK(clock), .RSTB(n100), .Q(n132) );
  DFFARX1 grant_reg_3_ ( .D(n79), .CLK(clock), .RSTB(n100), .Q(n131) );
  DFFARX1 grant_o_reg_3_ ( .D(n78), .CLK(clock), .RSTB(n100), .Q(grant_o[3])
         );
  DFFARX1 grant_reg_2_ ( .D(n77), .CLK(clock), .RSTB(n100), .Q(n130) );
  DFFARX1 grant_o_reg_2_ ( .D(n76), .CLK(clock), .RSTB(n100), .Q(grant_o[2])
         );
  DFFARX1 grant_reg_1_ ( .D(n75), .CLK(clock), .RSTB(n100), .Q(n129) );
  DFFARX1 grant_o_reg_1_ ( .D(n74), .CLK(clock), .RSTB(n100), .Q(grant_o[1])
         );
  DFFARX1 grant_reg_0_ ( .D(n73), .CLK(clock), .RSTB(n100), .Q(n128) );
  DFFARX1 grant_o_reg_0_ ( .D(n72), .CLK(clock), .RSTB(n100), .Q(grant_o[0])
         );
  INVX2 U78 ( .INP(reset), .ZN(n100) );
  MUX21X1 U79 ( .IN1(n102), .IN2(request1), .S(n101), .Q(n99) );
  MUX21X1 U80 ( .IN1(n150), .IN2(n102), .S(N70), .Q(n98) );
  MUX21X1 U81 ( .IN1(n104), .IN2(request2), .S(n101), .Q(n97) );
  MUX21X1 U82 ( .IN1(n148), .IN2(n104), .S(N70), .Q(n96) );
  MUX21X1 U83 ( .IN1(n105), .IN2(request3), .S(n101), .Q(n95) );
  MUX21X1 U84 ( .IN1(n146), .IN2(n105), .S(N70), .Q(n94) );
  MUX21X1 U85 ( .IN1(n107), .IN2(request4), .S(n101), .Q(n93) );
  MUX21X1 U86 ( .IN1(n144), .IN2(n107), .S(N70), .Q(n92) );
  AO222X1 U87 ( .IN1(n143), .IN2(n109), .IN3(n110), .IN4(n111), .IN5(n142), 
        .IN6(n112), .Q(n91) );
  NOR2X0 U88 ( .IN1(n104), .IN2(n102), .QN(n110) );
  AO222X1 U89 ( .IN1(n142), .IN2(n111), .IN3(n141), .IN4(n109), .IN5(n143), 
        .IN6(n112), .Q(n90) );
  AO222X1 U90 ( .IN1(n143), .IN2(n111), .IN3(n140), .IN4(n109), .IN5(n141), 
        .IN6(n112), .Q(n89) );
  AO22X1 U91 ( .IN1(n140), .IN2(n112), .IN3(n141), .IN4(n111), .Q(n88) );
  AO222X1 U92 ( .IN1(n139), .IN2(n109), .IN3(n113), .IN4(n111), .IN5(n138), 
        .IN6(n112), .Q(n87) );
  OA21X1 U93 ( .IN1(n147), .IN2(n104), .IN3(n151), .Q(n113) );
  AO222X1 U94 ( .IN1(n138), .IN2(n111), .IN3(n137), .IN4(n109), .IN5(n139), 
        .IN6(n112), .Q(n86) );
  AO222X1 U95 ( .IN1(n139), .IN2(n111), .IN3(n136), .IN4(n109), .IN5(n137), 
        .IN6(n112), .Q(n85) );
  AO22X1 U96 ( .IN1(n136), .IN2(n112), .IN3(n137), .IN4(n111), .Q(n84) );
  AO222X1 U97 ( .IN1(n135), .IN2(n109), .IN3(n111), .IN4(n114), .IN5(n134), 
        .IN6(n112), .Q(n83) );
  AO21X1 U98 ( .IN1(n149), .IN2(n147), .IN3(n102), .Q(n114) );
  AO222X1 U99 ( .IN1(n134), .IN2(n111), .IN3(n133), .IN4(n109), .IN5(n135), 
        .IN6(n112), .Q(n82) );
  AO222X1 U100 ( .IN1(n135), .IN2(n111), .IN3(n132), .IN4(n109), .IN5(n133), 
        .IN6(n112), .Q(n81) );
  NOR2X0 U101 ( .IN1(n112), .IN2(n71), .QN(n109) );
  AO22X1 U102 ( .IN1(n132), .IN2(n112), .IN3(n133), .IN4(n111), .Q(n80) );
  AND2X1 U103 ( .IN1(n115), .IN2(N70), .Q(n111) );
  INVX0 U104 ( .INP(n115), .ZN(n112) );
  MUX21X1 U105 ( .IN1(n116), .IN2(n117), .S(n106), .Q(n115) );
  NOR2X0 U106 ( .IN1(n101), .IN2(n118), .QN(n116) );
  MUX21X1 U107 ( .IN1(n119), .IN2(n150), .S(n102), .Q(n118) );
  MUX21X1 U108 ( .IN1(n120), .IN2(n148), .S(n104), .Q(n119) );
  MUX21X1 U109 ( .IN1(n121), .IN2(n146), .S(n105), .Q(n120) );
  OR2X1 U110 ( .IN1(n144), .IN2(n145), .Q(n121) );
  MUX21X1 U111 ( .IN1(n131), .IN2(n122), .S(n123), .Q(n79) );
  AND3X1 U112 ( .IN1(n134), .IN2(n103), .IN3(n108), .Q(n122) );
  MUX21X1 U113 ( .IN1(grant_o[3]), .IN2(n131), .S(N70), .Q(n78) );
  MUX21X1 U114 ( .IN1(n130), .IN2(n124), .S(n123), .Q(n77) );
  NOR3X0 U115 ( .IN1(n103), .IN2(n134), .IN3(n142), .QN(n124) );
  MUX21X1 U116 ( .IN1(grant_o[2]), .IN2(n130), .S(N70), .Q(n76) );
  MUX21X1 U117 ( .IN1(n129), .IN2(n125), .S(n123), .Q(n75) );
  NOR3X0 U118 ( .IN1(n108), .IN2(n134), .IN3(n138), .QN(n125) );
  MUX21X1 U119 ( .IN1(grant_o[1]), .IN2(n129), .S(N70), .Q(n74) );
  MUX21X1 U120 ( .IN1(n128), .IN2(n126), .S(n123), .Q(n73) );
  AND2X1 U121 ( .IN1(n117), .IN2(n106), .Q(n123) );
  AND2X1 U122 ( .IN1(n101), .IN2(n127), .Q(n117) );
  OR4X1 U123 ( .IN1(n150), .IN2(n148), .IN3(n146), .IN4(n144), .Q(n127) );
  AND3X1 U124 ( .IN1(n138), .IN2(n142), .IN3(n134), .Q(n126) );
  MUX21X1 U125 ( .IN1(grant_o[0]), .IN2(n128), .S(N70), .Q(n72) );
  NOR2X0 U126 ( .IN1(n106), .IN2(n101), .QN(N70) );
endmodule

