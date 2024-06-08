/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:08:08 2023
/////////////////////////////////////////////////////////////


module b09 ( reset, clock, x, y );
  input reset, clock, x;
  output y;
  wire   N43, N44, N45, N46, N47, N48, N49, N50, N51, N52, N53, n43, n44, n45,
         n46, n47, n48, n49, n51, n54, n55, n56, n57, n58, n59, n60, n61, n62,
         n63, n64, n65, n66, n67, n68, n69, n71, n72, n73, n74, n75, n76, n77,
         n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91,
         n92, n93, n94, n95, n96, n97, n98, n99, n100, n101, n102, n103, n104,
         n105, n106, n107, n108, n109, n110, n111, n112, n113, n114, n115,
         n116, n117, n118, n119, n120, n121, n122;
  wire   [7:0] old;

  DFFARX1 old_reg_0_ ( .D(n71), .CLK(clock), .RSTB(n122), .Q(old[0]) );
  DFFARX1 d_in_reg_8_ ( .D(N53), .CLK(clock), .RSTB(n122), .Q(n73), .QN(n43)
         );
  DFFARX1 d_in_reg_7_ ( .D(N52), .CLK(clock), .RSTB(n122), .Q(n77), .QN(n44)
         );
  DFFARX1 d_in_reg_6_ ( .D(N51), .CLK(clock), .RSTB(n122), .Q(n76), .QN(n45)
         );
  DFFARX1 d_in_reg_5_ ( .D(N50), .CLK(clock), .RSTB(n122), .Q(n75), .QN(n46)
         );
  DFFARX1 d_in_reg_4_ ( .D(N49), .CLK(clock), .RSTB(n122), .Q(n74), .QN(n47)
         );
  DFFARX1 d_in_reg_3_ ( .D(N48), .CLK(clock), .RSTB(n122), .Q(n78), .QN(n48)
         );
  DFFARX1 d_in_reg_2_ ( .D(N47), .CLK(clock), .RSTB(n122), .Q(n81), .QN(n49)
         );
  DFFARX1 d_in_reg_1_ ( .D(N46), .CLK(clock), .RSTB(n122), .Q(n82), .QN(n51)
         );
  DFFARX1 d_in_reg_0_ ( .D(N45), .CLK(clock), .RSTB(n122), .Q(n118), .QN(n72)
         );
  DFFARX1 stato_reg_1_ ( .D(N44), .CLK(clock), .RSTB(n122), .Q(n80), .QN(n121)
         );
  DFFARX1 stato_reg_0_ ( .D(N43), .CLK(clock), .RSTB(n122), .Q(n79), .QN(n120)
         );
  DFFARX1 old_reg_1_ ( .D(n69), .CLK(clock), .RSTB(n122), .Q(old[1]) );
  DFFARX1 old_reg_2_ ( .D(n68), .CLK(clock), .RSTB(n122), .Q(old[2]) );
  DFFARX1 old_reg_3_ ( .D(n67), .CLK(clock), .RSTB(n122), .Q(old[3]) );
  DFFARX1 old_reg_4_ ( .D(n66), .CLK(clock), .RSTB(n122), .Q(old[4]) );
  DFFARX1 old_reg_5_ ( .D(n65), .CLK(clock), .RSTB(n122), .Q(old[5]) );
  DFFARX1 old_reg_6_ ( .D(n64), .CLK(clock), .RSTB(n122), .Q(old[6]) );
  DFFARX1 old_reg_7_ ( .D(n63), .CLK(clock), .RSTB(n122), .Q(old[7]) );
  DFFARX1 d_out_reg_7_ ( .D(n62), .CLK(clock), .RSTB(n122), .Q(n117) );
  DFFARX1 d_out_reg_6_ ( .D(n61), .CLK(clock), .RSTB(n122), .Q(n116) );
  DFFARX1 d_out_reg_5_ ( .D(n60), .CLK(clock), .RSTB(n122), .Q(n115) );
  DFFARX1 d_out_reg_4_ ( .D(n59), .CLK(clock), .RSTB(n122), .Q(n114) );
  DFFARX1 d_out_reg_3_ ( .D(n58), .CLK(clock), .RSTB(n122), .Q(n113) );
  DFFARX1 d_out_reg_2_ ( .D(n57), .CLK(clock), .RSTB(n122), .Q(n112) );
  DFFARX1 d_out_reg_1_ ( .D(n56), .CLK(clock), .RSTB(n122), .Q(n111) );
  DFFARX1 d_out_reg_0_ ( .D(n55), .CLK(clock), .RSTB(n122), .Q(n119) );
  DFFARX1 y_reg ( .D(n54), .CLK(clock), .RSTB(n122), .Q(y) );
  INVX2 U65 ( .INP(reset), .ZN(n122) );
  AO22X1 U66 ( .IN1(n83), .IN2(n82), .IN3(old[0]), .IN4(n84), .Q(n71) );
  AO22X1 U67 ( .IN1(n83), .IN2(n81), .IN3(old[1]), .IN4(n84), .Q(n69) );
  AO22X1 U68 ( .IN1(n83), .IN2(n78), .IN3(old[2]), .IN4(n84), .Q(n68) );
  AO22X1 U69 ( .IN1(n83), .IN2(n74), .IN3(old[3]), .IN4(n84), .Q(n67) );
  AO22X1 U70 ( .IN1(n83), .IN2(n75), .IN3(old[4]), .IN4(n84), .Q(n66) );
  AO22X1 U71 ( .IN1(n83), .IN2(n76), .IN3(old[5]), .IN4(n84), .Q(n65) );
  AO22X1 U72 ( .IN1(n83), .IN2(n77), .IN3(old[6]), .IN4(n84), .Q(n64) );
  AO22X1 U73 ( .IN1(n83), .IN2(n73), .IN3(old[7]), .IN4(n84), .Q(n63) );
  AO22X1 U74 ( .IN1(n117), .IN2(n85), .IN3(n86), .IN4(n73), .Q(n62) );
  AO222X1 U75 ( .IN1(n86), .IN2(n77), .IN3(n87), .IN4(n117), .IN5(n116), .IN6(
        n85), .Q(n61) );
  AO222X1 U76 ( .IN1(n86), .IN2(n76), .IN3(n116), .IN4(n87), .IN5(n115), .IN6(
        n85), .Q(n60) );
  AO222X1 U77 ( .IN1(n86), .IN2(n75), .IN3(n115), .IN4(n87), .IN5(n114), .IN6(
        n85), .Q(n59) );
  AO222X1 U78 ( .IN1(n86), .IN2(n74), .IN3(n114), .IN4(n87), .IN5(n113), .IN6(
        n85), .Q(n58) );
  AO222X1 U79 ( .IN1(n86), .IN2(n78), .IN3(n113), .IN4(n87), .IN5(n112), .IN6(
        n85), .Q(n57) );
  AO222X1 U80 ( .IN1(n86), .IN2(n81), .IN3(n112), .IN4(n87), .IN5(n111), .IN6(
        n85), .Q(n56) );
  AO222X1 U81 ( .IN1(n86), .IN2(n82), .IN3(n111), .IN4(n87), .IN5(n119), .IN6(
        n85), .Q(n55) );
  NOR2X0 U82 ( .IN1(n88), .IN2(n85), .QN(n87) );
  NOR2X0 U83 ( .IN1(n85), .IN2(n120), .QN(n86) );
  NOR2X0 U84 ( .IN1(n89), .IN2(n90), .QN(n85) );
  MUX21X1 U85 ( .IN1(n121), .IN2(n120), .S(n72), .Q(n89) );
  NAND3X0 U86 ( .IN1(n91), .IN2(n92), .IN3(n93), .QN(n54) );
  INVX0 U87 ( .INP(n94), .ZN(n93) );
  NAND3X0 U88 ( .IN1(n95), .IN2(n72), .IN3(n119), .QN(n92) );
  NAND3X0 U89 ( .IN1(n121), .IN2(n79), .IN3(y), .QN(n91) );
  AO21X1 U90 ( .IN1(x), .IN2(n84), .IN3(n94), .Q(N53) );
  AO21X1 U91 ( .IN1(n121), .IN2(n83), .IN3(n90), .Q(n94) );
  NOR2X0 U92 ( .IN1(n96), .IN2(n97), .QN(n90) );
  INVX0 U93 ( .INP(n96), .ZN(n83) );
  INVX0 U94 ( .INP(n98), .ZN(n84) );
  NOR2X0 U95 ( .IN1(n43), .IN2(n98), .QN(N52) );
  NOR2X0 U96 ( .IN1(n44), .IN2(n98), .QN(N51) );
  NOR2X0 U97 ( .IN1(n45), .IN2(n98), .QN(N50) );
  NOR2X0 U98 ( .IN1(n46), .IN2(n98), .QN(N49) );
  NOR2X0 U99 ( .IN1(n47), .IN2(n98), .QN(N48) );
  NOR2X0 U100 ( .IN1(n48), .IN2(n98), .QN(N47) );
  NOR2X0 U101 ( .IN1(n49), .IN2(n98), .QN(N46) );
  NOR2X0 U102 ( .IN1(n51), .IN2(n98), .QN(N45) );
  OA21X1 U103 ( .IN1(n118), .IN2(n120), .IN3(n88), .Q(n98) );
  NAND2X0 U104 ( .IN1(n121), .IN2(n96), .QN(N44) );
  NAND2X0 U105 ( .IN1(n118), .IN2(n79), .QN(n96) );
  AO21X1 U106 ( .IN1(n95), .IN2(n118), .IN3(n99), .Q(N43) );
  MUX21X1 U107 ( .IN1(n100), .IN2(n121), .S(n120), .Q(n99) );
  AO21X1 U108 ( .IN1(n97), .IN2(n80), .IN3(n72), .Q(n100) );
  AND4X1 U109 ( .IN1(n101), .IN2(n102), .IN3(n103), .IN4(n104), .Q(n97) );
  NOR4X0 U110 ( .IN1(n105), .IN2(n106), .IN3(n107), .IN4(n108), .QN(n104) );
  XOR2X1 U111 ( .IN1(old[3]), .IN2(n74), .Q(n108) );
  XOR2X1 U112 ( .IN1(old[4]), .IN2(n75), .Q(n107) );
  XOR2X1 U113 ( .IN1(old[5]), .IN2(n76), .Q(n106) );
  XOR2X1 U114 ( .IN1(old[6]), .IN2(n77), .Q(n105) );
  NOR2X0 U115 ( .IN1(n109), .IN2(n110), .QN(n103) );
  XOR2X1 U116 ( .IN1(old[2]), .IN2(n78), .Q(n110) );
  XOR2X1 U117 ( .IN1(old[7]), .IN2(n73), .Q(n109) );
  XOR2X1 U118 ( .IN1(old[1]), .IN2(n49), .Q(n102) );
  XOR2X1 U119 ( .IN1(old[0]), .IN2(n51), .Q(n101) );
  INVX0 U120 ( .INP(n88), .ZN(n95) );
  NAND2X0 U121 ( .IN1(n120), .IN2(n80), .QN(n88) );
endmodule

