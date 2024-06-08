/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:09:09 2023
/////////////////////////////////////////////////////////////


module b02 ( reset, clock, linea, u );
  input reset, clock, linea;
  output u;
  wire   N33, N34, N35, N36, n1, n11, n12, n13, n14, n15, n16, n17, n18, n19,
         n20;
  wire   [2:0] stato;

  DFFARX1 stato_reg_0_ ( .D(N33), .CLK(clock), .RSTB(n1), .Q(stato[0]), .QN(
        n11) );
  DFFARX1 stato_reg_2_ ( .D(N35), .CLK(clock), .RSTB(n1), .Q(stato[2]), .QN(
        n13) );
  DFFARX1 stato_reg_1_ ( .D(N34), .CLK(clock), .RSTB(n1), .Q(stato[1]), .QN(
        n12) );
  DFFARX1 u_reg ( .D(N36), .CLK(clock), .RSTB(n1), .Q(u) );
  INVX0 U17 ( .INP(reset), .ZN(n1) );
  AND3X1 U18 ( .IN1(stato[2]), .IN2(n12), .IN3(n11), .Q(N36) );
  AO221X1 U19 ( .IN1(n14), .IN2(stato[1]), .IN3(linea), .IN4(n15), .IN5(n16), 
        .Q(N35) );
  XNOR2X1 U20 ( .IN1(stato[2]), .IN2(n17), .Q(n14) );
  AO221X1 U21 ( .IN1(n18), .IN2(stato[1]), .IN3(n15), .IN4(n19), .IN5(n16), 
        .Q(N34) );
  AND2X1 U22 ( .IN1(n15), .IN2(stato[2]), .Q(n16) );
  INVX0 U23 ( .INP(linea), .ZN(n19) );
  NOR2X0 U24 ( .IN1(n11), .IN2(stato[1]), .QN(n15) );
  NOR2X0 U25 ( .IN1(stato[2]), .IN2(stato[0]), .QN(n18) );
  AO22X1 U26 ( .IN1(n11), .IN2(n12), .IN3(n20), .IN4(n13), .Q(N33) );
  AO21X1 U27 ( .IN1(linea), .IN2(n12), .IN3(n17), .Q(n20) );
  NOR2X0 U28 ( .IN1(stato[0]), .IN2(linea), .QN(n17) );
endmodule

