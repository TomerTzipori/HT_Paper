/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:08:22 2023
/////////////////////////////////////////////////////////////


module b06 ( cc_mux, eql, uscite, clock, enable_count, ackout, reset, cont_eql
 );
  output [2:1] cc_mux;
  output [2:1] uscite;
  input eql, clock, reset, cont_eql;
  output enable_count, ackout;
  wire   N36, N37, N38, N40, N41, N42, N43, n2, n3, n28, n29, n30, n31, n32,
         n33, n34, n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46,
         n47, n48, n49, n50;
  wire   [2:0] state;

  DFFARX1 state_reg_0_ ( .D(N40), .CLK(clock), .RSTB(n2), .Q(state[0]), .QN(
        n29) );
  DFFARX1 state_reg_2_ ( .D(N42), .CLK(clock), .RSTB(n2), .Q(state[2]), .QN(
        n28) );
  DFFARX1 state_reg_1_ ( .D(N41), .CLK(clock), .RSTB(n2), .Q(state[1]), .QN(
        n30) );
  DFFARX1 uscite_reg_2_ ( .D(n3), .CLK(clock), .RSTB(n2), .Q(uscite[2]) );
  DFFARX1 uscite_reg_1_ ( .D(N38), .CLK(clock), .RSTB(n2), .Q(uscite[1]) );
  DFFARX1 cc_mux_reg_2_ ( .D(N37), .CLK(clock), .RSTB(n2), .Q(cc_mux[2]) );
  DFFARX1 cc_mux_reg_1_ ( .D(N36), .CLK(clock), .RSTB(n2), .Q(cc_mux[1]) );
  DFFARX1 enable_count_reg ( .D(N43), .CLK(clock), .RSTB(n2), .Q(enable_count)
         );
  DFFARX1 ackout_reg ( .D(N43), .CLK(clock), .RSTB(n2), .Q(ackout) );
  INVX0 U36 ( .INP(n31), .ZN(n3) );
  INVX0 U37 ( .INP(reset), .ZN(n2) );
  OAI21X1 U38 ( .IN1(eql), .IN2(n32), .IN3(cont_eql), .QN(N43) );
  OAI21X1 U39 ( .IN1(n33), .IN2(eql), .IN3(n34), .QN(N42) );
  NAND3X0 U40 ( .IN1(n31), .IN2(n35), .IN3(n36), .QN(N41) );
  OR2X1 U41 ( .IN1(n37), .IN2(n38), .Q(n35) );
  MUX21X1 U42 ( .IN1(n39), .IN2(n40), .S(n38), .Q(n31) );
  OAI21X1 U43 ( .IN1(n39), .IN2(eql), .IN3(n41), .QN(N40) );
  NAND3X0 U44 ( .IN1(n42), .IN2(n39), .IN3(n41), .QN(N38) );
  AOI21X1 U45 ( .IN1(n38), .IN2(n43), .IN3(n44), .QN(n41) );
  NAND3X0 U46 ( .IN1(n33), .IN2(n32), .IN3(n45), .QN(n43) );
  OR2X1 U47 ( .IN1(n40), .IN2(eql), .Q(n42) );
  NAND3X0 U48 ( .IN1(n46), .IN2(n33), .IN3(n34), .QN(N37) );
  AND2X1 U49 ( .IN1(n40), .IN2(n47), .Q(n34) );
  AO21X1 U50 ( .IN1(n39), .IN2(n45), .IN3(n38), .Q(n47) );
  OR2X1 U51 ( .IN1(n32), .IN2(n38), .Q(n46) );
  NAND4X0 U52 ( .IN1(n48), .IN2(n49), .IN3(n36), .IN4(n45), .QN(N36) );
  NAND3X0 U53 ( .IN1(state[2]), .IN2(n30), .IN3(state[0]), .QN(n45) );
  OA21X1 U54 ( .IN1(n33), .IN2(n38), .IN3(n32), .Q(n36) );
  NAND3X0 U55 ( .IN1(n29), .IN2(n28), .IN3(state[1]), .QN(n32) );
  NAND3X0 U56 ( .IN1(n30), .IN2(n28), .IN3(state[0]), .QN(n33) );
  INVX0 U57 ( .INP(n44), .ZN(n49) );
  NAND2X0 U58 ( .IN1(n37), .IN2(n50), .QN(n44) );
  NAND3X0 U59 ( .IN1(n30), .IN2(n28), .IN3(n29), .QN(n50) );
  NAND3X0 U60 ( .IN1(state[1]), .IN2(n28), .IN3(state[0]), .QN(n37) );
  MUX21X1 U61 ( .IN1(n40), .IN2(n39), .S(n38), .Q(n48) );
  INVX0 U62 ( .INP(eql), .ZN(n38) );
  NAND3X0 U63 ( .IN1(state[1]), .IN2(n29), .IN3(state[2]), .QN(n39) );
  NAND3X0 U64 ( .IN1(n29), .IN2(n30), .IN3(state[2]), .QN(n40) );
endmodule

