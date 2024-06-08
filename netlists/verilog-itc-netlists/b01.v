/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:09:11 2023
/////////////////////////////////////////////////////////////


module b01 ( line1, line2, reset, outp, overflw, clock );
  input line1, line2, reset, clock;
  output outp, overflw;
  wire   N41, N42, N43, N44, n1, n2, n22, n23, n24, n25, n26, n27, n28, n29,
         n30, n31, n32, n33, n34, n35, n36, n37, n38;
  wire   [2:0] stato;

  DFFARX1 stato_reg_0_ ( .D(N41), .CLK(clock), .RSTB(n1), .Q(stato[0]) );
  DFFARX1 stato_reg_2_ ( .D(N43), .CLK(clock), .RSTB(n1), .Q(stato[2]), .QN(
        n22) );
  DFFARX1 stato_reg_1_ ( .D(N42), .CLK(clock), .RSTB(n1), .Q(stato[1]), .QN(
        n23) );
  DFFARX1 overflw_reg ( .D(n2), .CLK(clock), .RSTB(n1), .Q(overflw) );
  DFFARX1 outp_reg ( .D(N44), .CLK(clock), .RSTB(n1), .Q(outp) );
  INVX0 U28 ( .INP(reset), .ZN(n1) );
  MUX21X1 U29 ( .IN1(n24), .IN2(n25), .S(n26), .Q(N44) );
  NAND2X0 U30 ( .IN1(n27), .IN2(n28), .QN(n25) );
  XOR2X1 U31 ( .IN1(line2), .IN2(line1), .Q(n24) );
  MUX21X1 U32 ( .IN1(n29), .IN2(n30), .S(n22), .Q(N43) );
  OR2X1 U33 ( .IN1(n31), .IN2(n32), .Q(n30) );
  OA21X1 U34 ( .IN1(stato[0]), .IN2(n27), .IN3(n23), .Q(n29) );
  AO222X1 U35 ( .IN1(n31), .IN2(n33), .IN3(n34), .IN4(stato[0]), .IN5(n35), 
        .IN6(stato[2]), .Q(N42) );
  MUX21X1 U36 ( .IN1(n23), .IN2(stato[0]), .S(n27), .Q(n35) );
  NOR2X0 U37 ( .IN1(stato[1]), .IN2(n32), .QN(n34) );
  NAND2X0 U38 ( .IN1(stato[2]), .IN2(n28), .QN(n33) );
  AO221X1 U39 ( .IN1(n36), .IN2(n37), .IN3(n26), .IN4(n27), .IN5(n38), .Q(N41)
         );
  MUX21X1 U40 ( .IN1(n2), .IN2(n31), .S(n32), .Q(n38) );
  INVX0 U41 ( .INP(n28), .ZN(n32) );
  AND3X1 U42 ( .IN1(stato[0]), .IN2(n22), .IN3(stato[1]), .Q(n2) );
  OR2X1 U43 ( .IN1(line1), .IN2(line2), .Q(n27) );
  NOR2X0 U44 ( .IN1(n22), .IN2(n31), .QN(n26) );
  NOR2X0 U45 ( .IN1(n23), .IN2(stato[0]), .QN(n31) );
  XOR2X1 U46 ( .IN1(n28), .IN2(stato[0]), .Q(n37) );
  NAND2X0 U47 ( .IN1(line1), .IN2(line2), .QN(n28) );
  NOR2X0 U48 ( .IN1(stato[2]), .IN2(stato[1]), .QN(n36) );
endmodule

