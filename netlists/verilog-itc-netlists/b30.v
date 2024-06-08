/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:10:26 2023
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
         n150, n151, n152, n153, n154, n155, n156, n158, n275, n374, n375,
         n376, n377, n378, n379, n380, n381, n382, n383, n384, n385, n386,
         n387, n388, n389, n390, n391, n392, n393, n394, n395, n396, n397,
         n398, n399, n400, n401, n402, n403, n404, n405, n406, n407, n408,
         n409, n410, n411, n412, n413, n414, n415, n416, n417, n418, n419,
         n420, n421, n422, n423, n424, n425, n426, n427, n428, n429, n430,
         n431, n432, n433, n434, n435, n436, n437, n438, n439, n440, n441,
         n442, n443, n444, n445, n446, n447, n448, n449, n450, n451, n452,
         n453, n454, n455, n456, n457, n458, n459, n460, n461, n462, n463,
         n464, n465, n466, n467, n468, n469, n470, n471, n472, n473, n474,
         n475, n476, n477, n478, n479, n480;
  wire   [7:0] out_reg;
  wire   [9:0] tx_conta;
  wire   [9:2] add_291_carry;
  assign canale[3] = 1'b0;

  DFFARX1 next_bit_reg_0_ ( .D(n158), .CLK(clock), .RSTB(n275), .Q(next_bit_0_), .QN(n375) );
  DFFARX1 next_bit_reg_1_ ( .D(n156), .CLK(clock), .RSTB(n275), .Q(n378), .QN(
        n478) );
  DFFARX1 next_bit_reg_3_ ( .D(n155), .CLK(clock), .RSTB(n275), .Q(n379), .QN(
        n477) );
  DFFARX1 tx_end_reg ( .D(N230), .CLK(clock), .RSTB(n275), .QN(n462) );
  DFFARX1 confirm_reg ( .D(n154), .CLK(clock), .RSTB(n275), .Q(n386), .QN(n459) );
  DFFARX1 mpx_reg ( .D(n153), .CLK(clock), .RSTB(n275), .QN(n463) );
  DFFARX1 rdy_reg ( .D(n152), .CLK(clock), .RSTB(n275), .Q(rdy) );
  DFFARX1 S1_reg_1_ ( .D(N90), .CLK(clock), .RSTB(n275), .Q(n381), .QN(n475)
         );
  DFFARX1 S1_reg_2_ ( .D(N91), .CLK(clock), .RSTB(n275), .Q(n374), .QN(n474)
         );
  DFFARX1 S1_reg_0_ ( .D(N89), .CLK(clock), .RSTB(n275), .Q(n377), .QN(n473)
         );
  DFFARX1 send_data_reg ( .D(n151), .CLK(clock), .RSTB(n275), .Q(send_data) );
  DFFARX1 S2_reg_0_ ( .D(N115), .CLK(clock), .RSTB(n275), .Q(n384), .QN(n479)
         );
  DFFARX1 shot_reg ( .D(n150), .CLK(clock), .RSTB(n275), .Q(n460) );
  DFFARX1 itfc_state_reg_0_ ( .D(n149), .CLK(clock), .RSTB(n275), .Q(n388), 
        .QN(n464) );
  DFFARX1 itfc_state_reg_1_ ( .D(n148), .CLK(clock), .RSTB(n275), .Q(n376), 
        .QN(n465) );
  DFFARX1 send_reg ( .D(n147), .CLK(clock), .RSTB(n275), .Q(n470), .QN(n389)
         );
  DFFARX1 load_reg ( .D(n146), .CLK(clock), .RSTB(n275), .Q(n385), .QN(n471)
         );
  DFFARX1 S2_reg_1_ ( .D(N116), .CLK(clock), .RSTB(n275), .QN(n480) );
  DFFARX1 mux_en_reg ( .D(n145), .CLK(clock), .RSTB(n275), .Q(mux_en) );
  DFFARX1 conta_tmp_reg_0_ ( .D(n144), .CLK(clock), .RSTB(n275), .Q(n461) );
  DFFARX1 canale_reg_0_ ( .D(n143), .CLK(clock), .RSTB(n275), .Q(canale[0]) );
  DFFARX1 conta_tmp_reg_1_ ( .D(n142), .CLK(clock), .RSTB(n275), .Q(n387), 
        .QN(n466) );
  DFFARX1 canale_reg_1_ ( .D(n141), .CLK(clock), .RSTB(n275), .Q(canale[1]) );
  DFFARX1 conta_tmp_reg_2_ ( .D(n140), .CLK(clock), .RSTB(n275), .QN(n468) );
  DFFARX1 canale_reg_2_ ( .D(n139), .CLK(clock), .RSTB(n275), .Q(canale[2]), 
        .QN(n467) );
  DFFARX1 load_dato_reg ( .D(n137), .CLK(clock), .RSTB(n275), .Q(load_dato) );
  DFFARX1 soc_reg ( .D(n136), .CLK(clock), .RSTB(n275), .Q(soc) );
  DFFARX1 add_mpx2_reg ( .D(n135), .CLK(clock), .RSTB(n275), .Q(add_mpx2) );
  DFFARX1 tre_reg ( .D(n134), .CLK(clock), .RSTB(n275), .Q(n469), .QN(n380) );
  DFFARX1 error_reg ( .D(n133), .CLK(clock), .RSTB(n275), .Q(error) );
  DFFARX1 out_reg_reg_0_ ( .D(n132), .CLK(clock), .RSTB(n275), .Q(out_reg[0]), 
        .QN(n390) );
  DFFARX1 out_reg_reg_1_ ( .D(n131), .CLK(clock), .RSTB(n275), .Q(out_reg[1]), 
        .QN(n391) );
  DFFARX1 out_reg_reg_2_ ( .D(n130), .CLK(clock), .RSTB(n275), .Q(out_reg[2])
         );
  DFFARX1 out_reg_reg_3_ ( .D(n129), .CLK(clock), .RSTB(n275), .Q(out_reg[3])
         );
  DFFARX1 out_reg_reg_4_ ( .D(n128), .CLK(clock), .RSTB(n275), .Q(out_reg[4])
         );
  DFFARX1 out_reg_reg_5_ ( .D(n127), .CLK(clock), .RSTB(n275), .Q(out_reg[5])
         );
  DFFARX1 out_reg_reg_6_ ( .D(n126), .CLK(clock), .RSTB(n275), .Q(out_reg[6])
         );
  DFFARX1 out_reg_reg_7_ ( .D(n125), .CLK(clock), .RSTB(n275), .Q(out_reg[7])
         );
  DFFARX1 send_en_reg ( .D(n124), .CLK(clock), .RSTB(n275), .Q(n383), .QN(n472) );
  DFFARX1 tx_conta_reg_1_ ( .D(n123), .CLK(clock), .RSTB(n275), .Q(tx_conta[1]) );
  DFFARX1 tx_conta_reg_9_ ( .D(n122), .CLK(clock), .RSTB(n275), .Q(tx_conta[9]) );
  DFFARX1 tx_conta_reg_0_ ( .D(n121), .CLK(clock), .RSTB(n275), .Q(tx_conta[0]) );
  DFFARX1 tx_conta_reg_2_ ( .D(n120), .CLK(clock), .RSTB(n275), .Q(tx_conta[2]) );
  DFFARX1 tx_conta_reg_3_ ( .D(n119), .CLK(clock), .RSTB(n275), .Q(tx_conta[3]) );
  DFFARX1 tx_conta_reg_4_ ( .D(n118), .CLK(clock), .RSTB(n275), .Q(tx_conta[4]) );
  DFFARX1 tx_conta_reg_5_ ( .D(n117), .CLK(clock), .RSTB(n275), .Q(tx_conta[5]) );
  DFFARX1 tx_conta_reg_6_ ( .D(n116), .CLK(clock), .RSTB(n275), .Q(tx_conta[6]) );
  DFFARX1 tx_conta_reg_7_ ( .D(n115), .CLK(clock), .RSTB(n275), .Q(tx_conta[7]) );
  DFFARX1 tx_conta_reg_8_ ( .D(n114), .CLK(clock), .RSTB(n275), .Q(tx_conta[8]) );
  DFFARX1 next_bit_reg_2_ ( .D(n113), .CLK(clock), .RSTB(n275), .Q(n476), .QN(
        n382) );
  DFFARX1 data_out_reg ( .D(N231), .CLK(clock), .RSTB(n275), .Q(data_out) );
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
  INVX2 U290 ( .INP(reset), .ZN(n275) );
  INVX0 U291 ( .INP(n392), .ZN(n158) );
  MUX21X1 U292 ( .IN1(n375), .IN2(n393), .S(n394), .Q(n392) );
  MUX21X1 U293 ( .IN1(n395), .IN2(n477), .S(n396), .Q(n393) );
  AO222X1 U294 ( .IN1(n397), .IN2(n398), .IN3(n399), .IN4(n394), .IN5(n400), 
        .IN6(n378), .Q(n156) );
  NOR2X0 U295 ( .IN1(n476), .IN2(n395), .QN(n399) );
  INVX0 U296 ( .INP(n401), .ZN(n398) );
  OAI221X1 U297 ( .IN1(n402), .IN2(n401), .IN3(n394), .IN4(n477), .IN5(n403), 
        .QN(n155) );
  AO22X1 U298 ( .IN1(n404), .IN2(n386), .IN3(n405), .IN4(n376), .Q(n154) );
  NOR2X0 U299 ( .IN1(n462), .IN2(n464), .QN(n405) );
  NAND2X0 U300 ( .IN1(n464), .IN2(n465), .QN(n404) );
  XOR2X1 U301 ( .IN1(n406), .IN2(n463), .Q(n153) );
  NAND2X0 U302 ( .IN1(n407), .IN2(n408), .QN(n152) );
  OAI21X1 U303 ( .IN1(n406), .IN2(n463), .IN3(rdy), .QN(n408) );
  NAND2X0 U304 ( .IN1(n409), .IN2(n410), .QN(n151) );
  NAND3X0 U305 ( .IN1(n374), .IN2(n377), .IN3(n381), .QN(n410) );
  OAI21X1 U306 ( .IN1(n411), .IN2(n473), .IN3(send_data), .QN(n409) );
  AO22X1 U307 ( .IN1(n480), .IN2(n384), .IN3(n460), .IN4(n412), .Q(n150) );
  NAND2X0 U308 ( .IN1(n459), .IN2(n413), .QN(n412) );
  AO22X1 U309 ( .IN1(n462), .IN2(n376), .IN3(n464), .IN4(n414), .Q(n149) );
  OR2X1 U310 ( .IN1(n376), .IN2(n460), .Q(n414) );
  MUX21X1 U311 ( .IN1(n415), .IN2(n376), .S(n464), .Q(n148) );
  OR2X1 U312 ( .IN1(n462), .IN2(n465), .Q(n415) );
  AO22X1 U313 ( .IN1(n470), .IN2(n465), .IN3(n416), .IN4(n388), .Q(n147) );
  NAND2X0 U314 ( .IN1(n376), .IN2(n389), .QN(n416) );
  AO22X1 U315 ( .IN1(n385), .IN2(n376), .IN3(n464), .IN4(n417), .Q(n146) );
  AO21X1 U316 ( .IN1(n460), .IN2(n465), .IN3(n385), .Q(n417) );
  AO22X1 U317 ( .IN1(n418), .IN2(mux_en), .IN3(n419), .IN4(n473), .Q(n145) );
  NOR2X0 U318 ( .IN1(n381), .IN2(n374), .QN(n419) );
  AO21X1 U319 ( .IN1(n461), .IN2(n420), .IN3(n421), .Q(n144) );
  AO21X1 U320 ( .IN1(n420), .IN2(canale[0]), .IN3(n421), .Q(n143) );
  NAND2X0 U321 ( .IN1(n422), .IN2(n423), .QN(n142) );
  AO21X1 U322 ( .IN1(n424), .IN2(n461), .IN3(n466), .Q(n423) );
  AO221X1 U323 ( .IN1(n421), .IN2(n387), .IN3(n420), .IN4(canale[1]), .IN5(
        n425), .Q(n141) );
  INVX0 U324 ( .INP(n422), .ZN(n425) );
  NAND3X0 U325 ( .IN1(n424), .IN2(n461), .IN3(n466), .QN(n422) );
  NOR2X0 U326 ( .IN1(n420), .IN2(n461), .QN(n421) );
  NAND2X0 U327 ( .IN1(n426), .IN2(n427), .QN(n140) );
  AO21X1 U328 ( .IN1(n428), .IN2(n424), .IN3(n468), .Q(n427) );
  NAND2X0 U329 ( .IN1(n429), .IN2(n426), .QN(n139) );
  NAND3X0 U330 ( .IN1(n428), .IN2(n424), .IN3(n468), .QN(n426) );
  MUX21X1 U331 ( .IN1(n467), .IN2(n430), .S(n424), .Q(n429) );
  INVX0 U332 ( .INP(n420), .ZN(n424) );
  OR2X1 U333 ( .IN1(n428), .IN2(n468), .Q(n430) );
  AND2X1 U334 ( .IN1(n461), .IN2(n387), .Q(n428) );
  AO21X1 U335 ( .IN1(n420), .IN2(load_dato), .IN3(n431), .Q(n137) );
  INVX0 U336 ( .INP(n418), .ZN(n431) );
  NAND4X0 U337 ( .IN1(n475), .IN2(n432), .IN3(n374), .IN4(n377), .QN(n418) );
  NAND2X0 U338 ( .IN1(n433), .IN2(n374), .QN(n420) );
  MUX21X1 U339 ( .IN1(soc), .IN2(n474), .S(n433), .Q(n136) );
  AO21X1 U340 ( .IN1(n434), .IN2(n463), .IN3(add_mpx2), .Q(n135) );
  INVX0 U341 ( .INP(n406), .ZN(n434) );
  NAND3X0 U342 ( .IN1(n462), .IN2(n380), .IN3(n471), .QN(n134) );
  MUX21X1 U343 ( .IN1(n435), .IN2(n436), .S(n470), .Q(n133) );
  NAND2X0 U344 ( .IN1(dsr), .IN2(n469), .QN(n436) );
  MUX21X1 U345 ( .IN1(n469), .IN2(error), .S(n471), .Q(n435) );
  MUX21X1 U346 ( .IN1(out_reg[0]), .IN2(data_in[0]), .S(n437), .Q(n132) );
  MUX21X1 U347 ( .IN1(out_reg[1]), .IN2(data_in[1]), .S(n437), .Q(n131) );
  MUX21X1 U348 ( .IN1(out_reg[2]), .IN2(data_in[2]), .S(n437), .Q(n130) );
  MUX21X1 U349 ( .IN1(out_reg[3]), .IN2(data_in[3]), .S(n437), .Q(n129) );
  MUX21X1 U350 ( .IN1(out_reg[4]), .IN2(data_in[4]), .S(n437), .Q(n128) );
  MUX21X1 U351 ( .IN1(out_reg[5]), .IN2(data_in[5]), .S(n437), .Q(n127) );
  MUX21X1 U352 ( .IN1(out_reg[6]), .IN2(data_in[6]), .S(n437), .Q(n126) );
  MUX21X1 U353 ( .IN1(out_reg[7]), .IN2(data_in[7]), .S(n437), .Q(n125) );
  NOR2X0 U354 ( .IN1(n469), .IN2(n471), .QN(n437) );
  AO22X1 U355 ( .IN1(n462), .IN2(n383), .IN3(n438), .IN4(dsr), .Q(n124) );
  NOR2X0 U356 ( .IN1(n389), .IN2(n380), .QN(n438) );
  AO22X1 U357 ( .IN1(tx_conta[1]), .IN2(n472), .IN3(N209), .IN4(n439), .Q(n123) );
  AO22X1 U358 ( .IN1(tx_conta[9]), .IN2(n472), .IN3(add_291_carry[9]), .IN4(
        n439), .Q(n122) );
  MUX21X1 U359 ( .IN1(n439), .IN2(n472), .S(tx_conta[0]), .Q(n121) );
  AO22X1 U360 ( .IN1(tx_conta[2]), .IN2(n472), .IN3(N210), .IN4(n439), .Q(n120) );
  AO22X1 U361 ( .IN1(n472), .IN2(tx_conta[3]), .IN3(N211), .IN4(n439), .Q(n119) );
  AO22X1 U362 ( .IN1(tx_conta[4]), .IN2(n472), .IN3(N212), .IN4(n439), .Q(n118) );
  AO22X1 U363 ( .IN1(n472), .IN2(tx_conta[5]), .IN3(N213), .IN4(n439), .Q(n117) );
  AO22X1 U364 ( .IN1(n472), .IN2(tx_conta[6]), .IN3(N214), .IN4(n439), .Q(n116) );
  AO22X1 U365 ( .IN1(tx_conta[7]), .IN2(n472), .IN3(N215), .IN4(n439), .Q(n115) );
  AO22X1 U366 ( .IN1(tx_conta[8]), .IN2(n472), .IN3(N216), .IN4(n439), .Q(n114) );
  AND2X1 U367 ( .IN1(n440), .IN2(n383), .Q(n439) );
  AO21X1 U368 ( .IN1(n397), .IN2(n477), .IN3(n441), .Q(n113) );
  MUX21X1 U369 ( .IN1(n442), .IN2(n400), .S(n476), .Q(n441) );
  NAND2X0 U370 ( .IN1(n394), .IN2(n395), .QN(n400) );
  NOR2X0 U371 ( .IN1(n478), .IN2(n401), .QN(n442) );
  AO21X1 U372 ( .IN1(n374), .IN2(n377), .IN3(n433), .Q(N91) );
  NOR2X0 U373 ( .IN1(n377), .IN2(n475), .QN(n433) );
  MUX21X1 U374 ( .IN1(n443), .IN2(n374), .S(n473), .Q(N90) );
  AO22X1 U375 ( .IN1(n475), .IN2(n432), .IN3(n474), .IN4(n411), .Q(n443) );
  NAND2X0 U376 ( .IN1(rdy), .IN2(n381), .QN(n411) );
  INVX0 U377 ( .INP(eoc), .ZN(n432) );
  NAND2X0 U378 ( .IN1(n444), .IN2(n377), .QN(N89) );
  MUX21X1 U379 ( .IN1(n445), .IN2(n446), .S(n474), .Q(n444) );
  OR2X1 U380 ( .IN1(n475), .IN2(rdy), .Q(n446) );
  NAND2X0 U381 ( .IN1(eoc), .IN2(n475), .QN(n445) );
  OAI221X1 U382 ( .IN1(n447), .IN2(n375), .IN3(n391), .IN4(n403), .IN5(n448), 
        .QN(N231) );
  OA21X1 U383 ( .IN1(n449), .IN2(n395), .IN3(n394), .Q(n448) );
  NAND2X0 U384 ( .IN1(n477), .IN2(n375), .QN(n395) );
  AOI221X1 U385 ( .IN1(n397), .IN2(out_reg[5]), .IN3(n450), .IN4(out_reg[3]), 
        .IN5(n451), .QN(n449) );
  AND3X1 U386 ( .IN1(n378), .IN2(n382), .IN3(out_reg[7]), .Q(n451) );
  NOR2X0 U387 ( .IN1(n382), .IN2(n378), .QN(n397) );
  NAND3X0 U388 ( .IN1(n379), .IN2(n375), .IN3(n396), .QN(n403) );
  OA22X1 U389 ( .IN1(n452), .IN2(n379), .IN3(n453), .IN4(n390), .Q(n447) );
  AOI221X1 U390 ( .IN1(n382), .IN2(out_reg[6]), .IN3(n478), .IN4(out_reg[4]), 
        .IN5(n454), .QN(n452) );
  AO21X1 U391 ( .IN1(n450), .IN2(out_reg[2]), .IN3(n396), .Q(n454) );
  INVX0 U392 ( .INP(n453), .ZN(n396) );
  INVX0 U393 ( .INP(n402), .ZN(n450) );
  NAND2X0 U394 ( .IN1(n476), .IN2(n378), .QN(n402) );
  NOR2X0 U395 ( .IN1(n453), .IN2(n401), .QN(N230) );
  NAND3X0 U396 ( .IN1(n394), .IN2(n477), .IN3(next_bit_0_), .QN(n401) );
  NOR2X0 U397 ( .IN1(n440), .IN2(n472), .QN(n394) );
  NOR4X0 U398 ( .IN1(n455), .IN2(tx_conta[7]), .IN3(tx_conta[8]), .IN4(
        tx_conta[9]), .QN(n440) );
  AND3X1 U399 ( .IN1(tx_conta[5]), .IN2(n456), .IN3(tx_conta[6]), .Q(n455) );
  AO21X1 U400 ( .IN1(tx_conta[3]), .IN2(n457), .IN3(tx_conta[4]), .Q(n456) );
  OR3X1 U401 ( .IN1(tx_conta[1]), .IN2(tx_conta[2]), .IN3(tx_conta[0]), .Q(
        n457) );
  NAND2X0 U402 ( .IN1(n478), .IN2(n382), .QN(n453) );
  AO22X1 U403 ( .IN1(n480), .IN2(n384), .IN3(n413), .IN4(n458), .Q(N116) );
  NAND2X0 U404 ( .IN1(n463), .IN2(n386), .QN(n458) );
  NAND2X0 U405 ( .IN1(n406), .IN2(n407), .QN(N115) );
  NAND3X0 U406 ( .IN1(n480), .IN2(n479), .IN3(send_data), .QN(n407) );
  NAND2X0 U407 ( .IN1(n413), .IN2(n386), .QN(n406) );
  NOR2X0 U408 ( .IN1(n384), .IN2(n480), .QN(n413) );
endmodule

