/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP2
// Date      : Sun Nov 19 22:09:15 2023
/////////////////////////////////////////////////////////////


module b05 ( CLOCK, RESET, START, SIGN, DISPMAX1, DISPMAX2, DISPMAX3, DISPNUM1, 
        DISPNUM2 );
  output [6:0] DISPMAX1;
  output [6:0] DISPMAX2;
  output [6:0] DISPMAX3;
  output [6:0] DISPNUM1;
  output [6:0] DISPNUM2;
  input CLOCK, RESET, START;
  output SIGN;
  wire   N464, N468, N469, N470, N471, N472, N473, N474, N475, N476, N477,
         N485, N486, N487, N488, N512, N513, N514, N515, N519, N621, N622,
         N623, N624, N625, N626, N627, N628, N830, N831, N832, N1015, N1016,
         N1017, N1025, N1026, N1027, U3_U2_Z_0, U3_U2_Z_1, U3_U2_Z_2,
         U3_U2_Z_3, n383, n384, n385, n389, n399, n400, n401, n402, n403, n404,
         n405, n406, n407, n408, n409, n410, n411, n412, n413, n414, n415,
         n416, n417, n418, n419, n420, n421, n422, n423, n424, n425, n426,
         n427, n428, n430, r471_DIFF_1_, r471_DIFF_2_, r471_DIFF_3_,
         r471_DIFF_4_, r471_DIFF_5_, r471_DIFF_6_, r471_DIFF_7_, r471_DIFF_8_,
         sub_73_carry_8_, n432, n433, n434, n435, n436, n437, n438, n439, n440,
         n441, n442, n443, n444, n445, n446, n447, n448, n449, n450, n451,
         n452, n453, n454, n455, n456, n457, n458, n459, n460, n461, n462,
         n463, n464, n465, n466, n467, n468, n469, n470, n471, n472, n473,
         n474, n475, n476, n477, n478, n479, n480, n481, n482, n483, n484,
         n485, n486, n487, n488, n489, n490, n491, n492, n493, n494, n495,
         n496, n497, n498, n499, n500, n501, n502, n503, n504, n505, n506,
         n507, n508, n509, n510, n511, n512, n513, n514, n515, n516, n517,
         n518, n519, n520, n521, n522, n523, n524, n525, n526, n527, n528,
         n529, n530, n531, n532, n533, n534, n535, n536, n537, n538, n539,
         n540, n541, n542, n543, n544, n545, n546, n547, n548, n549, n550,
         n551, n552, n553, n554, n555, n556, n557, n558, n559, n560, n561,
         n562, n563, n564, n565, n566, n567, n568, n569, n570, n571, n572,
         n573, n574, n575, n576, n577, n578, n579, n580, n581, n582, n583,
         n584, n585, n586, n587, n588, n589, n590, n591, n592, n593, n594,
         n595, n596, n597, n598, n599, n600, n601, n602, n603, n604, n605,
         n606, n607, n608, n609, n610, n611, n612, n613, n614, n615, n616,
         n617, n618, n619, n620, n621, n622, n623, n624, n625, n626, n627,
         n628, n629, n630, n631, n632, n633, n634, n635, n636, n637, n638,
         n639, n640, n641, n642, n643, n644, n645, n646, n647, n648, n649,
         n650, n651, n652, n653, n654, n655, n656, n657, n658, n659, n660,
         n661, n662, n663, n664, n665, n666, n667, n668, n669, n670, n671,
         n672, n673, n674, n675, n676, n677, n678, n679, n680, n681, n682,
         n683, n684, n685, n686, n687, n688, n689, n690, n691, n692, n693,
         n694, n695, n696, n697, n698, n699, n700, n701, n702, n703;
  wire   [7:0] AC1;
  wire   [4:1] NUM;
  wire   [8:2] sub_148_aco_carry;
  wire   [5:3] sub_199_carry;
  wire   [8:2] r471_carry;
  wire   [4:2] r94_carry;
  wire   [8:1] sub_60_carry;
  assign DISPMAX1[6] = DISPNUM1[6];
  assign DISPNUM1[2] = DISPNUM1[5];
  assign DISPNUM1[0] = DISPNUM1[5];
  assign DISPNUM1[1] = DISPNUM1[5];
  assign DISPMAX2[3] = DISPNUM1[4];
  assign DISPMAX1[5] = DISPNUM1[4];
  assign DISPMAX1[1] = DISPNUM1[4];
  assign DISPMAX1[0] = DISPNUM1[4];
  assign DISPMAX1[2] = DISPNUM1[4];
  assign DISPNUM1[3] = DISPNUM1[4];
  assign DISPMAX1[3] = DISPNUM1[4];
  assign DISPMAX1[4] = DISPNUM1[4];

  DFFARX1 TEMP_reg_0_ ( .D(n430), .CLK(CLOCK), .RSTB(n686), .Q(n451), .QN(n687) );
  DFFARX1 FLAG_reg ( .D(n428), .CLK(CLOCK), .RSTB(n686), .Q(n676) );
  DFFARX1 NUM_reg_1_ ( .D(n427), .CLK(CLOCK), .RSTB(n686), .Q(NUM[1]), .QN(
        n389) );
  DFFARX1 MAR_reg_4_ ( .D(n426), .CLK(CLOCK), .RSTB(n686), .Q(n679), .QN(n432)
         );
  DFFARX1 STATO_reg_1_ ( .D(N1026), .CLK(CLOCK), .RSTB(n686), .Q(n683), .QN(
        n445) );
  DFFARX1 STATO_reg_2_ ( .D(N1027), .CLK(CLOCK), .RSTB(n686), .Q(n680), .QN(
        n438) );
  DFFARX1 STATO_reg_0_ ( .D(N1025), .CLK(CLOCK), .RSTB(n686), .Q(n684) );
  DFFARX1 RES_DISP_reg ( .D(n425), .CLK(CLOCK), .RSTB(n686), .Q(n685) );
  DFFARX1 EN_DISP_reg ( .D(n424), .CLK(CLOCK), .RSTB(n686), .Q(n675), .QN(n435) );
  DFFARX1 NUM_reg_3_ ( .D(n423), .CLK(CLOCK), .RSTB(n686), .Q(NUM[3]) );
  DFFARX1 NUM_reg_2_ ( .D(n422), .CLK(CLOCK), .RSTB(n686), .Q(NUM[2]), .QN(
        n385) );
  DFFARX1 NUM_reg_0_ ( .D(n421), .CLK(CLOCK), .RSTB(n686), .Q(n444), .QN(n384)
         );
  DFFARX1 MAR_reg_3_ ( .D(n420), .CLK(CLOCK), .RSTB(n686), .Q(n682), .QN(n433)
         );
  DFFARX1 MAR_reg_2_ ( .D(n419), .CLK(CLOCK), .RSTB(n686), .Q(n681), .QN(n440)
         );
  DFFARX1 MAR_reg_1_ ( .D(n418), .CLK(CLOCK), .RSTB(n686), .Q(n678), .QN(n441)
         );
  DFFARX1 MAR_reg_0_ ( .D(n417), .CLK(CLOCK), .RSTB(n686), .Q(n677), .QN(n434)
         );
  DFFARX1 TEMP_reg_5_ ( .D(n416), .CLK(CLOCK), .RSTB(n686), .Q(n447), .QN(n688) );
  DFFARX1 TEMP_reg_6_ ( .D(n415), .CLK(CLOCK), .RSTB(n686), .QN(n689) );
  DFFARX1 TEMP_reg_8_ ( .D(n414), .CLK(CLOCK), .RSTB(n686), .Q(n448), .QN(n690) );
  DFFARX1 TEMP_reg_2_ ( .D(n413), .CLK(CLOCK), .RSTB(n686), .QN(n691) );
  DFFARX1 TEMP_reg_7_ ( .D(n412), .CLK(CLOCK), .RSTB(n686), .Q(n449), .QN(n692) );
  DFFARX1 TEMP_reg_3_ ( .D(n411), .CLK(CLOCK), .RSTB(n686), .QN(n693) );
  DFFARX1 TEMP_reg_1_ ( .D(n410), .CLK(CLOCK), .RSTB(n686), .QN(n694) );
  DFFARX1 TEMP_reg_4_ ( .D(n409), .CLK(CLOCK), .RSTB(n686), .QN(n695) );
  DFFARX1 NUM_reg_4_ ( .D(n408), .CLK(CLOCK), .RSTB(n686), .Q(NUM[4]), .QN(
        n383) );
  DFFARX1 MAX_reg_0_ ( .D(n407), .CLK(CLOCK), .RSTB(n686), .Q(n443), .QN(n698)
         );
  DFFARX1 MAX_reg_1_ ( .D(n406), .CLK(CLOCK), .RSTB(n686), .Q(n437), .QN(n699)
         );
  DFFARX1 MAX_reg_2_ ( .D(n405), .CLK(CLOCK), .RSTB(n686), .Q(n439), .QN(n700)
         );
  DFFARX1 MAX_reg_3_ ( .D(n404), .CLK(CLOCK), .RSTB(n686), .Q(n442), .QN(n701)
         );
  DFFARX1 MAX_reg_4_ ( .D(n403), .CLK(CLOCK), .RSTB(n686), .Q(n436), .QN(n702)
         );
  DFFARX1 MAX_reg_5_ ( .D(n402), .CLK(CLOCK), .RSTB(n686), .Q(n446) );
  DFFARX1 MAX_reg_6_ ( .D(n401), .CLK(CLOCK), .RSTB(n686), .Q(n450) );
  DFFARX1 MAX_reg_7_ ( .D(n400), .CLK(CLOCK), .RSTB(n686), .Q(n452), .QN(n674)
         );
  DFFARX1 MAX_reg_8_ ( .D(n399), .CLK(CLOCK), .RSTB(n686), .Q(n455), .QN(n703)
         );
  FADDX1 sub_148_aco_U2_3 ( .A(N514), .B(n697), .CI(sub_148_aco_carry[3]), 
        .CO(sub_148_aco_carry[4]), .S(N623) );
  FADDX1 r471_U2_3 ( .A(N514), .B(n696), .CI(n453), .CO(r471_carry[4]), .S(
        r471_DIFF_3_) );
  HADDX1 r94_U1_1_1 ( .A0(U3_U2_Z_1), .B0(U3_U2_Z_0), .C1(r94_carry[2]), .SO(
        N1015) );
  HADDX1 r94_U1_1_2 ( .A0(U3_U2_Z_2), .B0(r94_carry[2]), .C1(r94_carry[3]), 
        .SO(N1016) );
  HADDX1 r94_U1_1_3 ( .A0(U3_U2_Z_3), .B0(r94_carry[3]), .C1(r94_carry[4]), 
        .SO(N1017) );
  FADDX1 sub_73_U2_8 ( .A(N468), .B(n703), .CI(sub_73_carry_8_), .S(N477) );
  FADDX1 sub_60_U2_1 ( .A(N475), .B(n694), .CI(sub_60_carry[1]), .CO(
        sub_60_carry[2]), .S(AC1[1]) );
  FADDX1 sub_60_U2_2 ( .A(N474), .B(n691), .CI(sub_60_carry[2]), .CO(
        sub_60_carry[3]), .S(AC1[2]) );
  FADDX1 sub_60_U2_3 ( .A(N473), .B(n693), .CI(sub_60_carry[3]), .CO(
        sub_60_carry[4]), .S(AC1[3]) );
  FADDX1 sub_60_U2_4 ( .A(N472), .B(n695), .CI(sub_60_carry[4]), .CO(
        sub_60_carry[5]), .S(AC1[4]) );
  FADDX1 sub_60_U2_5 ( .A(N471), .B(n688), .CI(sub_60_carry[5]), .CO(
        sub_60_carry[6]), .S(AC1[5]) );
  FADDX1 sub_60_U2_6 ( .A(N470), .B(n689), .CI(sub_60_carry[6]), .CO(
        sub_60_carry[7]), .S(AC1[6]) );
  FADDX1 sub_60_U2_7 ( .A(N469), .B(n692), .CI(sub_60_carry[7]), .CO(
        sub_60_carry[8]), .S(AC1[7]) );
  FADDX1 sub_60_U2_8 ( .A(N468), .B(n690), .CI(sub_60_carry[8]), .S(N464) );
  INVX2 U392 ( .INP(RESET), .ZN(n686) );
  INVX0 U393 ( .INP(sub_148_aco_carry[7]), .ZN(n466) );
  NAND2X1 U394 ( .IN1(n466), .IN2(n476), .QN(sub_148_aco_carry[8]) );
  NAND2X1 U395 ( .IN1(n463), .IN2(n465), .QN(sub_148_aco_carry[6]) );
  INVX0 U396 ( .INP(sub_148_aco_carry[5]), .ZN(n463) );
  NAND2X1 U397 ( .IN1(n475), .IN2(n476), .QN(r471_carry[6]) );
  INVX0 U398 ( .INP(n454), .ZN(n475) );
  NAND2X1 U399 ( .IN1(n464), .IN2(n465), .QN(sub_148_aco_carry[7]) );
  INVX0 U400 ( .INP(sub_148_aco_carry[6]), .ZN(n464) );
  NAND2X1 U401 ( .IN1(n474), .IN2(n471), .QN(r471_carry[7]) );
  INVX0 U402 ( .INP(r471_carry[6]), .ZN(n474) );
  NAND2X1 U403 ( .IN1(n470), .IN2(n471), .QN(r471_carry[8]) );
  INVX0 U404 ( .INP(r471_carry[7]), .ZN(n470) );
  NAND2X1 U405 ( .IN1(n666), .IN2(n472), .QN(r471_carry[2]) );
  NAND2X1 U406 ( .IN1(n461), .IN2(n462), .QN(sub_148_aco_carry[5]) );
  INVX0 U407 ( .INP(sub_148_aco_carry[4]), .ZN(n461) );
  NAND2X1 U408 ( .IN1(n468), .IN2(n472), .QN(sub_148_aco_carry[2]) );
  INVX0 U409 ( .INP(n697), .ZN(n468) );
  NAND2X1 U410 ( .IN1(n469), .IN2(n383), .QN(sub_199_carry[5]) );
  INVX0 U411 ( .INP(n459), .ZN(n469) );
  INVX0 U412 ( .INP(N519), .ZN(n476) );
  INVX0 U413 ( .INP(N519), .ZN(n465) );
  INVX0 U414 ( .INP(N519), .ZN(n471) );
  INVX0 U415 ( .INP(N519), .ZN(n467) );
  AND2X1 U416 ( .IN1(r471_carry[2]), .IN2(N513), .Q(n453) );
  INVX0 U417 ( .INP(N515), .ZN(n462) );
  NAND2X1 U418 ( .IN1(n460), .IN2(n670), .QN(sub_148_aco_carry[3]) );
  INVX0 U419 ( .INP(sub_148_aco_carry[2]), .ZN(n460) );
  INVX0 U420 ( .INP(N512), .ZN(n472) );
  INVX0 U421 ( .INP(N476), .ZN(n473) );
  AND2X1 U422 ( .IN1(r471_carry[4]), .IN2(N515), .Q(n454) );
  AND2X1 U423 ( .IN1(n477), .IN2(n455), .Q(N519) );
  NAND2X1 U424 ( .IN1(n702), .IN2(n458), .QN(n477) );
  NAND2X1 U425 ( .IN1(n451), .IN2(n473), .QN(sub_60_carry[1]) );
  AND2X1 U426 ( .IN1(n699), .IN2(n698), .Q(n456) );
  AND2X1 U427 ( .IN1(n700), .IN2(n456), .Q(n457) );
  AND2X1 U428 ( .IN1(n701), .IN2(n457), .Q(n458) );
  NAND2X1 U429 ( .IN1(n389), .IN2(n385), .QN(sub_199_carry[3]) );
  AND2X1 U430 ( .IN1(sub_199_carry[3]), .IN2(NUM[3]), .Q(n459) );
  XOR2X1 U431 ( .IN1(n670), .IN2(sub_148_aco_carry[2]), .Q(N622) );
  XOR2X1 U432 ( .IN1(n462), .IN2(sub_148_aco_carry[4]), .Q(N624) );
  XOR2X1 U433 ( .IN1(n471), .IN2(sub_148_aco_carry[5]), .Q(N625) );
  XOR2X1 U434 ( .IN1(n465), .IN2(sub_148_aco_carry[6]), .Q(N626) );
  XOR2X1 U435 ( .IN1(n467), .IN2(sub_148_aco_carry[7]), .Q(N627) );
  XOR2X1 U436 ( .IN1(n467), .IN2(sub_148_aco_carry[8]), .Q(N628) );
  XOR2X1 U437 ( .IN1(n472), .IN2(n697), .Q(N621) );
  XOR2X1 U438 ( .IN1(n385), .IN2(NUM[1]), .Q(N830) );
  XOR2X1 U439 ( .IN1(n383), .IN2(n459), .Q(N832) );
  XOR2X1 U440 ( .IN1(n471), .IN2(r471_carry[7]), .Q(r471_DIFF_7_) );
  XOR2X1 U441 ( .IN1(n467), .IN2(r471_carry[8]), .Q(r471_DIFF_8_) );
  XOR2X1 U442 ( .IN1(n472), .IN2(n696), .Q(r471_DIFF_1_) );
  XOR2X1 U443 ( .IN1(n473), .IN2(n687), .Q(AC1[0]) );
  XOR2X1 U444 ( .IN1(n465), .IN2(r471_carry[6]), .Q(r471_DIFF_6_) );
  XOR2X1 U445 ( .IN1(n476), .IN2(n454), .Q(r471_DIFF_5_) );
  XOR2X1 U446 ( .IN1(sub_199_carry[3]), .IN2(NUM[3]), .Q(N831) );
  XOR2X1 U447 ( .IN1(n699), .IN2(n698), .Q(N485) );
  XOR2X1 U448 ( .IN1(n700), .IN2(n456), .Q(N486) );
  XOR2X1 U449 ( .IN1(n701), .IN2(n457), .Q(N487) );
  XOR2X1 U450 ( .IN1(n702), .IN2(n458), .Q(N488) );
  XOR2X1 U451 ( .IN1(r471_carry[2]), .IN2(N513), .Q(r471_DIFF_2_) );
  XOR2X1 U452 ( .IN1(r471_carry[4]), .IN2(N515), .Q(r471_DIFF_4_) );
  AO22X1 U453 ( .IN1(n478), .IN2(N469), .IN3(n674), .IN4(n479), .Q(
        sub_73_carry_8_) );
  OR2X1 U454 ( .IN1(n478), .IN2(N469), .Q(n479) );
  AOI22X1 U455 ( .IN1(n480), .IN2(n481), .IN3(n482), .IN4(n450), .QN(n478) );
  OR2X1 U456 ( .IN1(n480), .IN2(n481), .Q(n482) );
  AO22X1 U457 ( .IN1(n483), .IN2(n484), .IN3(n485), .IN4(n446), .Q(n480) );
  OR2X1 U458 ( .IN1(n483), .IN2(n484), .Q(n485) );
  INVX0 U459 ( .INP(N471), .ZN(n484) );
  AO22X1 U460 ( .IN1(n486), .IN2(n487), .IN3(n488), .IN4(n436), .Q(n483) );
  OR2X1 U461 ( .IN1(n486), .IN2(n487), .Q(n488) );
  AO22X1 U462 ( .IN1(n489), .IN2(n490), .IN3(n491), .IN4(n442), .Q(n486) );
  OR2X1 U463 ( .IN1(n489), .IN2(n490), .Q(n491) );
  AO22X1 U464 ( .IN1(n492), .IN2(n493), .IN3(n494), .IN4(n439), .Q(n489) );
  OR2X1 U465 ( .IN1(n492), .IN2(n493), .Q(n494) );
  AO22X1 U466 ( .IN1(n495), .IN2(n437), .IN3(n496), .IN4(n497), .Q(n492) );
  NAND2X0 U467 ( .IN1(n699), .IN2(N475), .QN(n497) );
  NOR2X0 U468 ( .IN1(n698), .IN2(N476), .QN(n496) );
  MUX21X1 U469 ( .IN1(N476), .IN2(n451), .S(n498), .Q(n430) );
  AO22X1 U470 ( .IN1(n676), .IN2(n499), .IN3(n500), .IN4(n501), .Q(n428) );
  AND2X1 U471 ( .IN1(n502), .IN2(n503), .Q(n500) );
  AO22X1 U472 ( .IN1(n499), .IN2(NUM[1]), .IN3(N1015), .IN4(n504), .Q(n427) );
  AO22X1 U473 ( .IN1(n679), .IN2(n505), .IN3(n506), .IN4(n507), .Q(n426) );
  AO22X1 U474 ( .IN1(n508), .IN2(n684), .IN3(n685), .IN4(n509), .Q(n425) );
  NAND2X0 U475 ( .IN1(n510), .IN2(n438), .QN(n509) );
  AO22X1 U476 ( .IN1(n508), .IN2(n511), .IN3(n675), .IN4(n512), .Q(n424) );
  AO21X1 U477 ( .IN1(n680), .IN2(n513), .IN3(n511), .Q(n512) );
  AND3X1 U478 ( .IN1(n445), .IN2(n438), .IN3(START), .Q(n508) );
  AO22X1 U479 ( .IN1(n499), .IN2(NUM[3]), .IN3(N1017), .IN4(n504), .Q(n423) );
  AO22X1 U480 ( .IN1(n499), .IN2(NUM[2]), .IN3(N1016), .IN4(n504), .Q(n422) );
  AO22X1 U481 ( .IN1(n499), .IN2(n444), .IN3(n504), .IN4(n514), .Q(n421) );
  AO22X1 U482 ( .IN1(n682), .IN2(n505), .IN3(N1017), .IN4(n506), .Q(n420) );
  AO22X1 U483 ( .IN1(n681), .IN2(n505), .IN3(N1016), .IN4(n506), .Q(n419) );
  AO22X1 U484 ( .IN1(n678), .IN2(n505), .IN3(n506), .IN4(N1015), .Q(n418) );
  AO22X1 U485 ( .IN1(n677), .IN2(n505), .IN3(n506), .IN4(n514), .Q(n417) );
  NOR2X0 U486 ( .IN1(n505), .IN2(n511), .QN(n506) );
  MUX21X1 U487 ( .IN1(n447), .IN2(n515), .S(n516), .Q(n416) );
  INVX0 U488 ( .INP(n517), .ZN(n415) );
  MUX21X1 U489 ( .IN1(n689), .IN2(n481), .S(n516), .Q(n517) );
  INVX0 U490 ( .INP(N470), .ZN(n481) );
  MUX21X1 U491 ( .IN1(n448), .IN2(N468), .S(n516), .Q(n414) );
  INVX0 U492 ( .INP(n518), .ZN(n413) );
  MUX21X1 U493 ( .IN1(n691), .IN2(n493), .S(n516), .Q(n518) );
  INVX0 U494 ( .INP(N474), .ZN(n493) );
  MUX21X1 U495 ( .IN1(n449), .IN2(N469), .S(n516), .Q(n412) );
  INVX0 U496 ( .INP(n519), .ZN(n411) );
  MUX21X1 U497 ( .IN1(n693), .IN2(n490), .S(n516), .Q(n519) );
  INVX0 U498 ( .INP(N473), .ZN(n490) );
  INVX0 U499 ( .INP(n520), .ZN(n410) );
  MUX21X1 U500 ( .IN1(n694), .IN2(n495), .S(n516), .Q(n520) );
  INVX0 U501 ( .INP(N475), .ZN(n495) );
  INVX0 U502 ( .INP(n521), .ZN(n409) );
  MUX21X1 U503 ( .IN1(n695), .IN2(n487), .S(n516), .Q(n521) );
  INVX0 U504 ( .INP(N472), .ZN(n487) );
  AO22X1 U505 ( .IN1(n499), .IN2(NUM[4]), .IN3(n507), .IN4(n504), .Q(n408) );
  NOR2X0 U506 ( .IN1(n499), .IN2(n522), .QN(n504) );
  XNOR2X1 U507 ( .IN1(r94_carry[4]), .IN2(n523), .Q(n507) );
  OA22X1 U508 ( .IN1(n383), .IN2(n524), .IN3(n432), .IN4(n525), .Q(n523) );
  NAND3X0 U509 ( .IN1(n526), .IN2(n438), .IN3(n527), .QN(n499) );
  AO21X1 U510 ( .IN1(n676), .IN2(N464), .IN3(n445), .Q(n526) );
  MUX21X1 U511 ( .IN1(n443), .IN2(N476), .S(n528), .Q(n407) );
  MUX21X1 U512 ( .IN1(n437), .IN2(N475), .S(n528), .Q(n406) );
  MUX21X1 U513 ( .IN1(n439), .IN2(N474), .S(n528), .Q(n405) );
  MUX21X1 U514 ( .IN1(n442), .IN2(N473), .S(n528), .Q(n404) );
  MUX21X1 U515 ( .IN1(n436), .IN2(N472), .S(n528), .Q(n403) );
  MUX21X1 U516 ( .IN1(n446), .IN2(n515), .S(n528), .Q(n402) );
  AO221X1 U517 ( .IN1(n529), .IN2(n434), .IN3(n530), .IN4(n441), .IN5(n531), 
        .Q(n515) );
  AO21X1 U518 ( .IN1(n532), .IN2(n533), .IN3(n534), .Q(n531) );
  MUX21X1 U519 ( .IN1(n450), .IN2(n535), .S(n528), .Q(n401) );
  OA221X1 U520 ( .IN1(n432), .IN2(n536), .IN3(n537), .IN4(n538), .IN5(n539), 
        .Q(n535) );
  NAND2X0 U521 ( .IN1(n540), .IN2(n433), .QN(n538) );
  MUX21X1 U522 ( .IN1(n452), .IN2(N469), .S(n528), .Q(n400) );
  MUX21X1 U523 ( .IN1(n455), .IN2(N468), .S(n528), .Q(n399) );
  AOI21X1 U524 ( .IN1(n684), .IN2(n541), .IN3(n498), .QN(n528) );
  NAND3X0 U525 ( .IN1(n503), .IN2(n502), .IN3(N477), .QN(n541) );
  INVX0 U526 ( .INP(N464), .ZN(n502) );
  NAND2X0 U527 ( .IN1(n542), .IN2(n543), .QN(n503) );
  NOR4X0 U528 ( .IN1(AC1[7]), .IN2(AC1[6]), .IN3(AC1[5]), .IN4(AC1[4]), .QN(
        n543) );
  NOR4X0 U529 ( .IN1(AC1[3]), .IN2(AC1[2]), .IN3(AC1[1]), .IN4(AC1[0]), .QN(
        n542) );
  AO22X1 U530 ( .IN1(n544), .IN2(n682), .IN3(n545), .IN4(NUM[3]), .Q(U3_U2_Z_3) );
  AO22X1 U531 ( .IN1(n544), .IN2(n681), .IN3(n545), .IN4(NUM[2]), .Q(U3_U2_Z_2) );
  AO22X1 U532 ( .IN1(n544), .IN2(n678), .IN3(n545), .IN4(NUM[1]), .Q(U3_U2_Z_1) );
  INVX0 U533 ( .INP(n524), .ZN(n545) );
  INVX0 U534 ( .INP(n525), .ZN(n544) );
  INVX0 U535 ( .INP(n514), .ZN(U3_U2_Z_0) );
  OA22X1 U536 ( .IN1(n525), .IN2(n434), .IN3(n524), .IN4(n384), .Q(n514) );
  NAND3X0 U537 ( .IN1(N464), .IN2(n501), .IN3(n676), .QN(n524) );
  INVX0 U538 ( .INP(n522), .ZN(n501) );
  AO21X1 U539 ( .IN1(n435), .IN2(n455), .IN3(DISPNUM1[6]), .Q(SIGN) );
  AND2X1 U540 ( .IN1(n546), .IN2(n539), .Q(N476) );
  OA22X1 U541 ( .IN1(n677), .IN2(n547), .IN3(n678), .IN4(n548), .Q(n539) );
  OA21X1 U542 ( .IN1(n681), .IN2(n549), .IN3(n536), .Q(n548) );
  OA21X1 U543 ( .IN1(n433), .IN2(n441), .IN3(n540), .Q(n547) );
  MUX21X1 U544 ( .IN1(n550), .IN2(n434), .S(n551), .Q(n546) );
  OA21X1 U545 ( .IN1(n681), .IN2(n682), .IN3(n679), .Q(n551) );
  AO221X1 U546 ( .IN1(n552), .IN2(n441), .IN3(n682), .IN4(n553), .IN5(n554), 
        .Q(N475) );
  AO22X1 U547 ( .IN1(n532), .IN2(n679), .IN3(n555), .IN4(n440), .Q(n554) );
  NAND3X0 U548 ( .IN1(n556), .IN2(n557), .IN3(n558), .QN(n553) );
  NAND3X0 U549 ( .IN1(n432), .IN2(n434), .IN3(n678), .QN(n558) );
  NAND3X0 U550 ( .IN1(n559), .IN2(n560), .IN3(n561), .QN(N474) );
  OA22X1 U551 ( .IN1(n529), .IN2(n533), .IN3(n537), .IN4(n549), .Q(n561) );
  NAND3X0 U552 ( .IN1(n433), .IN2(n434), .IN3(n681), .QN(n560) );
  AO21X1 U553 ( .IN1(n678), .IN2(n679), .IN3(n562), .Q(n559) );
  AO221X1 U554 ( .IN1(n555), .IN2(n563), .IN3(n529), .IN4(n564), .IN5(n565), 
        .Q(N473) );
  AO21X1 U555 ( .IN1(n566), .IN2(n679), .IN3(n567), .Q(n565) );
  MUX21X1 U556 ( .IN1(n568), .IN2(n569), .S(n677), .Q(n567) );
  OA21X1 U557 ( .IN1(n570), .IN2(n571), .IN3(n441), .Q(n569) );
  AO221X1 U558 ( .IN1(n568), .IN2(n678), .IN3(n532), .IN4(n572), .IN5(n573), 
        .Q(N472) );
  OAI22X1 U559 ( .IN1(n574), .IN2(n550), .IN3(n575), .IN4(n557), .QN(n573) );
  OA22X1 U560 ( .IN1(n677), .IN2(n571), .IN3(n433), .IN4(n432), .Q(n575) );
  NOR2X0 U561 ( .IN1(n530), .IN2(n571), .QN(n574) );
  INVX0 U562 ( .INP(n549), .ZN(n571) );
  AO221X1 U563 ( .IN1(n529), .IN2(n533), .IN3(n677), .IN4(n530), .IN5(n576), 
        .Q(N471) );
  AO21X1 U564 ( .IN1(n532), .IN2(n682), .IN3(n534), .Q(n576) );
  AO221X1 U565 ( .IN1(n564), .IN2(n537), .IN3(n555), .IN4(n433), .IN5(n577), 
        .Q(n534) );
  AO21X1 U566 ( .IN1(n570), .IN2(n677), .IN3(n568), .Q(n577) );
  INVX0 U567 ( .INP(n578), .ZN(n568) );
  INVX0 U568 ( .INP(n557), .ZN(n529) );
  NAND2X0 U569 ( .IN1(n678), .IN2(n681), .QN(n557) );
  NAND4X0 U570 ( .IN1(n579), .IN2(n556), .IN3(n562), .IN4(n580), .QN(N470) );
  OA21X1 U571 ( .IN1(n677), .IN2(n533), .IN3(n581), .Q(n580) );
  MUX21X1 U572 ( .IN1(n582), .IN2(n540), .S(n532), .Q(n581) );
  NAND2X0 U573 ( .IN1(n681), .IN2(n433), .QN(n582) );
  NAND2X0 U574 ( .IN1(n563), .IN2(n441), .QN(n556) );
  NAND4X0 U575 ( .IN1(n583), .IN2(n584), .IN3(n585), .IN4(n578), .QN(N469) );
  NAND2X0 U576 ( .IN1(n682), .IN2(n530), .QN(n578) );
  INVX0 U577 ( .INP(n540), .ZN(n530) );
  NAND2X0 U578 ( .IN1(n432), .IN2(n440), .QN(n540) );
  AO21X1 U579 ( .IN1(n681), .IN2(n550), .IN3(n533), .Q(n585) );
  NAND2X0 U580 ( .IN1(n532), .IN2(n432), .QN(n584) );
  INVX0 U581 ( .INP(n586), .ZN(n583) );
  MUX21X1 U582 ( .IN1(n552), .IN2(n563), .S(n677), .Q(n586) );
  NOR2X0 U583 ( .IN1(n432), .IN2(n681), .QN(n563) );
  AO221X1 U584 ( .IN1(n555), .IN2(n572), .IN3(n566), .IN4(n679), .IN5(n587), 
        .Q(N468) );
  INVX0 U585 ( .INP(n579), .ZN(n587) );
  NAND3X0 U586 ( .IN1(n682), .IN2(n432), .IN3(n532), .QN(n579) );
  INVX0 U587 ( .INP(n562), .ZN(n566) );
  NAND3X0 U588 ( .IN1(n682), .IN2(n440), .IN3(n677), .QN(n562) );
  AO21X1 U589 ( .IN1(n564), .IN2(n440), .IN3(n552), .Q(n572) );
  NOR2X0 U590 ( .IN1(n549), .IN2(n440), .QN(n552) );
  NAND2X0 U591 ( .IN1(n432), .IN2(n433), .QN(n549) );
  INVX0 U592 ( .INP(n533), .ZN(n564) );
  NAND2X0 U593 ( .IN1(n679), .IN2(n433), .QN(n533) );
  INVX0 U594 ( .INP(n550), .ZN(n555) );
  NAND2X0 U595 ( .IN1(n441), .IN2(n434), .QN(n550) );
  NAND2X0 U596 ( .IN1(n522), .IN2(n588), .QN(N1027) );
  NAND4X0 U597 ( .IN1(n589), .IN2(n510), .IN3(n680), .IN4(START), .QN(n588) );
  NAND2X0 U598 ( .IN1(n684), .IN2(n516), .QN(n522) );
  INVX0 U599 ( .INP(n498), .ZN(n516) );
  OAI21X1 U600 ( .IN1(n498), .IN2(n684), .IN3(n505), .QN(N1026) );
  NAND2X0 U601 ( .IN1(n590), .IN2(n445), .QN(n505) );
  MUX21X1 U602 ( .IN1(n591), .IN2(n592), .S(n680), .Q(n590) );
  NOR2X0 U603 ( .IN1(n589), .IN2(n684), .QN(n592) );
  INVX0 U604 ( .INP(n513), .ZN(n589) );
  AND2X1 U605 ( .IN1(n684), .IN2(START), .Q(n591) );
  NAND2X0 U606 ( .IN1(n683), .IN2(n438), .QN(n498) );
  OAI221X1 U607 ( .IN1(n527), .IN2(n680), .IN3(n511), .IN4(START), .IN5(n525), 
        .QN(N1025) );
  NAND3X0 U608 ( .IN1(n680), .IN2(n513), .IN3(n510), .QN(n525) );
  NAND3X0 U609 ( .IN1(n532), .IN2(n679), .IN3(n570), .QN(n513) );
  INVX0 U610 ( .INP(n536), .ZN(n570) );
  NAND2X0 U611 ( .IN1(n681), .IN2(n682), .QN(n536) );
  INVX0 U612 ( .INP(n537), .ZN(n532) );
  NAND2X0 U613 ( .IN1(n677), .IN2(n678), .QN(n537) );
  INVX0 U614 ( .INP(n510), .ZN(n511) );
  NOR2X0 U615 ( .IN1(n683), .IN2(n684), .QN(n510) );
  OA21X1 U616 ( .IN1(START), .IN2(n683), .IN3(n684), .Q(n527) );
  AO21X1 U617 ( .IN1(n593), .IN2(n435), .IN3(DISPNUM1[6]), .Q(DISPNUM2[6]) );
  AO21X1 U618 ( .IN1(n594), .IN2(n595), .IN3(n596), .Q(n593) );
  OA21X1 U619 ( .IN1(n597), .IN2(n598), .IN3(DISPNUM1[4]), .Q(DISPNUM2[5]) );
  NOR2X0 U620 ( .IN1(n599), .IN2(n600), .QN(n597) );
  OA21X1 U621 ( .IN1(n601), .IN2(n602), .IN3(DISPNUM1[4]), .Q(DISPNUM2[4]) );
  OA21X1 U622 ( .IN1(n603), .IN2(n604), .IN3(DISPNUM1[4]), .Q(DISPNUM2[3]) );
  NOR2X0 U623 ( .IN1(n605), .IN2(n606), .QN(DISPNUM2[2]) );
  OA21X1 U624 ( .IN1(n599), .IN2(n602), .IN3(n603), .Q(n605) );
  INVX0 U625 ( .INP(n607), .ZN(n603) );
  NOR2X0 U626 ( .IN1(n608), .IN2(n594), .QN(n599) );
  OA21X1 U627 ( .IN1(n608), .IN2(n596), .IN3(DISPNUM1[4]), .Q(DISPNUM2[1]) );
  AO21X1 U628 ( .IN1(n595), .IN2(n600), .IN3(n598), .Q(n596) );
  AO21X1 U629 ( .IN1(n595), .IN2(n607), .IN3(n609), .Q(n598) );
  AO21X1 U630 ( .IN1(n610), .IN2(n611), .IN3(n612), .Q(n607) );
  OA221X1 U631 ( .IN1(n613), .IN2(n609), .IN3(n614), .IN4(n615), .IN5(
        DISPNUM1[4]), .Q(DISPNUM2[0]) );
  OA21X1 U632 ( .IN1(n616), .IN2(n617), .IN3(n618), .Q(n615) );
  NOR3X0 U633 ( .IN1(n444), .IN2(n619), .IN3(n611), .QN(n616) );
  OA21X1 U634 ( .IN1(n620), .IN2(n612), .IN3(n595), .Q(n613) );
  INVX0 U635 ( .INP(n604), .ZN(n595) );
  NAND2X0 U636 ( .IN1(n621), .IN2(n622), .QN(n604) );
  NAND3X0 U637 ( .IN1(n611), .IN2(n619), .IN3(n610), .QN(n622) );
  NAND2X0 U638 ( .IN1(n621), .IN2(n623), .QN(n612) );
  NAND3X0 U639 ( .IN1(n619), .IN2(n624), .IN3(n611), .QN(n623) );
  OA21X1 U640 ( .IN1(n608), .IN2(n602), .IN3(n625), .Q(n620) );
  INVX0 U641 ( .INP(n600), .ZN(n625) );
  AO21X1 U642 ( .IN1(n610), .IN2(n619), .IN3(n600), .Q(n602) );
  AO21X1 U643 ( .IN1(n611), .IN2(n624), .IN3(n609), .Q(n600) );
  INVX0 U644 ( .INP(n621), .ZN(n609) );
  AO21X1 U645 ( .IN1(n617), .IN2(n618), .IN3(n614), .Q(n621) );
  NAND2X0 U646 ( .IN1(N832), .IN2(n626), .QN(n618) );
  INVX0 U647 ( .INP(n627), .ZN(n617) );
  MUX21X1 U648 ( .IN1(N831), .IN2(NUM[3]), .S(n628), .Q(n627) );
  MUX21X1 U649 ( .IN1(NUM[2]), .IN2(N830), .S(n626), .Q(n611) );
  NOR2X0 U650 ( .IN1(n594), .IN2(n610), .QN(n608) );
  NOR2X0 U651 ( .IN1(n614), .IN2(n384), .QN(n610) );
  INVX0 U652 ( .INP(n601), .ZN(n594) );
  NAND2X0 U653 ( .IN1(n619), .IN2(n624), .QN(n601) );
  INVX0 U654 ( .INP(n614), .ZN(n624) );
  NOR2X0 U655 ( .IN1(sub_199_carry[5]), .IN2(n628), .QN(n614) );
  XOR2X1 U656 ( .IN1(n628), .IN2(n389), .Q(n619) );
  INVX0 U657 ( .INP(n626), .ZN(n628) );
  NOR2X0 U658 ( .IN1(n626), .IN2(n606), .QN(DISPNUM1[5]) );
  AO21X1 U659 ( .IN1(n629), .IN2(NUM[3]), .IN3(NUM[4]), .Q(n626) );
  NAND2X0 U660 ( .IN1(n389), .IN2(n385), .QN(n629) );
  AO21X1 U661 ( .IN1(n630), .IN2(n435), .IN3(DISPNUM1[6]), .Q(DISPMAX3[6]) );
  AO21X1 U662 ( .IN1(n631), .IN2(n632), .IN3(n633), .Q(n630) );
  OA21X1 U663 ( .IN1(n634), .IN2(n635), .IN3(DISPNUM1[4]), .Q(DISPMAX3[5]) );
  NOR2X0 U664 ( .IN1(n636), .IN2(n637), .QN(n634) );
  OA21X1 U665 ( .IN1(n638), .IN2(n639), .IN3(DISPNUM1[4]), .Q(DISPMAX3[4]) );
  OA21X1 U666 ( .IN1(n640), .IN2(n641), .IN3(DISPNUM1[4]), .Q(DISPMAX3[3]) );
  NOR2X0 U667 ( .IN1(n642), .IN2(n606), .QN(DISPMAX3[2]) );
  OA21X1 U668 ( .IN1(n636), .IN2(n639), .IN3(n640), .Q(n642) );
  INVX0 U669 ( .INP(n643), .ZN(n640) );
  NOR2X0 U670 ( .IN1(n644), .IN2(n631), .QN(n636) );
  OA21X1 U671 ( .IN1(n644), .IN2(n633), .IN3(DISPNUM1[4]), .Q(DISPMAX3[1]) );
  AO21X1 U672 ( .IN1(n632), .IN2(n637), .IN3(n635), .Q(n633) );
  AO21X1 U673 ( .IN1(n632), .IN2(n643), .IN3(n645), .Q(n635) );
  AO21X1 U674 ( .IN1(n646), .IN2(n647), .IN3(n648), .Q(n643) );
  OA221X1 U675 ( .IN1(n649), .IN2(n645), .IN3(n650), .IN4(n651), .IN5(
        DISPNUM1[4]), .Q(DISPMAX3[0]) );
  OA21X1 U676 ( .IN1(n652), .IN2(n653), .IN3(n654), .Q(n650) );
  NOR3X0 U677 ( .IN1(n443), .IN2(n647), .IN3(n655), .QN(n652) );
  OA21X1 U678 ( .IN1(n656), .IN2(n648), .IN3(n632), .Q(n649) );
  INVX0 U679 ( .INP(n641), .ZN(n632) );
  NAND2X0 U680 ( .IN1(n657), .IN2(n658), .QN(n641) );
  NAND3X0 U681 ( .IN1(n655), .IN2(n647), .IN3(n646), .QN(n658) );
  NAND2X0 U682 ( .IN1(n657), .IN2(n659), .QN(n648) );
  NAND3X0 U683 ( .IN1(n655), .IN2(n647), .IN3(n660), .QN(n659) );
  OA21X1 U684 ( .IN1(n644), .IN2(n639), .IN3(n661), .Q(n656) );
  INVX0 U685 ( .INP(n637), .ZN(n661) );
  AO21X1 U686 ( .IN1(n646), .IN2(n655), .IN3(n637), .Q(n639) );
  AO21X1 U687 ( .IN1(n660), .IN2(n647), .IN3(n645), .Q(n637) );
  INVX0 U688 ( .INP(n657), .ZN(n645) );
  AO21X1 U689 ( .IN1(n653), .IN2(n654), .IN3(n651), .Q(n657) );
  MUX21X1 U690 ( .IN1(n662), .IN2(n663), .S(n664), .Q(n654) );
  NOR4X0 U691 ( .IN1(r471_DIFF_7_), .IN2(r471_DIFF_6_), .IN3(r471_DIFF_5_), 
        .IN4(r471_DIFF_4_), .QN(n663) );
  NOR4X0 U692 ( .IN1(N627), .IN2(N626), .IN3(N625), .IN4(N624), .QN(n662) );
  INVX0 U693 ( .INP(n665), .ZN(n653) );
  MUX21X1 U694 ( .IN1(N623), .IN2(r471_DIFF_3_), .S(n664), .Q(n665) );
  MUX21X1 U695 ( .IN1(N622), .IN2(r471_DIFF_2_), .S(n664), .Q(n647) );
  NOR2X0 U696 ( .IN1(n631), .IN2(n646), .QN(n644) );
  NOR2X0 U697 ( .IN1(n651), .IN2(n698), .QN(n646) );
  INVX0 U698 ( .INP(n638), .ZN(n631) );
  NAND2X0 U699 ( .IN1(n660), .IN2(n655), .QN(n638) );
  MUX21X1 U700 ( .IN1(N621), .IN2(r471_DIFF_1_), .S(n664), .Q(n655) );
  INVX0 U701 ( .INP(n651), .ZN(n660) );
  MUX21X1 U702 ( .IN1(N628), .IN2(r471_DIFF_8_), .S(n664), .Q(n651) );
  AO21X1 U703 ( .IN1(n664), .IN2(n435), .IN3(DISPNUM1[6]), .Q(DISPMAX2[6]) );
  NOR2X0 U704 ( .IN1(n685), .IN2(n675), .QN(DISPNUM1[6]) );
  AO21X1 U705 ( .IN1(n664), .IN2(DISPNUM1[4]), .IN3(DISPMAX2[1]), .Q(
        DISPMAX2[5]) );
  OA21X1 U706 ( .IN1(n666), .IN2(n667), .IN3(DISPNUM1[4]), .Q(DISPMAX2[4]) );
  NOR3X0 U707 ( .IN1(n606), .IN2(n666), .IN3(n668), .QN(DISPMAX2[2]) );
  NOR2X0 U708 ( .IN1(n697), .IN2(n664), .QN(n668) );
  INVX0 U709 ( .INP(n667), .ZN(n664) );
  NAND3X0 U710 ( .IN1(n669), .IN2(n467), .IN3(N515), .QN(n667) );
  NAND2X0 U711 ( .IN1(n670), .IN2(n671), .QN(n669) );
  AO21X1 U712 ( .IN1(DISPNUM1[4]), .IN2(n666), .IN3(DISPMAX2[1]), .Q(
        DISPMAX2[0]) );
  AND2X1 U713 ( .IN1(DISPNUM1[4]), .IN2(n697), .Q(DISPMAX2[1]) );
  AO21X1 U714 ( .IN1(n462), .IN2(n672), .IN3(N519), .Q(n697) );
  AO21X1 U715 ( .IN1(n472), .IN2(n670), .IN3(n671), .Q(n672) );
  INVX0 U716 ( .INP(N514), .ZN(n671) );
  INVX0 U717 ( .INP(n696), .ZN(n666) );
  NAND4X0 U718 ( .IN1(N514), .IN2(n476), .IN3(N515), .IN4(n673), .QN(n696) );
  NOR2X0 U719 ( .IN1(n670), .IN2(n472), .QN(n673) );
  MUX21X1 U720 ( .IN1(N485), .IN2(n437), .S(n703), .Q(N512) );
  INVX0 U721 ( .INP(N513), .ZN(n670) );
  MUX21X1 U722 ( .IN1(N486), .IN2(n439), .S(n703), .Q(N513) );
  MUX21X1 U723 ( .IN1(N488), .IN2(n436), .S(n703), .Q(N515) );
  MUX21X1 U724 ( .IN1(N487), .IN2(n442), .S(n703), .Q(N514) );
  INVX0 U725 ( .INP(n606), .ZN(DISPNUM1[4]) );
  NAND2X0 U726 ( .IN1(n685), .IN2(n435), .QN(n606) );
endmodule

