// Gmsh project created on Thu Sep 19 11:02:14 2024
SetFactory("OpenCASCADE");
//+
Hide "*";
//+
Hide "*";
//+
Point(1) = {0, 0, 0, 1.};
//+
Point(2) = {1, 0, 0, 1.};
//+
Point(3) = {1, 1, 0, 1.};
//+
Point(4) = {0, 1, 0, 1.};
//+
Point(5) = {0, 0, 4, 1.};
//+
Point(6) = {0, 1, 4, 1.};
//+
Point(7) = {1, 1, 4, 1.};
//+
Point(8) = {1, 0, 2, 1.};
//+
Point(9) = {1, 0, 4, 1.};
//+
Line(1) = {6, 4};
//+
Line(2) = {3, 4};
//+
Line(3) = {3, 2};
//+
Line(4) = {2, 1};
//+
Line(5) = {1, 5};
//+
Line(6) = {6, 5};
//+
Line(7) = {7, 9};
//+
Line(8) = {5, 9};
//+
Line(9) = {9, 2};
//+
Line(10) = {4, 1};
//+
Line(11) = {6, 7};
//+
Line(12) = {7, 3};
//+
Physical Curve("wall3", 13) = {9, 12, 1};
//+
Physical Curve("wall1", 14) = {9};
//+
Physical Curve("wall2", 15) = {12};
//+
Physical Curve("wall3", 13) += {1};
//+
Physical Curve("wall4", 16) = {5};
//+
Physical Curve("in", 17) = {8, 7, 11, 6};
//+
Physical Curve("out", 18) = {4, 2, 3, 10};
//+
Physical Point(19) -= {8};
//+
Physical Point(19) -= {8};
//+
Physical Point(19) -= {8};
//+
Physical Point(19) -= {8};
//+
Recursive Delete {
  Point{8}; 
}
//+
Curve Loop(1) = {11, 12, 2, -1};
