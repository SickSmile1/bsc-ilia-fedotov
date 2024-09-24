//+
Point(1) = {0, 0, 0, 1.};
//+
Point(2) = {1, 0, 0, 1.};
//+
Point(3) = {1, 4, 0, 1.};
//+
Point(4) = {0, 4, 0, 1.};
//+
Point(5) = {4, 4, 0, 1.};
//+
Point(6) = {4, 3, 0, 1.};
//+
Point(7) = {1, 3, 0, 1.};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 7};
//+
Line(4) = {7, 6};
//+
Line(5) = {6, 5};
//+
Line(6) = {4, 5};
//+
Recursive Delete {
  Point{3}; 
}
//+
Curve Loop(1) = {6, -5, -4, -3, -2, -1};
//+
Plane Surface(1) = {1};
//+
Physical Curve("wall1", 7) = {1, 6};
//+
Physical Curve("wall2", 8) = {3, 4};
//+
Physical Curve("in", 9) = {2};
//+
Physical Curve("out", 10) = {5};
//+
Physical Surface("fluid", 11) = {1};
//+
Physical Surface("fluid", 11) += {1};
