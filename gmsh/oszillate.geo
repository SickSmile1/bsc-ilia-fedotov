// Gmsh project created on Thu Sep 19 13:07:51 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {-1.2, 0, -0, 1.0};
//+
Point(3) = {0.5, 0, -0, 0.1};
//+
Point(4) = {0.5, 0.5, 0, 0.1};
//+
Point(5) = {-0.9, 0.3, 0.2, 0.1};
//+
Recursive Delete {
  Point{1}; 
}
//+
Recursive Delete {
  Point{5}; 
}
//+
Recursive Delete {
  Point{4}; 
}
//+
Recursive Delete {
  Point{3}; 
}
//+
Recursive Delete {
  Point{2}; 
}
//+
Point(1) = {0, 0, 0, 0.1};
//+
Point(2) = {1, 0, 0, 0.1};
//+
Point(3) = {0.5, 1, 0, 0.1};
//+
Bezier(1) = {1, 1, 3, 2};
//+
Recursive Delete {
  Point{2}; Point{1}; Curve{1}; 
}
//+
Recursive Delete {
  Point{3}; 
}
//+
Circle(1) = {1, 1, 0, 1, 0, 2*Pi};
//+
Recursive Delete {
  Point{1}; Curve{1}; 
}
//+
Circle(1) = {2, 2, 0, 1, 0, 2*Pi};
//+
Point(2) = {0, 0, 0, 0.1};
//+
Point(3) = {4, 0, 0, 0.1};
//+
Point(4) = {1, 0, 0, 0.1};
//+
Point(5) = {3, 0, 0, 0.1};
//+
Point(6) = {2.1, 3, 0, 0.1};
//+
Point(7) = {0.9, 3, 0, 0.1};
//+
Point(8) = {1.9, 3, 0, 0.1};
//+
Point(9) = {1.2, 0, 0, 0.1};
//+
Point(10) = {2.8, 0, 0, 0.1};
//+
Point(11) = {2.1, 4, 0, 0.1};
//+
Point(12) = {1.9, 4, 0, 0.1};
//+
Circle(2) = {2, 2, 0, .9, 0, 2*Pi};
//+
Point(14) = {0, -0.2, 0, 0.1};
//+
Point(15) = {1, -0.2, 0, 0.1};
//+
Point(16) = {1.2, -0.2, 0, 0.1};
//+
Point(17) = {2.8, -0.2, 0, 0.1};
//+
Point(18) = {3, -0.2, 0, 0.1};
//+
Point(19) = {4, -0.2, 0, 0.1};
//+
Recursive Delete {
  Point{7}; 
}
//+
Line(3) = {2, 4};
//+
Line(4) = {9, 10};
//+
Line(5) = {5, 3};
//+
Line(6) = {19, 14};
//+
Line(7) = {4, 8};
//+
Line(8) = {6, 5};
//+
Line(9) = {12, 8};
//+
Line(10) = {11, 6};
//+
Recursive Delete {
  Curve{7}; 
}
//+
Recursive Delete {
  Curve{8}; 
}
//+
Point(20) = {1.9, 2.9, -0, 0.1};
//+
Point(21) = {2.1, 2.8, -0, 0.1};
//+
Point(22) = {2.1, 2.9, -0, 0.1};
//+
Line(11) = {20, 4};
//+
Line(12) = {22, 5};
//+
Point(23) = {2, 2.9, -0, 0.1};
//+
Line(13) = {23, 9};
//+
Line(14) = {23, 10};
//+
Line(15) = {2, 14};
//+
Line(16) = {3, 19};
