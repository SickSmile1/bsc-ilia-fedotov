// Gmsh project created on Fri Sep 20 11:04:36 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, -0, 0, 1.0};
//+
Point(2) = {0, 0.5, 0, 1.0};
//+
Point(3) = {1, 1, 0, 0.1};
//+
Point(4) = {1.5, 1, 0, 0.1};
//+
Point(5) = {3.5, 1, 0, 0.1};
//+
Point(6) = {4, 1, 0, 0.1};
//+
Point(7) = {4, 1, 0, 0.1};
//+
Point(8) = {5, 1, 0, 0.1};
//+
Recursive Delete {
  Point{2}; 
}
//+
Point(9) = {0, 1, 0, 0.1};
//+
Point(10) = {2.5, 6, 0, 0.1};
//+
Point(11) = {2.5, 5.5, 0, 0.1};
//+
Recursive Delete {
  Point{10}; 
}
//+
Point(12) = {2.4, 6, 0, 0.1};
//+
Point(13) = {2.6, 6, 0, 0.1};
//+
Point(14) = {2.6, 8, 0, 0.1};
//+
Point(15) = {2.4, 8, 0, 0.1};
//+
Line(1) = {9, 3};
//+
Line(2) = {4, 5};
//+
Line(3) = {6, 8};
//+
Line(4) = {9, 1};
//+
Point(16) = {5, -0, 0, 0.1};
//+
Line(5) = {1, 16};
//+
Line(6) = {16, 8};
//+
Line(7) = {5, 11};
//+
Line(8) = {4, 11};
//+
Line(9) = {15, 12};
//+
Line(10) = {14, 13};
//+
Line(11) = {15, 14};
//+
Point(17) = {1.5, 3, -0, 0.1};
//+
Point(18) = {3.5, 3, 0, 0.1};
//+
Line(12) = {17, 3};
//+
Line(13) = {18, 6};
//+
Point(19) = {1.95, 4.5, -0, 0.1};
//+
Point(20) = {3.05, 4.5, -0, 0.1};
//+
Circle(14) = {17, 19, 12};
//+
Circle(15) = {18, 20, 13};
//+
Recursive Delete {
  Curve{14}; 
}
//+
Recursive Delete {
  Curve{15}; 
}
//+
Recursive Delete {
  Point{20}; 
}
//+
Recursive Delete {
  Point{19}; 
}
//+
Point(19) = {4.5, 4, -0, 0.1};
//+
Point(20) = {2.5, 4, -0, 0.1};
//+
Recursive Delete {
  Point{19}; 
}
//+
Circle(14) = {17, 20, 12};
//+
Circle(14) = {12, 20, 17};
//+
Circle(14) = {17, 20, 18};
//+
Recursive Delete {
  Curve{14}; 
}
//+
Point(21) = {2.5, 4.5, -0, 0.1};
//+
Recursive Delete {
  Point{20}; 
}
//+
Circle(14) = {17, 21, 12};
//+
Circle(14) = {2.5, 4.5, -0.0056974, 1.8, 0, 2*Pi};
//+
Point(23) = {2.4, 6.3, -0.0282585, 0.1};
//+
Point(24) = {2.6, 6.3, -0.0386203, 0.1};
//+
Recursive Delete {
  Curve{14}; 
}
//+
Circle(14) = {17, 21, 23};
//+
Point(25) = {2.4, 6.1, 0, 0.1};
//+
Point(26) = {2.6, 6.1, 0, 0.1};
//+
Circle(14) = {17, 21, 25};
//+
Circle(14) = {17, 21, 25};
//+
Circle(14) = {17, 21, 23};
//+
Circle(14) = {23, 23, 23};
//+
Recursive Delete {
  Point{25}; 
}
//+
Recursive Delete {
  Point{12}; Point{26}; 
}
//+
Recursive Delete {
  Point{23}; 
}
//+
Recursive Delete {
  Point{24}; 
}
//+
Point(22) = {2.4, 6.300002, 0, 0.1};
//+
Point(23) = {2.6, 6.300002, 0, 0.1};
//+
Circle(14) = {17, 21, 22};
//+
Point(24) = {1, 5, 0, 0.1};
//+
Point(25) = {4, 5, 0, 0.1};
//+
Recursive Delete {
  Point{21}; 
}
//+
Recursive Delete {
  Point{22}; 
}
//+
Recursive Delete {
  Point{23}; 
}
//+
Point(26) = {2.5, 4.5, 0, 0.1};
//+
Ellipse(14) = {17, 26, 24, 12};
//+
Recursive Delete {
  Point{26}; 
}
//+
Recursive Delete {
  Curve{14}; 
}
//+
Ellipse(14) = {17, 11, 24, 12};
//+
Point(26) = {1.9, 4.6, -0.10043, 0.1};
//+
Point(27) = {3.1, 4.6, -0.0184927, 0.1};
//+
Ellipse(14) = {17, 26, 24, 12};
//+
Recursive Delete {
  Point{26}; 
}
//+
Recursive Delete {
  Point{27}; 
}
//+
Point(26) = {0, 5, 0, 0.1};
//+
Point(27) = {5, 5, 0, 0.1};
//+
Point(28) = {2.5, 4, 0, 0.1};
//+
Ellipse(14) = {17, 28, 26, 12};
//+
Bezier(14) = {17, 12, 26};
//+
Recursive Delete {
  Curve{14}; 
}
//+
Bezier(14) = {17, 24, 12};
//+
Recursive Delete {
  Curve{14}; 
}
//+
Point(29) = {0, 5, 0, 0.1};
//+
Bezier(14) = {17, 29, 12};
//+
Bezier(15) = {18, 27, 13};
//+
Point(30) = {1.6, 3.5, -0.0883288, 0.1};
//+
Point(31) = {3.3, 3.4, 0.0772074, 0.1};
//+
Point(32) = {2.8, 5.4, -0.0203601, 0.1};
//+
Point(33) = {2.1, 5.5, -0.11076, 0.1};
//+
Point(34) = {2.8, 5.5, -0.0354324, 0.1};
//+
Point(35) = {2.9, 5.5, -0.0260033, 0.1};
//+
Recursive Delete {
  Point{32}; 
}
//+
Recursive Delete {
  Point{34}; 
}
//+
Bezier(16) = {30, 24, 33};
//+
Bezier(17) = {31, 25, 35};
//+
Line(18) = {33, 30};
//+
Line(19) = {35, 31};
//+
Curve Loop(1) = {11, 10, -15, 13, 3, -6, -5, -4, 1, -12, 14, -9};
//+
Curve Loop(2) = {7, -8, 2};
//+
Curve Loop(3) = {19, 17};
//+
Curve Loop(4) = {18, 16};
//+
Curve Loop(5) = {19, 17};
//+
Curve Loop(6) = {17, 19};
//+
Plane Surface(1) = {1, 2, 3, 4, 5, 6};
//+
Curve Loop(7) = {1, -12, 14, -9, 11, 10, -15, 13, 3, -6, -5, -4};
//+
Curve Loop(8) = {2, 7, -8};
//+
Plane Surface(2) = {7, 8};
//+
Curve Loop(9) = {10, -15, 13, 3, -6, -5, -4, 1, -12, 14, -9, 11};
//+
Point(36) = {2.1, 5.5, 0, 0.1};
//+
Delete {
  Point{33}; 
}
//+
Delete {
  Point{33}; Curve{16}; Curve{18}; 
}
//+
Delete {
  Curve{16}; Curve{18}; Point{33}; 
}
//+
Delete {
  Curve{18}; Point{33}; Curve{16}; 
}
//+
Delete {
  Curve{18}; Point{33}; Curve{16}; 
}
//+
Delete {
  Surface{1}; Curve{18}; Curve{16}; Point{36}; 
}
//+
Bezier(20) = {30, 24, 33};
//+
Line(21) = {33, 30};
//+
Curve Loop(10) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(11) = {8, -7, -2};
//+
Curve Loop(12) = {21, 20};
//+
Curve Loop(13) = {19, 17};
//+
Surface(3) = {10, 11};
//+
Curve Loop(14) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(15) = {7, -8, 2};
//+
Plane Surface(3) = {14, 15};
//+
Curve Loop(16) = {17, 19};
//+
Plane Surface(4) = {16};
//+
Curve Loop(17) = {20, 21};
//+
Plane Surface(5) = {17};
//+
Curve Loop(18) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(19) = {8, -7, -2};
//+
Plane Surface(6) = {18, 19};
//+
Curve Loop(20) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(21) = {8, -7, -2};
//+
Plane Surface(7) = {20, 21};
//+
Curve Loop(22) = {21, 20};
//+
Plane Surface(8) = {22};
//+
Curve Loop(23) = {17, 19};
//+
Plane Surface(9) = {23};
//+
Curve Loop(24) = {15, -10, -11, 9, -14, 12, -1, 4, 5, 6, -3, -13};
//+
Curve Loop(25) = {7, -8, 2};
//+
Plane Surface(10) = {24, 25};
//+
Curve Loop(26) = {21, 20};
//+
Curve Loop(27) = {8, -7, -2};
//+
Curve Loop(28) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(29) = {19, 17};
//+
Surface(11) = {26, 27, 28, 29};
//+
Curve Loop(30) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(31) = {8, -7, -2};
//+
Surface(11) = {30, 31};
//+
Curve Loop(32) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(33) = {8, -7, -2};
//+
Surface(11) = {32, 33};
//+
Curve Loop(34) = {20, 21};
//+
Curve Loop(35) = {20, 21};
//+
Surface(11) = {34, 35};
//+
Curve Loop(36) = {19, 17};
//+
Surface(11) = {36};
//+
Curve Loop(38) = {19, 17};
//+
Curve Loop(39) = {19, 17};
//+
Surface(12) = {38, 39};
//+
Recursive Delete {
  Surface{11}; 
}
//+
Curve Loop(40) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(41) = {20, 21};
//+
Curve Loop(42) = {17, 19};
//+
Curve Loop(43) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(44) = {8, -7, -2};
//+
Curve Loop(45) = {14, -9, 11, 10, -15, 13, 3, -6, -5, -4, 1, -12};
//+
Curve Loop(46) = {20, 21};
//+
Curve Loop(47) = {8, -7, -2};
//+
Curve Loop(48) = {19, 17};
