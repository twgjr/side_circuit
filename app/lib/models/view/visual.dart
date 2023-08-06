import 'dart:math';

class Visual<T> {
  List<Point<double>> points = [Point<double>(0, 0)];

  Visual();

  Point<double> get startPosition => points[0];

  set startPosition(Point<double> value) {
    points[0] = value;
  }
}
