import 'dart:math';
import 'package:app/models/view/vector_path.dart';

class Visual<T> {
  Point<double> position = Point<double>(0, 0);
  VectorPath shape = VectorPath();

  Visual();

  Visual copy() {
    final visual = Visual();
    visual.shape = shape.copyWith();
    visual.position = Point<double>(this.position.x, this.position.y);
    return visual;
  }
}
