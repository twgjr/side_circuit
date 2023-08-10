import 'dart:math';

import 'package:flutter/material.dart';

class Visual<T> {
  Point<double> position = Point<double>(0, 0);
  Path shape = Path();

  Visual() {
    shape.addRect(Rect.fromLTWH(0, 0, 100, 100));
  }

  Visual copy() {
    final visual = Visual();
    visual.position = this.position;
    visual.shape = Path.from(this.shape);
    return visual;
  }
}
