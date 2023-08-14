import 'package:app/models/visual/vector_path.dart';
import 'package:flutter/material.dart';

class Visual {
  Offset position = Offset(0, 0);
  VectorPath shape = VectorPath();

  Visual();

  Visual copy() {
    final visual = Visual();
    visual.shape = shape.copyWith();
    visual.position = Offset(position.dx, position.dy);
    return visual;
  }

  Offset center() {
    final Rect bounds = shape.getPath().getBounds();
    return Offset(
        -bounds.left - bounds.width / 2, -bounds.top - bounds.height / 2);
  }

  double width() {
    final Rect bounds = shape.getPath().getBounds();
    return bounds.width;
  }

  double height() {
    final Rect bounds = shape.getPath().getBounds();
    return bounds.height;
  }
}
