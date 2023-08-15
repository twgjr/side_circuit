import 'package:flutter/material.dart';

import 'package:app/widgets/general/shape.dart';

class Symbol {
  Offset position = Offset(0, 0);
  double angle = 0;
  Shape shape = Shape();

  Symbol();

  Symbol copy() {
    final symbol = Symbol();
    symbol.shape = shape.copyWith();
    symbol.position = Offset(position.dx, position.dy);
    symbol.angle = angle;
    return symbol;
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
