import 'package:flutter/material.dart';

import 'package:app/widgets/symbol/shape_painter.dart';

class DiagramSymbol {
  Offset position = Offset(0, 0);
  double rotation = 0;
  ShapePainter shape = ShapePainter();

  DiagramSymbol();

  DiagramSymbol.segment({required Offset end}) {
    shape = ShapePainter.segment(end: end);
  }

  DiagramSymbol.vertex() {
    shape = ShapePainter.vertex();
  }

  DiagramSymbol.terminal() {
    shape = ShapePainter.terminal();
  }

  DiagramSymbol copy() {
    final symbol = DiagramSymbol();
    symbol.shape = shape.copyWith();
    symbol.position = Offset(position.dx, position.dy);
    symbol.rotation = rotation;
    return symbol;
  }

  Offset center() {
    final Rect bounds = shape.bounds();
    return Offset(
        -bounds.left - bounds.width / 2, -bounds.top - bounds.height / 2);
  }

  double width() {
    final Rect bounds = shape.bounds();
    return bounds.width;
  }

  double height() {
    final Rect bounds = shape.bounds();
    return bounds.height;
  }

  void set fillColor(Color color) {
    shape.fillColor = color;
  }
}
