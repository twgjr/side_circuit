import 'package:flutter/material.dart';

import 'package:app/widgets/symbol/shape_painter.dart';

class ShapeWidget extends StatelessWidget {
  final ShapePainter shape;
  ShapeWidget({super.key, required this.shape});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: shape,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(
            color: Colors.red,
            width: 1,
          ),
        ),
        width: shape.bounds().width,
        height: shape.bounds().height,
      ),
    );
  }
}
