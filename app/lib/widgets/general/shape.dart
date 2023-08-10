import 'package:flutter/material.dart';

class Shape extends StatelessWidget {
  final Path shape;

  Shape({required this.shape});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: ShapePainter(shape),
      child: Container(
        width: shape.getBounds().width,
        height: shape.getBounds().height,
      ),
    );
  }
}

class ShapePainter extends CustomPainter {
  final Path shape;

  ShapePainter(this.shape);

  @override
  void paint(Canvas canvas, Size size) {
    print(size);
    final paint = Paint()
      ..color = Colors.black
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    canvas.drawPath(shape, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
