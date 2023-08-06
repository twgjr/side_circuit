import 'package:flutter/material.dart';
import 'dart:math';

class LinePainter extends CustomPainter {
  final Point<double> start;
  final Point<double> end;

  LinePainter(this.start, this.end);

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()
      ..color = Colors.black
      ..strokeWidth = 2.0;

    canvas.drawLine(Offset(start.x, start.y), Offset(end.x, end.y), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}

class Line extends StatelessWidget {
  final Point<double> start;
  final Point<double> end;

  Line({required this.start, required this.end});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: LinePainter(start, end),
      size: Size.infinite,
    );
  }
}
