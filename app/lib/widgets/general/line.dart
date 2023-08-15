import 'package:flutter/material.dart';

import 'package:app/widgets/general/shape.dart';

class Line extends StatelessWidget {
  final Offset start;
  final Offset end;

  Line({required this.start, required this.end});

  @override
  Widget build(BuildContext context) {
    // final distance = (end - start).distance;
    // return CustomPaint(
    //   painter: LinePainter(start, end),
    //   size: Size(distance, 1),
    // );
    return ShapeWidget(
      shape: Shape()
        ..lineTo(end.dx - start.dx, end.dy - start.dy)
        ..end_path(),
    );
  }
}

class LinePainter extends CustomPainter {
  final Offset start;
  final Offset end;

  LinePainter(this.start, this.end);

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()
      ..color = Colors.black
      ..strokeWidth = 2.0;

    canvas.drawLine(start, end, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
