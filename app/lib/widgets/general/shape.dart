import 'package:flutter/material.dart';
import 'dart:math' as math;

class ShapeWidget extends StatelessWidget {
  final Shape shape;
  ShapeWidget({super.key, required this.shape});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: shape,
      child: Container(
        width: shape.bounds().width,
        height: shape.bounds().height,
      ),
    );
  }
}

class Shape extends CustomPainter {
  Path _path = Path();
  Offset currentPoint = Offset(0, 0);
  var strokeColor;
  var strokeWidth;
  var fillColor;

  Shape({
    this.strokeColor = Colors.black,
    this.strokeWidth = 2.0,
    this.fillColor = Colors.transparent,
  });

  Shape.wireSegment({
    required Offset end,
    this.strokeColor = Colors.black,
    this.strokeWidth = 2.0,
    this.fillColor = Colors.transparent,
  }) {
    reset();
    lineTo(end);
  }

  Offset center() => _path.getBounds().center;

  Rect bounds() => _path.getBounds();

  double width() => _path.getBounds().width;
  double height() => _path.getBounds().height;

  void angleLine(double angle, double length) {
    angle = angle * math.pi / 180;
    final dx = math.cos(angle) * length;
    final dy = math.sin(angle) * length;
    currentPoint = Offset(currentPoint.dx + dx, currentPoint.dy + dy);
    _path.lineTo(currentPoint.dx, currentPoint.dy);
  }

  void lineTo(Offset offset) {
    currentPoint = offset;
    _path.lineTo(offset.dx, offset.dy);
  }

  void addRect(double width, double height) {
    _path.addRect(Rect.fromLTWH(0, 0, width, height));
  }

  void addCircle(double diameter) {
    _path.addOval(Rect.fromCircle(center: Offset(0, 0), radius: diameter / 2));
  }

  Offset end_path() {
    final Rect bounds = _path.getBounds();
    Offset offset = Offset(-bounds.left, -bounds.top);
    _path = _path.shift(offset);
    return offset;
  }

  void reset() {
    _path.reset();
    currentPoint = Offset(0, 0);
  }

  void pathFrom(Path path) {
    _path = Path.from(path);
  }

  copyWith() {
    final shape = Shape();
    shape._path = Path.from(_path);
    shape.currentPoint = Offset(currentPoint.dx, currentPoint.dy);
    shape.strokeColor = strokeColor;
    shape.strokeWidth = strokeWidth;
    shape.fillColor = fillColor;
    return shape;
  }

  @override
  void paint(Canvas canvas, Size size) {
    final strokePaint = Paint()
      ..color = strokeColor
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..color = fillColor
      ..style = PaintingStyle.fill;

    canvas.drawPath(_path, strokePaint);
    canvas.drawPath(_path, fillPaint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
