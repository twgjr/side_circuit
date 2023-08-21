import 'package:flutter/material.dart';
import 'dart:math' as math;

class ShapeWidget extends StatelessWidget {
  final Shape shape;
  ShapeWidget({super.key, required this.shape});

  @override
  Widget build(BuildContext context) {
    final path = shape.getPath();
    return CustomPaint(
      painter: ShapePainter(shape: shape),
      child: Container(
        width: path.getBounds().width,
        height: path.getBounds().height,
      ),
    );
  }
}

class Shape {
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

  void addRect(double width, double height, bool centered) {
    _path.addRect(Rect.fromLTWH(0, 0, width, height));
    if (centered) {
      _path = _path.shift(Offset(-width / 2, -height / 2));
    }
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

  Path getPath() => _path;

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
}

class ShapePainter extends CustomPainter {
  final Shape shape;

  ShapePainter({required this.shape});

  @override
  void paint(Canvas canvas, Size size) {
    final strokePaint = Paint()
      ..color = shape.strokeColor
      ..strokeWidth = shape.strokeWidth
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..color = shape.fillColor
      ..style = PaintingStyle.fill;

    canvas.drawPath(shape.getPath(), strokePaint);
    canvas.drawPath(shape.getPath(), fillPaint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
