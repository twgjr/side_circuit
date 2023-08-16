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
  Offset _currentPoint = Offset(0, 0);
  var strokeColor;
  var strokeWidth;
  var fillColor;

  Shape({
    this.strokeColor = Colors.black,
    this.strokeWidth = 2.0,
    this.fillColor = Colors.transparent,
  });

  void angleLine(double angle, double length) {
    angle = angle * math.pi / 180;
    final dx = math.cos(angle) * length;
    final dy = math.sin(angle) * length;
    _currentPoint = Offset(_currentPoint.dx + dx, _currentPoint.dy + dy);
    _path.lineTo(_currentPoint.dx, _currentPoint.dy);
  }

  void lineTo(double x, double y) {
    _currentPoint = Offset(x, y);
    _path.lineTo(x, y);
  }

  void addRect(double width, double height) {
    _path.addRect(Rect.fromLTWH(0, 0, width, height));
  }

  void addCircle(double diameter) {
    _path.addOval(Rect.fromCircle(center: Offset(0, 0), radius: diameter / 2));
  }

  void end_path() {
    final Rect bounds = _path.getBounds();
    _path = _path.shift(Offset(-bounds.left, -bounds.top));
  }

  Path getPath() => _path;

  void reset() {
    _path.reset();
    _currentPoint = Offset(0, 0);
  }

  void pathFrom(Path path) {
    _path = Path.from(path);
  }

  copyWith() {
    final shape = Shape();
    shape._path = Path.from(_path);
    shape._currentPoint = Offset(_currentPoint.dx, _currentPoint.dy);
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
