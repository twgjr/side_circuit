import 'package:flutter/material.dart';
import 'dart:math' as math;

class ShapeWidget extends StatefulWidget {
  final Shape shape;
  ShapeWidget({super.key, required this.shape});

  @override
  State<StatefulWidget> createState() => ShapeWidgetState();
}

class ShapeWidgetState extends State<ShapeWidget> {
  @override
  Widget build(BuildContext context) {
    final path = widget.shape.getPath();
    return CustomPaint(
      painter: ShapePainter(path),
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
    return shape;
  }
}

class ShapePainter extends CustomPainter {
  final Path shape;

  ShapePainter(this.shape);

  @override
  void paint(Canvas canvas, Size size) {
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
