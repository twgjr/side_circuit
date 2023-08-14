import 'package:flutter/material.dart';
import 'dart:math' as math;

class VectorPath {
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

  VectorPath copyWith() {
    final path = VectorPath();
    path._path = Path.from(_path);
    path._currentPoint = Offset(_currentPoint.dx, _currentPoint.dy);
    return path;
  }
}
