import 'package:flutter/material.dart';
import 'dart:math' as math;

class VectorPath {
  Path _path = Path();
  Offset _currentPoint = Offset(0, 0);

  void vectorLineTo(double angle, double length) {
    angle = angle * math.pi / 180;
    final dx = math.cos(angle) * length;
    final dy = math.sin(angle) * length;
    _currentPoint = Offset(_currentPoint.dx + dx, _currentPoint.dy + dy);
    _path.lineTo(_currentPoint.dx, _currentPoint.dy);
  }

  void addRect(double width, double height) {
    _path.addRect(Rect.fromLTWH(0, 0, width, height));
  }

  void end_path() {
    final Rect bounds = _path.getBounds();
    _path = _path.shift(Offset(-bounds.left, -bounds.top));
  }

  Path getPath() {
    return _path;
  }

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
