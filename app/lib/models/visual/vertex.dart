import 'package:flutter/material.dart';

class Vertex {
  Offset position;
  double angle;

  Vertex({required this.position, required this.angle});

  Vertex copy() {
    return Vertex(position: position, angle: angle);
  }
}
