import 'package:flutter/material.dart';

import 'package:app/widgets/general/shape.dart';

class Line extends StatelessWidget {
  final Offset start;
  final Offset end;

  Line({required this.start, required this.end});

  @override
  Widget build(BuildContext context) {
    return ShapeWidget(
      shape: Shape()
        ..lineTo(end.dx - start.dx, end.dy - start.dy)
        ..end_path(),
    );
  }
}
