import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

import 'package:app/models/visual/segment.dart';

class WireSegmentView extends StatelessWidget {
  final Segment segment;

  WireSegmentView({super.key, required this.segment});

  @override
  Widget build(BuildContext context) {
    Offset start = segment.start.diagramPosition;
    Offset end = segment.end.diagramPosition;
    return Positioned(
      left: start.dx,
      top: start.dy,
      child: ShapeWidget(
        shape: Shape.wireSegment(end: end - start),
      ),
    );
  }
}
