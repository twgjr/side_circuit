import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

import 'package:app/models/visual/segment.dart';

class WireSegmentView extends StatelessWidget {
  final Segment wireSegment;

  WireSegmentView({super.key, required this.wireSegment});

  @override
  Widget build(BuildContext context) {
    Offset start = wireSegment.start.diagramPosition;
    Offset end = wireSegment.end.diagramPosition;
    return Positioned(
      left: start.dx,
      top: start.dy,
      child: ShapeWidget(
        shape: Shape.wireSegment(end: end - start),
      ),
    );
  }
}
