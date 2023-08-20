import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

import 'package:app/models/visual/wire_segment.dart';

class WireSegmentView extends StatelessWidget {
  final WireSegment wireSegment;

  WireSegmentView({super.key, required this.wireSegment});

  @override
  Widget build(BuildContext context) {
    Offset start = wireSegment.start();
    Offset end = wireSegment.end();
    return Positioned(
      left: start.dx,
      top: start.dy,
      child: ShapeWidget(
        shape: Shape.wireSegment(end: end - start),
      ),
    );
  }
}
