import 'package:flutter/material.dart';

import 'package:app/models/visual/wire.dart';
import 'package:app/models/visual/vertex.dart';
import 'package:app/models/visual/wire_segment.dart';
import 'package:app/widgets/wire/wire_segment_view.dart';
import 'package:app/widgets/wire/vertex_view.dart';

class WireView extends StatelessWidget {
  final Wire wire;

  WireView({required this.wire, super.key});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        for (WireSegment wireSegment in wire.segments)
          WireSegmentView(wireSegment: wireSegment),
        for (Vertex vertex in wire.vertices) VertexView(vertex: vertex),
      ],
    );
  }
}
