import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/wire.dart';
import 'package:app/models/circuit/wire_segment.dart';
import 'package:app/models/view/vertex.dart';
import 'package:app/widgets/wire/wire_segment_view.dart';
import 'package:app/widgets/wire/vertex_view.dart';

class WireView extends ConsumerWidget {
  final Wire wire;

  WireView({required this.wire, super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Stack(
      children: [
        for (WireSegment segment in wire.segments)
          WireSegmentView(wireSegment: segment),
        for (Vertex vertex in wire.vertices) VertexView(vertex: vertex),
      ],
    );
  }
}
