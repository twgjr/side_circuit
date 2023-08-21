import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/active_vertex_provider.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/mode_provider.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/vertex.dart';
import 'package:app/models/visual/segment.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/visual/node.dart';
import 'package:app/widgets/terminal/terminal_view.dart';
import 'package:app/widgets/wire/vertex_view.dart';
import 'package:app/widgets/wire/segment_view.dart';
import 'package:app/widgets/device/device_view.dart';
import 'package:app/widgets/node/node_view.dart';

class Diagram extends ConsumerWidget {
  Diagram({super.key});
  final GlobalKey key = GlobalKey();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final circuitWatch = ref.watch(circuitProvider);
    final modeWatch = ref.watch(modeProvider);
    final activeVertexWatch = ref.watch(activeVertexProvider);
    return GestureDetector(
      onTapDown: (details) {
        if (modeWatch.addWire) {
          final circuitRead = ref.read(circuitProvider.notifier);
          Vertex last = circuitRead.startWire(position: details.localPosition);
          ref.read(activeVertexProvider.notifier).set(last);
        }
      },
      child: MouseRegion(
        onHover: (event) {
          if (modeWatch.addWire) {
            final circuitRead = ref.read(circuitProvider.notifier);
            circuitRead.dragUpdateVertex(
                activeVertexWatch, event.localPosition);
          }
        },
        child: Stack(
          children: [
            for (Segment wireSegment in circuitWatch.segments())
              WireSegmentView(segment: wireSegment),
            for (Vertex vertex in circuitWatch.vertices())
              VertexView(vertex: vertex),
            for (Device device in circuitWatch.devices)
              DeviceView(device: device),
            for (Terminal terminal in circuitWatch.terminals())
              TerminalView(terminal: terminal),
            for (Node node in circuitWatch.nodes()) NodeView(node: node),
          ],
        ),
      ),
    );
  }
}
