import 'package:app/models/net.dart';
import 'package:app/models/wire.dart';
import 'package:app/widgets/device/diagram_device_widget.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/active_wire_provider.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/mode_provider.dart';
import 'package:app/models/terminal.dart';
import 'package:app/models/vertex.dart';
import 'package:app/models/segment.dart';
import 'package:app/models/device.dart';
import 'package:app/models/node.dart';
import 'package:app/widgets/terminal/diagram_terminal_widget.dart';
import 'package:app/widgets/wire/vertex_widget.dart';
import 'package:app/widgets/wire/segment_widget.dart';
import 'package:app/widgets/node/node_view.dart';

class Diagram extends ConsumerWidget {
  Diagram({super.key});
  final GlobalKey key = GlobalKey();

  List<Widget> _stackChildren(WidgetRef ref) {
    final circuitWatch = ref.watch(circuitProvider);
    List<Widget> children = [];
    for (Net net in circuitWatch.nets) {
      for (Wire wire in net.wires) {
        for (Vertex vertex in wire.vertices) {
          children.add(VertexWidget(vertex: vertex));
        }
        for (Segment wireSegment in wire.segments) {
          children.add(SegmentWidget(segment: wireSegment));
        }
      }
    }
    for (Device device in circuitWatch.devices) {
      children.add(DiagramDeviceWidget(device: device));
      for (Terminal terminal in device.terminals) {
        children.add(DiagramTerminalWidget(
            terminal: terminal, offset: device.position()));
      }
    }
    for (Node node in circuitWatch.nodes()) {
      children.add(NodeView(node: node));
    }
    return children;
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final modeWatch = ref.watch(modeProvider);
    final activeWireWatch = ref.watch(activeWireProvider);
    return GestureDetector(
      onTapDown: (details) {
        if (modeWatch.addWire) {
          final circuitRead = ref.read(circuitProvider.notifier);
          Wire wire = circuitRead.startWire(position: details.localPosition);
          ref.read(activeWireProvider.notifier).update(wire);
        }
      },
      child: MouseRegion(
        onHover: (event) {
          if (modeWatch.addWire) {
            final circuitRead = ref.read(circuitProvider.notifier);
            circuitRead.dragUpdateVertex(
                activeWireWatch.tail(), event.localPosition);
          }
        },
        child: Stack(children: _stackChildren(ref)),
      ),
    );
  }
}
