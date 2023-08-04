import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/widgets/diagram/draggable_item.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/widgets/device/device_view.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/widgets/node/node_view.dart';

class Diagram extends ConsumerWidget {
  Diagram({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final circuitWatch = ref.watch(circuitProvider);
    print('Diagram.build: circuitWatch=$circuitWatch');
    print('Diagram.build: circuitWatch.devices=${circuitWatch.devices}');
    print('Diagram.build: circuitWatch.nodes=${circuitWatch.nodes}');
    return Stack(
      children: [
        for (Device device in circuitWatch.devices)
          DraggableItem(
            visual: device.visual,
            child: DeviceView(
              device: device,
              cktViewCtx: context,
            ),
          ),
        for (Node node in circuitWatch.nodes)
          DraggableItem(
            visual: node.visual,
            child: NodeView(
              node: node,
              cktViewCtx: context,
            ),
          ),
      ],
    );
  }
}
