import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/widgets/device/add_new_device.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/mode_provider.dart';

class DiagramControls extends ConsumerWidget {
  DiagramControls();

  Color _addWireIconColor(WidgetRef ref) {
    final modeStateWatch = ref.watch(modeStateProvider);
    if (modeStateWatch.addWire) {
      return Colors.red;
    } else {
      return Colors.black;
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final circuitRead = ref.read(circuitProvider.notifier);
    final modeStateRead = ref.read(modeStateProvider.notifier);
    return Row(
      children: [
        IconButton(
          icon: Icon(Icons.linear_scale),
          color: _addWireIconColor(ref),
          tooltip: "add wires",
          onPressed: () {
            modeStateRead.invertModeState(ModeStates.addWire);
          },
        ),
        VerticalDivider(
          width: 1,
          thickness: 1,
          color: Colors.grey,
        ),
        AddNewDevice(),
        IconButton(
          icon: Icon(Icons.add_circle_outline),
          tooltip: "add new node",
          onPressed: () {
            circuitRead.newNode();
          },
        ),
        VerticalDivider(
          width: 1,
          thickness: 1,
          color: Colors.grey,
        ),
      ],
    );
  }
}
