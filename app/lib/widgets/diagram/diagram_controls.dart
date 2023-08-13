import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/widgets/device/add_new_device.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/gesture_state_provider.dart';

class DiagramControls extends ConsumerWidget {
  DiagramControls();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final circuit = ref.read(circuitProvider.notifier);
    return Row(
      children: [
        AddNewDevice(),
        IconButton(
          icon: Icon(Icons.add_circle_outline),
          tooltip: "add new node",
          onPressed: () {
            circuit.newNode();
          },
        ),
        //divider
        VerticalDivider(
          width: 1,
          thickness: 1,
          color: Colors.grey,
        ),
        // icon button to trigger add wire mode
        IconButton(
          icon: Icon(Icons.linear_scale),
          tooltip: "add wires",
          onPressed: () {
            ref
                .read(gestureStateProvider.notifier)
                .update(GestureState(addWire: true));
          },
        ),
      ],
    );
  }
}
