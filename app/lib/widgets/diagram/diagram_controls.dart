import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/widgets/device/add_new_device.dart';
import 'package:app/providers/circuit_providers.dart';

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
      ],
    );
  }
}
