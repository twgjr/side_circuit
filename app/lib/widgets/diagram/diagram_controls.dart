import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/widgets/device/add_new_device.dart';
import 'package:app/providers/mode_provider.dart';

class DiagramControls extends ConsumerWidget {
  DiagramControls();

  IconButton _addWireIcon(WidgetRef ref) {
    final modeStateWatch = ref.watch(modeProvider);
    final modeStateRead = ref.read(modeProvider.notifier);
    if (modeStateWatch.addWire) {
      return IconButton(
        icon: Icon(Icons.linear_scale),
        color: Colors.red,
        tooltip: "add wires",
        onPressed: () {
          modeStateRead.invertModeState(ModeState.addWire);
        },
      );
    } else {
      return IconButton(
        icon: Icon(Icons.linear_scale),
        tooltip: "add wires",
        onPressed: () {
          modeStateRead.invertModeState(ModeState.addWire);
        },
      );
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Row(
      children: [
        _addWireIcon(ref),
        VerticalDivider(
          width: 1,
          thickness: 1,
          color: Colors.grey,
        ),
        AddNewDevice(),
        VerticalDivider(
          width: 1,
          thickness: 1,
          color: Colors.grey,
        ),
      ],
    );
  }
}
