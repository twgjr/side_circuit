import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/widgets/device/add_new_device.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/mode_provider.dart';

class DiagramControls extends ConsumerWidget {
  DiagramControls();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final circuitRead = ref.read(circuitProvider.notifier);
    final modeStateRead = ref.read(modeStateProvider.notifier);
    final modeStateWatch = ref.watch(modeStateProvider);
    return Row(
      children: [
        IconButton(
          icon: Icon(Icons.linear_scale),
          color: (modeStateWatch.addWire) ? Colors.red : Colors.black,
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
        // icon button for toggling the theme mode
        IconButton(
          icon: (modeStateWatch.activeTheme == ThemeMode.light)
              ? Icon(Icons.brightness_4)
              : Icon(Icons.brightness_4_outlined),
          tooltip: "toggle theme",
          onPressed: () {
            modeStateRead.invertModeState(ModeStates.theme);
          },
        ),
      ],
    );
  }
}
