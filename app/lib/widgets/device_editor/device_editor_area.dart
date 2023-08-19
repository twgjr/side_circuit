import 'package:flutter/material.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/device/device_view.dart';
import 'package:app/widgets/terminal/terminal_view.dart';

class DeviceEditorArea extends StatelessWidget {
  final Device device;

  DeviceEditorArea({super.key, required this.device});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return Stack(
          children: [
            DeviceView(
              constraints: constraints,
              device: device,
            ),
            for (Terminal terminal in device.terminals)
              TerminalView(
                offset: Offset.zero,
                terminal: terminal,
                editable: true,
              ),
          ],
        );
      },
    );
  }
}
