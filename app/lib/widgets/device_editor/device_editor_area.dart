import 'package:flutter/material.dart';

import 'package:app/models/device.dart';
import 'package:app/models/terminal.dart';
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
              editorConstraints: constraints,
              device: device,
            ),
            for (Terminal terminal in device.terminals)
              TerminalView(
                editorConstraints: constraints,
                terminal: terminal,
              ),
          ],
        );
      },
    );
  }
}
