import 'package:app/widgets/terminal/editor_terminal_widget.dart';
import 'package:flutter/material.dart';

import 'package:app/models/device.dart';
import 'package:app/models/terminal.dart';
import 'package:app/widgets/device/editor_device_widget.dart';

class DeviceEditorArea extends StatelessWidget {
  final Device device;

  DeviceEditorArea({super.key, required this.device});

  List<Widget> _stackChildren(BoxConstraints constraints) {
    final stackCenter = Offset(
      constraints.maxWidth / 2,
      constraints.maxHeight / 2,
    );
    List<Widget> children = [];
    children.add(EditorDeviceWidget(
      offset: stackCenter,
      device: device,
    ));
    for (Terminal terminal in device.terminals) {
      children.add(EditorTerminalWidget(
        offset: stackCenter,
        terminal: terminal,
      ));
    }
    return children;
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return Stack(
          children: _stackChildren(constraints),
        );
      },
    );
  }
}
