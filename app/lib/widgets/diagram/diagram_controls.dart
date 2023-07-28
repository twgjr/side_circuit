import 'package:flutter/material.dart';
import 'package:app/widgets/device/add_new_device.dart';
import 'package:app/models/circuit/device.dart';

class DiagramControls extends StatelessWidget {
  final void Function(DeviceKind) onAddDevice;
  final void Function() onAddNode;

  DiagramControls({
    required this.onAddDevice,
    required this.onAddNode,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        AddNewDevice(onAddDevice),
        // icon button that adds a new node
        IconButton(
          icon: Icon(Icons.add_circle_outline),
          tooltip: "add new node",
          onPressed: onAddNode,
        ),
      ],
    );
  }
}
