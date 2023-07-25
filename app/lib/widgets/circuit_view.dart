import 'package:app/models/circuit/device.dart';
import 'package:flutter/material.dart';

import 'view_item.dart';

class CircuitView extends StatefulWidget {
  final void Function(Device) deleteDevice;
  final List<Device> devices;

  CircuitView({
    super.key,
    required this.deleteDevice,
    required this.devices,
  });

  @override
  _CircuitViewState createState() => _CircuitViewState();
}

class _CircuitViewState extends State<CircuitView> {
  @override
  Widget build(BuildContext context) {
    return Stack(
      children: widget.devices.map(
        (device) {
          return ViewItem(
            device: device,
            onDeleteDevice: widget.deleteDevice,
            cktViewCtx: context,
          );
        },
      ).toList(),
    );
  }
}
