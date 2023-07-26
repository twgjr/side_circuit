import 'package:app/models/circuit/device.dart';
import 'package:flutter/material.dart';

import 'item_view.dart';

class CircuitView extends StatefulWidget {
  final void Function(Device) onDeleteDevice;
  final List<Device> devices;

  CircuitView({
    super.key,
    required this.onDeleteDevice,
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
          return ItemView(
            device: device,
            onDeleteDevice: widget.onDeleteDevice,
            cktViewCtx: context,
          );
        },
      ).toList(),
    );
  }
}
