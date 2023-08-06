import 'package:flutter/material.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/widgets/device/device_editable.dart';

class DeviceEditorArea extends StatelessWidget {
  final Device device;
  DeviceEditorArea({super.key, required this.device});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        DeviceEditable(
          device: device,
          cktViewCtx: context,
        ),
      ],
    );
  }
}
