import 'package:flutter/material.dart';
import 'package:app/models/circuit/device.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';

class AddNewDevice extends ConsumerWidget {
  AddNewDevice();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final onAddDevice = ref.read(circuitProvider.notifier).addDeviceOf;
    return PopupMenuButton(
      icon: Icon(Icons.add),
      tooltip: "add new device",
      itemBuilder: (BuildContext context) {
        return [
          PopupMenuItem(
            value: DeviceKind.R.name,
            child: Text("Resistor"),
            onTap: () => onAddDevice(DeviceKind.R),
          ),
          PopupMenuItem(
            value: DeviceKind.V.name,
            child: Text("Independent Source"),
            onTap: () => onAddDevice(DeviceKind.V),
          ),
          PopupMenuItem(
            value: DeviceKind.BLOCK.name,
            child: Text("Block"),
            onTap: () => onAddDevice(DeviceKind.BLOCK),
          ),
        ];
      },
    );
  }
}
