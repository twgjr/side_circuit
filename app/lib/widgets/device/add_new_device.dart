import 'package:flutter/material.dart';
import 'package:app/models/circuit/device.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_providers.dart';

class AddNewDevice extends ConsumerWidget {
  AddNewDevice();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final onAddDevice = ref.read(circuitProvider.notifier).addDeviceOf;
    return IconButton(
      icon: Icon(Icons.add),
      tooltip: "add new device",
      onPressed: () {
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text("Add Device"),
              content: Text("Select the device to add"),
              actions: [
                TextButton(
                  child: Text("Resistor"),
                  onPressed: () {
                    onAddDevice(DeviceKind.R);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Voltage Source"),
                  onPressed: () {
                    onAddDevice(DeviceKind.V);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Current Source"),
                  onPressed: () {
                    onAddDevice(DeviceKind.I);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Capacitor"),
                  onPressed: () {
                    onAddDevice(DeviceKind.C);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Inductor"),
                  onPressed: () {
                    onAddDevice(DeviceKind.L);
                    Navigator.of(context).pop();
                  },
                ),
              ],
            );
          },
        );
      },
    );
  }
}
