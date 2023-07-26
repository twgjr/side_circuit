import 'package:flutter/material.dart';
import 'package:app/models/circuit/device.dart';

//a stateless widget class and state class for the add element button
class AddNewDevice extends StatelessWidget {
  final void Function(DeviceKind) addItem;

  AddNewDevice(this.addItem);

  @override
  Widget build(BuildContext context) {
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
                    addItem(DeviceKind.R);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Voltage Source"),
                  onPressed: () {
                    addItem(DeviceKind.V);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Current Source"),
                  onPressed: () {
                    addItem(DeviceKind.I);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Capacitor"),
                  onPressed: () {
                    addItem(DeviceKind.C);
                    Navigator.of(context).pop();
                  },
                ),
                TextButton(
                  child: Text("Inductor"),
                  onPressed: () {
                    addItem(DeviceKind.L);
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
