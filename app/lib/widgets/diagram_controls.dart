import 'package:flutter/material.dart';
import 'package:app/models/circuit/device.dart';

class DiagramControls extends StatelessWidget {
  final void Function(DeviceKind) addItem;

  DiagramControls(this.addItem);

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        IconButton(
          icon: Icon(Icons.add),
          tooltip: "add new element",
          onPressed: () {
            showDialog(
              context: context,
              builder: (BuildContext context) {
                return AlertDialog(
                  title: Text("Add Element"),
                  content: Text("Select the element to add"),
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
        ),
      ],
    );
  }
}
