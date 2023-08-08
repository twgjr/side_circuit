import 'package:flutter/material.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/device/terminal_editable.dart';

class DeviceEditable extends StatelessWidget {
  final Device deviceCopy;

  DeviceEditable({required this.deviceCopy});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        width: 100,
        height: 100,
        decoration: BoxDecoration(
          color: Colors.white,
          border: Border.all(
            color: Colors.black,
            width: 1.0,
          ),
        ),
        child: GestureDetector(
          onSecondaryTapDown: (details) {},
          child: Stack(
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Center(
                  child: Text('${deviceCopy.kind.name}${deviceCopy.id}'),
                ),
              ),
              for (Terminal terminal in deviceCopy.terminals)
                TerminalEditable(
                  device: deviceCopy,
                  terminalCopy: terminal,
                  terminalRadius: 10,
                ),
            ],
          ),
        ),
      ),
    );
  }
}
