import 'package:flutter/material.dart';

import 'package:app/models/circuit/device.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class DeviceEditorTopBar extends ConsumerWidget {
  final Device deviceCopy;
  final void Function(double) onXChanged;
  final void Function(double) onYChanged;
  final void Function(WidgetRef, bool) onEditComplete;

  DeviceEditorTopBar(
      {required this.deviceCopy,
      required this.onXChanged,
      required this.onYChanged,
      required this.onEditComplete});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onPanUpdate: (details) {
        onXChanged(details.delta.dx);
        onYChanged(details.delta.dy);
      },
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).primaryColor,
          border: Border(
            bottom: BorderSide(
              color: Theme.of(context).primaryColorDark,
              width: 1.0,
            ),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text(
                  'Device Editor',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: ElevatedButton(
                child: Icon(Icons.check_box),
                onPressed: () {
                  onEditComplete(ref, true);
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: ElevatedButton(
                onPressed: () {
                  onEditComplete(ref, false);
                },
                child: Icon(Icons.close),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
