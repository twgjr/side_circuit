import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/device_providers.dart';

class DeviceEditorTopBar extends ConsumerWidget {
  final void Function(Offset) onDrag;

  DeviceEditorTopBar({super.key, required this.onDrag});

  void closeDeviceEditor({
    required WidgetRef ref,
    required bool save,
    required BuildContext context,
  }) {
    final circuitRead = ref.read(circuitProvider.notifier);
    final deviceChangeRead = ref.read(deviceChangeProvider);
    final deviceOpenRead = ref.read(deviceOpenProvider);

    if (save) {
      final circuitWatch = ref.watch(circuitProvider);
      circuitRead.replaceDeviceWith(
          deviceChangeRead, circuitWatch.devices.indexOf(deviceOpenRead));
    }
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onPanUpdate: (details) {
        onDrag(details.delta);
      },
      child: Container(
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(
              color: Colors.black,
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
                  closeDeviceEditor(
                    ref: ref,
                    save: true,
                    context: context,
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: ElevatedButton(
                onPressed: () {
                  closeDeviceEditor(
                    ref: ref,
                    save: false,
                    context: context,
                  );
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
