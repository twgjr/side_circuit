import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/overlay_provider.dart';

class DeviceEditorTopBar extends ConsumerWidget {
  final Function(double) addX;
  final Function(double) addY;

  DeviceEditorTopBar({
    required this.addX,
    required this.addY,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onPanUpdate: (details) {
        addX(details.delta.dx);
        addY(details.delta.dy);
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
                onPressed: () {
                  final overlayEntry = ref.read(overlayEntryProvider).last;
                  ref
                      .read(overlayEntryProvider.notifier)
                      .removeOverlay(overlayEntry);
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
