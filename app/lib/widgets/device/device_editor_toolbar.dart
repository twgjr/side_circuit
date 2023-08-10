import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';

class DeviceEditorToolbar extends ConsumerWidget {
  DeviceEditorToolbar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
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
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: IconButton(
              tooltip: 'Add Terminal',
              onPressed: () {
                final deviceChangeRead =
                    ref.read(deviceChangeProvider.notifier);
                deviceChangeRead.addTerminal();
              },
              icon: Icon(
                Icons.circle,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: IconButton(
              tooltip: 'Huh?',
              onPressed: () {
                print('DeviceEditorToolbar: onPressed: Huh?');
              },
              icon: Icon(
                Icons.question_mark,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
