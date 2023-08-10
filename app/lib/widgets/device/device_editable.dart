import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/device/terminal_editable.dart';
import 'package:app/widgets/general/shape.dart';

class DeviceEditable extends ConsumerWidget {
  DeviceEditable({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final deviceCopy = ref.watch(deviceChangeProvider);
    return Container(
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Shape(shape: deviceCopy.visual.shape),
          Text('${deviceCopy.kind.name}${deviceCopy.id}'),
          for (Terminal terminal in deviceCopy.terminals)
            TerminalEditable(
              device: deviceCopy,
              terminalCopy: terminal,
              terminalRadius: 10,
            ),
        ],
      ),
    );
  }
}
