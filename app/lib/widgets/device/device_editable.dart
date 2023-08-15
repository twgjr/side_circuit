import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/terminal/terminal_editable.dart';
import 'package:app/widgets/general/shape.dart';

class DeviceEditable extends ConsumerWidget {
  DeviceEditable({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final device = ref.watch(deviceChangeProvider);
    return Container(
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          ShapeWidget(shape: device.symbol.shape),
          Text('${device.kind.name}${device.id}'),
          for (Terminal terminal in device.terminals)
            TerminalEditable(terminal: terminal),
        ],
      ),
    );
  }
}
