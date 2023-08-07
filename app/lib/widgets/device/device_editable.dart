import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/device/terminal_editable.dart';
import 'package:app/providers/circuit_provider.dart';

class DeviceEditable extends ConsumerWidget {
  final Device device;
  final BuildContext cktViewCtx;

  DeviceEditable({required this.device, required this.cktViewCtx});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final circuitWatch = ref.watch(circuitProvider);
    final deviceWatch =
        circuitWatch.devices[circuitWatch.devices.indexOf(device)];
    return GestureDetector(
      onSecondaryTapDown: (details) {},
      child: Card(
        child: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                children: [
                  Text('${device.kind.name}${device.id}'),
                ],
              ),
            ),
            for (Terminal terminal in deviceWatch.terminals)
              TerminalEditable(
                device: device,
                terminal: terminal,
                terminalRadius: 10,
              ),
          ],
        ),
      ),
    );
  }
}
