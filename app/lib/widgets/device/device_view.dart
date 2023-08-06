import 'package:app/providers/overlay_provider.dart';
import 'package:app/widgets/device/terminal_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/widgets/device/device_editor.dart';

class DeviceView extends ConsumerWidget {
  final Device device;

  DeviceView({
    required this.device,
  });

  void _showPopupMenu(Offset position, BuildContext context, WidgetRef ref) {
    final RenderBox overlay =
        Overlay.of(context).context.findRenderObject() as RenderBox;

    final relativePosition = RelativeRect.fromSize(
      Rect.fromLTWH(position.dx, position.dy, 0, 0),
      overlay.size,
    );

    showMenu(
      context: context,
      position: relativePosition,
      items: [
        PopupMenuItem(value: "editor", child: Text("Open Editor")),
        PopupMenuItem(value: "delete", child: Text("Delete")),
      ],
    ).then(
      (value) {
        if (value != null) {
          switch (value) {
            case "delete":
              ref.read(circuitProvider.notifier).removeDevice(device);
              break;
            case "editor":
              showDeviceEditor(context, ref);
              break;
            case "add terminal":
              ref.read(circuitProvider.notifier).addTerminal(device);
              break;
          }
        }
      },
    );
  }

  void showDeviceEditor(BuildContext context, WidgetRef ref) {
    OverlayEntry overlayEntry = OverlayEntry(
      builder: (context) => DeviceEditor(device: device),
    );
    Overlay.of(context).insert(overlayEntry);
    ref.read(overlayEntryProvider.notifier).addOverlay(overlayEntry);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return GestureDetector(
      onSecondaryTapDown: (details) {
        _showPopupMenu(details.globalPosition, context, ref);
      },
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
            for (Terminal terminal in device.terminals)
              TerminalView(
                device: device,
                terminal: terminal,
                terminalIndex: device.terminals.indexOf(terminal),
                terminalCount: device.terminals.length,
                terminalRadius: 10,
              ),
          ],
        ),
      ),
    );
  }
}
