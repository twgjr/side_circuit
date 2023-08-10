import 'package:app/widgets/diagram/draggable_item.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/device_providers.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/terminal/terminal_view.dart';
import 'package:app/widgets/device_editor/device_editor.dart';
import 'package:app/widgets/general/shape.dart';

class DeviceView extends ConsumerWidget {
  final Device device;

  DeviceView({required this.device});

  void showDeviceEditor(BuildContext context, WidgetRef ref, Device device) {
    ref.read(deviceOpenProvider.notifier).update(device);
    ref.read(deviceChangeProvider.notifier).update(device.copyWith());
    showDialog(context: context, builder: (_) => DeviceEditor());
  }

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
              showDeviceEditor(context, ref, device);
              break;
          }
        }
      },
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return DraggableItem(
      visual: device.visual,
      child: GestureDetector(
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition, context, ref);
        },
        child: Container(
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              Shape(shape: device.visual.shape),
              Text('${device.kind.name}${device.id}'),
              for (Terminal terminal in device.terminals)
                TerminalView(
                  device: device,
                  terminalRadius: 10,
                  terminal: terminal,
                ),
            ],
          ),
        ),
      ),
    );
  }
}
