import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/providers/device_providers.dart';
import 'package:app/models/device.dart';
import 'package:app/widgets/device_editor/device_editor.dart';
import 'package:app/widgets/symbol/symbol_widget.dart';

class DeviceWidget extends ConsumerWidget {
  final Device device;
  final bool editable;

  DeviceWidget({super.key, required this.device, required this.editable});

  void showDeviceEditor(BuildContext context, WidgetRef ref, Device device) {
    ref.read(deviceOpenProvider.notifier).update(device);
    ref.read(deviceChangeProvider.notifier).update(device.copy());
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

  Widget selectWidget(bool editable, WidgetRef ref, BuildContext context) {
    if (editable) {
      return SymbolWidget(device.symbol);
    } else {
      return GestureDetector(
        behavior: HitTestBehavior.deferToChild,
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition, context, ref);
        },
        child: SymbolWidget(device.symbol),
      );
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          selectWidget(editable, ref, context),
          Positioned(
            bottom: -20,
            left: device.symbol.width() / 2 - 10,
            child: Text('${device.kind.name}${device.id}'),
          ),
        ],
      ),
    );
  }
}
