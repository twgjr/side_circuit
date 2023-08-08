import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/widgets/diagram/draggable_item.dart';
import 'package:app/widgets/device/device_view.dart';
import 'package:app/widgets/node/node_view.dart';
import 'package:app/widgets/device/device_editor.dart';

class Diagram extends ConsumerWidget {
  OverlayEntry? deviceEditorOverlayEntry;
  Device? deviceBeingEdited;
  Device? copyOfDeviceBeingEdited;

  Diagram({super.key});

  void showDeviceEditor(BuildContext context, WidgetRef ref, Device device) {
    deviceBeingEdited = device;
    copyOfDeviceBeingEdited = device.copyWith();
    deviceEditorOverlayEntry = OverlayEntry(
        builder: (context) => DeviceEditor(
              deviceCopy: copyOfDeviceBeingEdited!,
              onEditComplete: closeDeviceEditor,
            ));
    Overlay.of(context).insert(deviceEditorOverlayEntry!);
  }

  void closeDeviceEditor(WidgetRef ref, bool save) {
    if (save) {
      final circuitRead = ref.read(circuitProvider.notifier);
      circuitRead.replaceDeviceWith(
        copyOfDeviceBeingEdited!,
        deviceBeingEdited!.index(),
      );
    }
    if (deviceEditorOverlayEntry != null) {
      deviceEditorOverlayEntry!.remove();
      deviceEditorOverlayEntry = null;
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final circuitWatch = ref.watch(circuitProvider);
    return Stack(
      children: [
        for (Device device in circuitWatch.devices)
          DraggableItem(
            visual: device.visual,
            child: DeviceView(
              device: device,
              onEdit: showDeviceEditor,
            ),
          ),
        for (Node node in circuitWatch.nodes)
          DraggableItem(
            visual: node.visual,
            child: NodeView(
              node: node,
              cktViewCtx: context,
            ),
          ),
      ],
    );
  }
}
