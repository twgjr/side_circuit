import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/view/visual.dart';
import 'package:app/widgets/diagram/draggable_item.dart';
import 'package:app/widgets/general/popup_menu.dart';

class TerminalEditable extends ConsumerWidget {
  final Device device;
  final Terminal terminal;
  final double terminalRadius;
  final Visual tempVisual = Visual();
  OverlayEntry? menuOverlayEntry;

  TerminalEditable({
    super.key,
    required this.device,
    required this.terminal,
    required this.terminalRadius,
  });

  void _showTerminalPopupMenu(
      Offset position, BuildContext context, WidgetRef ref) {
    final RenderBox renderBox =
        Overlay.of(context).context.findRenderObject() as RenderBox;

    final relativePosition = RelativeRect.fromSize(
      Rect.fromLTWH(position.dx, position.dy, 0, 0),
      renderBox.size,
    );

    menuOverlayEntry = OverlayEntry(
      builder: (context) {
        return PopupMenu(
          relativePosition: relativePosition,
          children: [
            MenuItemButton(
              child: Text(
                "Delete",
                style: TextStyle(fontWeight: FontWeight.w500),
              ),
              onPressed: () {
                ref.read(circuitProvider.notifier).removeTerminal(terminal);
                menuOverlayEntry!.remove();
              },
            ),
          ],
        );
      },
    );
    Overlay.of(context).insert(menuOverlayEntry!);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return DraggableItem(
      visual: tempVisual,
      child: GestureDetector(
        onSecondaryTapDown: (details) {
          _showTerminalPopupMenu(details.globalPosition, context, ref);
        },
        child: Container(
          width: terminalRadius * 2,
          height: terminalRadius * 2,
          decoration: BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.black,
              width: 1.0,
            ),
          ),
          child: Text(
            terminal.name,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 10,
            ),
          ),
        ),
      ),
    );
  }
}
