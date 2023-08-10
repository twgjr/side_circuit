import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/diagram/draggable_item.dart';
import 'package:app/widgets/general/popup_menu.dart';

class TerminalEditable extends ConsumerStatefulWidget {
  final Device device;
  final Terminal terminalCopy;
  final double terminalRadius;

  TerminalEditable({
    super.key,
    required this.device,
    required this.terminalCopy,
    required this.terminalRadius,
  });

  @override
  TerminalEditableState createState() => TerminalEditableState();
}

class TerminalEditableState extends ConsumerState<TerminalEditable> {
  OverlayEntry? menuOverlayEntry;

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
                ref
                    .read(deviceChangeProvider.notifier)
                    .removeTerminal(widget.terminalCopy);
                menuOverlayEntry!.remove();
                print("Delete");
              },
            ),
          ],
        );
      },
    );
    Overlay.of(context).insert(menuOverlayEntry!);
  }

  @override
  Widget build(BuildContext context) {
    return DraggableItem(
      visual: widget.terminalCopy.visual,
      child: GestureDetector(
        onSecondaryTapDown: (details) {
          _showTerminalPopupMenu(details.globalPosition, context, ref);
        },
        child: Container(
          width: widget.terminalRadius * 2,
          height: widget.terminalRadius * 2,
          decoration: BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.black,
              width: 1.0,
            ),
          ),
          child: Text(
            widget.terminalCopy.device.terminals
                .indexOf(widget.terminalCopy)
                .toString(),
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
