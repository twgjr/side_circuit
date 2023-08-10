import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/diagram/draggable_item.dart';

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
  void _showPopupMenu(Offset position, WidgetRef ref) {
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
        PopupMenuItem(value: "delete", child: Text("Delete")),
      ],
    ).then(
      (value) {
        if (value != null) {
          switch (value) {
            case "delete":
              final deviceRead = ref.read(deviceChangeProvider.notifier);
              deviceRead.removeTerminal(widget.terminalCopy);
              break;
          }
        }
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return DraggableItem(
      visual: widget.terminalCopy.visual,
      child: GestureDetector(
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition, ref);
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
