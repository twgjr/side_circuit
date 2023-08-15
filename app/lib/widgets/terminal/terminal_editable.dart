import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/diagram/draggable_item.dart';
import 'package:app/widgets/general/shape.dart';

class TerminalEditable extends ConsumerStatefulWidget {
  final Terminal terminal;

  TerminalEditable({super.key, required this.terminal});

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
              deviceRead.removeTerminal(widget.terminal);
              break;
          }
        }
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return DraggableItem(
      symbol: widget.terminal.symbol,
      child: GestureDetector(
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition, ref);
        },
        child: Container(
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              ShapeWidget(shape: widget.terminal.symbol.shape),
              Text(
                  '${widget.terminal.device.terminals.indexOf(widget.terminal)}'),
            ],
          ),
        ),
      ),
    );
  }
}
