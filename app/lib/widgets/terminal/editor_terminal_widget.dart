import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/device_providers.dart';
import 'package:app/models/terminal.dart';
import 'package:app/widgets/terminal/terminal_widget.dart';

class EditorTerminalWidget extends ConsumerWidget {
  final Offset offset;
  final Terminal terminal;

  EditorTerminalWidget({
    super.key,
    required this.terminal,
    required this.offset,
  });

  void _showPopupMenu(Offset position, WidgetRef ref, BuildContext context) {
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
              deviceRead.removeTerminal(terminal);
              break;
          }
        }
      },
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      top: terminal.position(offset: offset).dy,
      left: terminal.position(offset: offset).dx,
      child: GestureDetector(
        onPanUpdate: (details) {
          ref
              .read(deviceChangeProvider.notifier)
              .dragUpdateTerminal(terminal, details);
        },
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition, ref, context);
        },
        child: TerminalWidget(terminal: terminal, editable: true),
      ),
    );
  }
}
