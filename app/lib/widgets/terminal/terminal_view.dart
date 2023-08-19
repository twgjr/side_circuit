import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/active_wire_provider.dart';
import 'package:app/providers/mode_provider.dart';
import 'package:app/providers/device_providers.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/terminal/terminal_widget.dart';

class TerminalView extends ConsumerWidget {
  final Terminal terminal;
  final bool editable;
  final Offset offset;

  TerminalView({
    super.key,
    required this.terminal,
    required this.editable,
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
    if (editable) {
      return Positioned(
        top: terminal.position.dy,
        left: terminal.position.dx,
        child: GestureDetector(
          onPanUpdate: (details) {
            ref
                .read(deviceChangeProvider.notifier)
                .dragUpdateTerminal(terminal, details);
          },
          onSecondaryTapDown: (details) {
            _showPopupMenu(details.globalPosition, ref, context);
          },
          child: TerminalWidget(terminal: terminal, editable: editable),
        ),
      );
    } else {
      return Positioned(
        left: terminal.position.dx + terminal.device.symbol.position.dx,
        top: terminal.position.dy + terminal.device.symbol.position.dy,
        child: GestureDetector(
          onTap: () {
            if (ref.read(modeStateProvider).addWire) {
              final circuitRead = ref.read(circuitProvider.notifier);
              final newWire = circuitRead.startWireAt(terminal);
              ref.read(activeWireProvider.notifier).setActiveWire(newWire);
            }
            print('tapped terminal');
          },
          onPanUpdate: (details) {
            // do nothing except prevent the parent from receiving the event
          },
          child: TerminalWidget(terminal: terminal, editable: editable),
        ),
      );
    }
  }
}
