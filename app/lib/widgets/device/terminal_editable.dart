import 'package:app/widgets/general/popup_menu.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/providers/overlay_provider.dart';
import 'package:app/providers/circuit_provider.dart';

class TerminalEditable extends ConsumerWidget {
  final Device device;
  final Terminal terminal;
  final int terminalIndex;
  final int terminalCount;
  final double terminalRadius;

  TerminalEditable({
    super.key,
    required this.device,
    required this.terminal,
    required this.terminalIndex,
    required this.terminalCount,
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

    OverlayEntry overlayEntry = OverlayEntry(
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
                ref.read(overlayEntryProvider.notifier).removeLastOverlay();
              },
            ),
          ],
        );
      },
    );
    Overlay.of(context).insert(overlayEntry);
    ref.read(overlayEntryProvider.notifier).addOverlay(overlayEntry);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: terminal.visual.startPosition.x,
      top: terminal.visual.startPosition.y,
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
