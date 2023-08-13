import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/wire_provider.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';

class TerminalView extends ConsumerWidget {
  final Device device;
  final Terminal terminal;
  final double terminalRadius;

  TerminalView({
    super.key,
    required this.device,
    required this.terminal,
    required this.terminalRadius,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: terminal.visual.position.x,
      top: terminal.visual.position.y,
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () {
          final wireRead = ref.read(activeWireProvider.notifier);
          wireRead.start(terminal, ref);
        },
        onPanUpdate: (details) {
          // do nothing except prevent the parent from receiving the event
        },
        onSecondaryTapDown: (details) {
          // _showPopupMenu(details.globalPosition, ref, context);
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
