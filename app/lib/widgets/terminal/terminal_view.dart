import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/active_wire_provider.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/models/circuit/terminal.dart';

class TerminalView extends ConsumerWidget {
  final Terminal terminal;

  TerminalView({super.key, required this.terminal});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: terminal.position.dx,
      top: terminal.position.dy,
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () {
          final circuitRead = ref.read(circuitProvider.notifier);
          final newWire = circuitRead.startWireAt(terminal);
          ref.read(activeWireProvider.notifier).setActiveWire(newWire);
        },
        onPanUpdate: (details) {
          // do nothing except prevent the parent from receiving the event
        },
        onSecondaryTapDown: (details) {
          // _showPopupMenu(details.globalPosition, ref, context);
        },
        child: Container(
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              ShapeWidget(shape: terminal.symbol.shape),
              Text('${terminal.device.terminals.indexOf(terminal)}'),
            ],
          ),
        ),
      ),
    );
  }
}
