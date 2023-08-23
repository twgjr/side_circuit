import 'package:app/models/wire.dart';
import 'package:app/providers/active_wire_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/mode_provider.dart';
import 'package:app/providers/circuit_provider.dart';
import 'package:app/models/terminal.dart';
import 'package:app/widgets/terminal/terminal_widget.dart';

class DiagramTerminalWidget extends ConsumerWidget {
  final Terminal terminal;
  final Offset offset;

  DiagramTerminalWidget({
    super.key,
    required this.terminal,
    required this.offset,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: terminal.position(offset: offset).dx,
      top: terminal.position(offset: offset).dy,
      child: GestureDetector(
        onTapDown: (details) {
          if (ref.read(modeProvider).addWire) {
            Wire activeWireRead = ref.read(activeWireProvider);
            if (activeWireRead.tail() != terminal.vertex) {
              final circuitRead = ref.read(circuitProvider.notifier);
              circuitRead.endWire(activeWireRead, terminal: terminal);
            } else {
              final circuitRead = ref.read(circuitProvider.notifier);
              Wire wire = circuitRead.startWire(terminal: terminal);
              ref.read(activeWireProvider.notifier).update(wire);
            }
          }
        },
        onPanUpdate: (details) {
          // do nothing except prevent the parent from receiving the event
        },
        child: TerminalWidget(terminal: terminal, editable: false),
      ),
    );
  }
}
