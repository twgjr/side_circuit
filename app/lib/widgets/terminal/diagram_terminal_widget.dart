import 'package:app/models/wire.dart';
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
          final modeRead = ref.read(modeProvider.notifier);
          final modeWatch = ref.watch(modeProvider);
          if (modeWatch.addWire) {
            if (modeWatch.activeWire == null && terminal.vertex == null) {
              final circuitRead = ref.read(circuitProvider.notifier);
              Wire wire = circuitRead.startWireAt(terminal: terminal);
              modeRead.setActiveWire(wire);
            }
            if (modeWatch.activeWire != null && terminal.vertex == null) {
              final circuitRead = ref.read(circuitProvider.notifier);
              final wire = modeWatch.activeWire!;
              circuitRead.endWireAt(wire: wire, terminal: terminal);
              modeRead.reset();
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
