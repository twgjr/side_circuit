import 'package:flutter/material.dart';

import 'package:app/models/circuit/terminal.dart';
import 'package:app/widgets/general/shape.dart';

class TerminalWidget extends StatelessWidget {
  final Terminal terminal;
  final bool editable;

  TerminalWidget({super.key, required this.terminal, required this.editable});

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        ShapeWidget(shape: terminal.shape),
        Text('${terminal.device.terminals.indexOf(terminal)}'),
      ],
    );
  }
}
