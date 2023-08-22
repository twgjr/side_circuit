import 'package:flutter/material.dart';

import 'package:app/models/terminal.dart';
import 'package:app/widgets/general/shape.dart';

class TerminalWidget extends StatelessWidget {
  final Terminal terminal;
  final bool editable;

  TerminalWidget({super.key, required this.terminal, required this.editable});

  Shape _selectShape() {
    if (terminal.vertex != null) {
      terminal.shape.fillColor = Colors.black;
      return terminal.shape;
    } else {
      terminal.shape.fillColor = Colors.white;
      return terminal.shape;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        ShapeWidget(shape: _selectShape()),
        Positioned(
          top: terminal.position(center: true).dy,
          child: Text('${terminal.device.terminals.indexOf(terminal)}'),
        ),
      ],
    );
  }
}
