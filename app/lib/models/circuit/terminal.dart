import 'package:app/models/visual/vertex.dart';
import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/visual/diagram_symbol.dart';

class Terminal {
  Device device;
  Vertex? vertex;
  DiagramSymbol _symbol = DiagramSymbol();

  int get index => device.terminals.indexOf(this);

  Terminal(this.device) {
    _symbol.shape.addRect(10, 10);
    _symbol.shape.fillColor = Colors.white;
  }

  Terminal copyWith({Device? device, Vertex? wire}) {
    final terminal = Terminal(device ?? this.device);
    terminal.vertex = wire ?? this.vertex;
    terminal._symbol = _symbol.copy();
    return terminal;
  }

  Offset get diagramPosition {
    return _symbol.position + device.diagramPosition;
  }

  Offset editorPosition(BoxConstraints constraints) {
    return Offset(
      _symbol.position.dx + constraints.maxWidth / 2,
      _symbol.position.dy + constraints.maxHeight / 2,
    );
  }

  void updatePosition(Offset delta) {
    _symbol.position += delta;
  }

  Shape get shape {
    return _symbol.shape;
  }
}
