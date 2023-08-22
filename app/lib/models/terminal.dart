import 'package:app/models/vertex.dart';
import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

import 'package:app/models/device.dart';
import 'package:app/models/diagram_symbol.dart';

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

  void setPosition(Offset position) {
    _symbol.position = position;
  }

  Offset position({bool? diagram, BoxConstraints? constraints, bool? center}) {
    Offset position = _symbol.position; // relative to device
    if (constraints != null) {
      return Offset(
        // center in box
        position.dx + constraints.maxWidth / 2 - device.shape.center().dx,
        position.dy + constraints.maxHeight / 2 - device.shape.center().dy,
      );
    }
    if (diagram == true) {
      position += device.position();
    }
    if (center == true) {
      position += shape.center();
    }
    return position;
  }

  void updatePosition(Offset delta) {
    _symbol.position += delta;
  }

  Shape get shape {
    return _symbol.shape;
  }
}
