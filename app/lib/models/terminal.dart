import 'package:app/models/vertex.dart';
import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

import 'package:app/models/device.dart';
import 'package:app/models/diagram_symbol.dart';

class Terminal {
  Vertex? vertex;
  DiagramSymbol _symbol = DiagramSymbol();

  Terminal() {
    _symbol.shape.addRect(10, 10);
    _symbol.shape.fillColor = Colors.white;
  }

  Terminal copyWith({Device? device, Vertex? wire}) {
    final terminal = Terminal();
    terminal.vertex = wire ?? this.vertex;
    terminal._symbol = _symbol.copy();
    return terminal;
  }

  Offset position({bool offsetOverride = false, Offset? offset}) {
    if (offset == null) {
      return _symbol.position;
    } else if (offsetOverride) {
      return offset;
    } else {
      return _symbol.position + offset;
    }
  }

  void setPosition(Offset position) {
    _symbol.position = position;
  }

  void addPosition(Offset delta) {
    _symbol.position += delta;
  }

  Shape get shape {
    return _symbol.shape;
  }
}
