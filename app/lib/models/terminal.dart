import 'package:flutter/material.dart';

import 'package:app/models/vertex.dart';
import 'package:app/models/device.dart';
import 'package:app/models/diagram_symbol.dart';

class Terminal {
  Device device;
  Vertex? vertex;
  DiagramSymbol symbol = DiagramSymbol.terminal();

  Terminal(Device this.device);

  Terminal copyWith({Device? device, Vertex? wire}) {
    final terminal = Terminal(device ?? this.device);
    terminal.vertex = wire ?? this.vertex;
    terminal.symbol = symbol.copy();
    return terminal;
  }

  Offset position({bool offsetOverride = false, Offset? offset}) {
    if (offset == null) {
      return symbol.position;
    } else if (offsetOverride) {
      return offset;
    } else {
      return symbol.position + offset;
    }
  }

  void setPosition(Offset position) {
    symbol.position = position;
  }

  void addPosition(Offset delta) {
    symbol.position += delta;
  }
}
