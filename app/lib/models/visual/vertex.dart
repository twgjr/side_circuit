import 'package:flutter/material.dart';

import 'package:app/models/visual/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/diagram_symbol.dart';
import 'package:app/models/visual/wire.dart';
import 'package:app/widgets/general/shape.dart';

class Vertex {
  Wire wire;
  Terminal? terminal;
  Node? node;
  DiagramSymbol _symbol = DiagramSymbol();

  Vertex({required this.wire, this.terminal, this.node, Offset? position}) {
    if (position != null) {
      _symbol.position = position;
    }
  }

  Vertex copy() {
    return Vertex(wire: wire, terminal: terminal, node: node);
  }

  Offset position() {
    if (terminal != null) {
      return terminal!.position(diagram: true);
    } else if (node != null) {
      return node!.position();
    } else {
      return _symbol.position;
    }
  }

  void updatePosition(Offset position) {
    _symbol.position = position;
  }

  Shape get shape {
    return _symbol.shape;
  }
}
