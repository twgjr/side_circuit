import 'package:flutter/material.dart';

import 'package:app/models/node.dart';
import 'package:app/models/terminal.dart';
import 'package:app/models/diagram_symbol.dart';

class Vertex {
  Terminal? terminal;
  Node? node;
  DiagramSymbol symbol = DiagramSymbol.vertex();

  Vertex({this.terminal, this.node, Offset? position}) {
    if (position != null) {
      symbol.position = position;
    }
  }

  Vertex copy() {
    return Vertex(terminal: terminal, node: node);
  }

  Offset position() {
    if (terminal != null) {
      return terminal!.position() +
          terminal!.device.position() -
          terminal!.symbol.center();
    } else if (node != null) {
      return node!.position() - symbol.shape.center();
    } else {
      return symbol.position - symbol.shape.center();
    }
  }

  void updatePosition(Offset position) {
    symbol.position = position;
  }
}
