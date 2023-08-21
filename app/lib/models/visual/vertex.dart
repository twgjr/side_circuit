import 'package:app/models/visual/node.dart';
import 'package:app/models/circuit/terminal.dart';

import 'package:app/models/visual/diagram_symbol.dart';
import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

class Vertex {
  Terminal? terminal;
  Node? node;
  DiagramSymbol _symbol = DiagramSymbol();

  Vertex({this.terminal, this.node, Offset? position}) {
    if (position != null) {
      _symbol.position = position;
    }
  }

  Vertex copy() {
    return Vertex(terminal: terminal, node: node);
  }

  Offset get diagramPosition {
    if (terminal != null) {
      return terminal!.diagramPosition;
    } else if (node != null) {
      return node!.diagramPosition();
    } else {
      return _symbol.position;
    }
  }

  Shape get shape {
    return _symbol.shape;
  }
}
