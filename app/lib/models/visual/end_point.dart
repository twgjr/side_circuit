import 'package:flutter/material.dart';

import 'package:app/models/visual/vertex.dart';
import 'package:app/models/circuit/terminal.dart';

class EndPoint {
  final Vertex? vertex;
  final Terminal? terminal;

  EndPoint({this.vertex, this.terminal});

  bool get isVertex => vertex != null;
  bool get isTerminal => terminal != null;

  Offset get position {
    if (isVertex) {
      return vertex!.symbol.position;
    } else if (isTerminal) {
      return terminal!.diagramPosition();
    } else {
      throw Exception('EndPoint has no position');
    }
  }
}
