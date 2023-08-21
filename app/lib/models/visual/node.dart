import 'package:app/models/visual/vertex.dart';
import 'package:app/models/visual/diagram_symbol.dart';
import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

class Node {
  List<Vertex> vertices = [];
  DiagramSymbol _symbol = DiagramSymbol();

  Node();

  Offset position() {
    return _symbol.position;
  }

  Shape get shape {
    return _symbol.shape;
  }
}
