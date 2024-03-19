import 'package:app/models/vertex.dart';
import 'package:app/models/diagram_symbol.dart';
import 'package:flutter/material.dart';

class Node {
  List<Vertex> vertices = [];
  DiagramSymbol symbol = DiagramSymbol();

  Node();

  Offset position() {
    return symbol.position;
  }
}
