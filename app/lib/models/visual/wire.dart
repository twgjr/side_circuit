import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/end_point.dart';
import 'package:app/models/visual/wire_segment.dart';
import 'package:app/models/visual/vertex.dart';
import 'package:flutter/material.dart';

class Wire {
  Terminal? terminal;
  Node? node;
  List<WireSegment> segments = [];
  List<Vertex> vertices = [];

  Wire({this.terminal, this.node}) {
    final vertex = Vertex()
      ..symbol.shape.addCircle(5)
      ..symbol.position = Offset(200, 200)
      ..symbol.angle = 0;
    vertices.add(vertex);
    segments.add(WireSegment(wire: this, endPoint: EndPoint(vertex: vertex)));
  }

  Wire copyWith({Terminal? terminal, Node? node}) {
    final wire =
        Wire(terminal: terminal ?? this.terminal, node: node ?? this.node);
    for (WireSegment segment in segments) {
      wire.segments.add(WireSegment(wire: wire, endPoint: segment.endPoint));
    }
    for (Vertex vertex in vertices) {
      wire.vertices.add(vertex.copy());
    }
    return wire;
  }

  void removeTerminal(Terminal terminal) {
    if (this.terminal == terminal) {
      this.terminal = null;
    }
  }

  void addVertexAt(Offset position, double angleIncrement) {
    // final vertex = Vertex(position: position, angle: angleIncrement);
    final vertex = Vertex()
      ..symbol.shape.addCircle(5)
      ..symbol.position = position
      ..symbol.angle = angleIncrement;
    vertices.add(vertex);
    if (vertices.length > 1) {
      final end = vertices[vertices.length - 1];
      final segment = WireSegment(wire: this, endPoint: EndPoint(vertex: end));
      segments.add(segment);
    }
  }
}
