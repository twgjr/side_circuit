import 'package:app/models/terminal.dart';
import 'package:flutter/material.dart';

import 'package:app/models/segment.dart';
import 'package:app/models/vertex.dart';

class Wire {
  List<Segment> segments = [];
  List<Vertex> vertices = [];

  Wire();

  Vertex head() {
    return vertices.first;
  }

  Vertex tail() {
    return vertices.last;
  }

  void start({Terminal? terminal, Offset? position}) {
    if (terminal != null) {
      vertices.add(Vertex(terminal: terminal));
      terminal.vertex = vertices.last;
    } else if (position != null) {
      vertices.add(Vertex(position: position));
    } else {
      throw Exception('Must provide terminal or position');
    }
    vertices.add(Vertex(position: position));
    segments.add(Segment(start: head(), end: tail()));
  }

  void end({Terminal? terminal, Offset? position}) {
    if (terminal != null) {
      vertices.add(Vertex(terminal: terminal));
      terminal.vertex = vertices.last;
    } else if (position != null) {
      vertices.add(Vertex(position: position));
    } else {
      throw Exception('Must provide terminal or position');
    }
  }

  Wire copy() {
    final newWire = Wire();
    newWire.segments = [];
    for (Segment segment in segments) {
      newWire.segments.add(segment);
    }
    newWire.vertices = [];
    for (Vertex vertex in vertices) {
      newWire.vertices.add(vertex);
    }
    return newWire;
  }

  void setTail({Terminal? terminal, Offset? position}) {
    if (terminal != null) {
      vertices.last.terminal = terminal;
      terminal.vertex = vertices.last;
    } else if (position != null) {
      vertices.last.updatePosition(position);
    } else {
      throw Exception('Must provide terminal or position');
    }
  }
}
