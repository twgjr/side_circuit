import 'package:app/models/terminal.dart';

import 'package:app/models/segment.dart';
import 'package:app/models/vertex.dart';
import 'package:flutter/material.dart';

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

  void connect({Terminal? terminal}) {
    if (terminal != null) {
      if (vertices.isEmpty) {
        //start wire with terminal
        vertices.add(Vertex(terminal: terminal));
        terminal.vertex = vertices.last;
      } else {
        //end wire with terminal
        vertices.last.terminal = terminal;
        terminal.vertex = vertices.last;
      }
    } else {
      throw Exception('Must provide terminal');
    }
  }

  void startAt({Terminal? terminal}) {
    if (terminal != null) {
      vertices.add(Vertex(terminal: terminal));
      terminal.vertex = vertices.last;
      vertices.add(Vertex(
          position: terminal.position() +
              terminal.device.position() +
              terminal.shape.center()));
      segments.add(Segment(start: head(), end: tail()));
    } else {
      throw Exception('Must provide terminal');
    }
  }

  void endAt({Terminal? terminal}) {
    if (terminal != null) {
      vertices.last.terminal = terminal;
      terminal.vertex = vertices.last;
    } else {
      throw Exception('Must provide terminal');
    }
  }

  void placeAndContinue(Offset position) {
    Vertex oldTail = vertices.last;
    vertices.last.updatePosition(position);
    vertices.add(Vertex(position: position));
    segments.add(Segment(start: oldTail, end: tail()));
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
}
