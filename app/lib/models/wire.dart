import 'package:app/models/net.dart';
import 'package:app/models/terminal.dart';
import 'package:flutter/material.dart';

import 'package:app/models/segment.dart';
import 'package:app/models/vertex.dart';

class Wire {
  Net net;
  List<Segment> segments = [];
  List<Vertex> vertices = [];

  Wire(this.net);

  Vertex first() {
    return vertices.first;
  }

  Vertex last() {
    return vertices.last;
  }

  void start({Terminal? terminal, required Offset position}) {
    if (terminal != null) {
      vertices.add(Vertex(wire: this, terminal: terminal));
    } else {
      vertices.add(Vertex(wire: this, position: position));
    }
    vertices.add(Vertex(wire: this, position: position));
    segments.add(Segment(start: first(), end: last()));
  }
}
