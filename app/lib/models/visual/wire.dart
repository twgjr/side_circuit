import 'package:flutter/material.dart';

import 'package:app/models/visual/segment.dart';
import 'package:app/models/visual/vertex.dart';

class Wire {
  List<Segment> segments = [];
  List<Vertex> vertices = [];

  Wire();

  Vertex first() {
    return vertices.first;
  }

  Vertex last() {
    return vertices.last;
  }

  void start({Vertex? vertex, Offset? position}) {
    if (vertex != null) {
      vertices.add(vertex);
    } else if (position != null) {
      vertices.add(Vertex(position: position));
    } else {
      vertices.add(Vertex());
    }
  }
}
