import 'package:app/models/circuit.dart';
import 'package:app/models/net.dart';
import 'package:app/models/wire.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/vertex.dart';

class ActiveVertexNotifier extends StateNotifier<Vertex> {
  ActiveVertexNotifier() : super(Vertex(wire: Wire(Net(Circuit()))));

  void set(Vertex vertex) {
    state = vertex;
  }
}

final activeVertexProvider =
    StateNotifierProvider<ActiveVertexNotifier, Vertex>(
  (ref) => ActiveVertexNotifier(),
);
