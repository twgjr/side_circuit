import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/net.dart';
import 'package:app/models/visual/wire.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/visual/vertex.dart';

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
