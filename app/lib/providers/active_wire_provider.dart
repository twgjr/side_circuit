import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/visual/wire.dart';

class ActiveWireNotifier extends StateNotifier<Wire> {
  ActiveWireNotifier() : super(Wire(terminal: null, node: null));

  void setActiveWire(Wire wire) {
    state = wire;
  }
}

final activeWireProvider = StateNotifierProvider<ActiveWireNotifier, Wire>(
  (ref) => ActiveWireNotifier(),
);
