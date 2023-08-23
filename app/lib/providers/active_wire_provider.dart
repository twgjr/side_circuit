import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/wire.dart';

class ActiveWireNotifier extends StateNotifier<Wire> {
  ActiveWireNotifier() : super(Wire());

  void update(Wire wire) {
    state = wire;
  }
}

final activeWireProvider = StateNotifierProvider<ActiveWireNotifier, Wire>(
  (ref) => ActiveWireNotifier(),
);
