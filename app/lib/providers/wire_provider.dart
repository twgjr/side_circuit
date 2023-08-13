import 'package:app/models/circuit/terminal.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/wire.dart';
import 'package:app/providers/gesture_state_provider.dart';

class ActiveWireNotifier extends StateNotifier<Wire> {
  ActiveWireNotifier() : super(Wire(terminal: null, node: null));

  void start(Terminal terminal, WidgetRef ref) {
    final gestureStateRead = ref.read(gestureStateProvider);
    if (gestureStateRead.addWire) {
      state = Wire(terminal: terminal);
    }
  }
}

final activeWireProvider = StateNotifierProvider<ActiveWireNotifier, Wire>(
  (ref) => ActiveWireNotifier(),
);
