import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';

class CircuitProvider extends StateNotifier<Circuit> {
  CircuitProvider() : super(Circuit());

  void addDeviceOf(DeviceKind kind) {
    final circuit = state.copy();
    circuit.addDeviceOf(kind);
    state = circuit;
  }

  void newNode() {
    final circuit = state.copy();
    circuit.newNode();
    state = circuit;
  }
}

final circuitProvider = StateNotifierProvider<CircuitProvider, Circuit>(
  (ref) => CircuitProvider(),
);
