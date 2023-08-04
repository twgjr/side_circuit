import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';

class CircuitProvider extends StateNotifier<Circuit> {
  CircuitProvider() : super(Circuit());

  void addDeviceOf(DeviceKind kind) {
    final circuit = state.copy();
    circuit.addDeviceOf(kind);
    state = circuit;
  }

  void removeDevice(Device element) {
    final circuit = state.copy();
    circuit.removeDevice(element);
    state = circuit;
  }

  void newNode() {
    final circuit = state.copy();
    circuit.newNode();
    state = circuit;
  }

  void removeNode(Node node) {
    final circuit = state.copy();
    circuit.removeNode(node);
    state = circuit;
  }
}

final circuitProvider = StateNotifierProvider<CircuitProvider, Circuit>(
  (ref) => CircuitProvider(),
);
