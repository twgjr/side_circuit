import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';

class CircuitNotifier extends StateNotifier<Circuit> {
  CircuitNotifier() : super(Circuit());

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

  void addTerminal(Device device) {
    final circuit = state.copy();
    circuit.addTerminal(device);
    state = circuit;
  }

  void removeTerminal(Terminal terminal) {
    final circuit = state.copy();
    circuit.removeTerminal(terminal);
    state = circuit;
  }

  void connect(Terminal terminal, Node node) {
    final circuit = state.copy();
    circuit.connect(terminal, node);
    state = circuit;
  }

  void disconnect(Terminal terminal) {
    final circuit = state.copy();
    circuit.disconnect(terminal);
    state = circuit;
  }
}

final circuitProvider = StateNotifierProvider<CircuitNotifier, Circuit>(
  (ref) => CircuitNotifier(),
);