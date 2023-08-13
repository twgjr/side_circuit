import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';

class CircuitNotifier extends StateNotifier<Circuit> {
  CircuitNotifier() : super(Circuit());

  void addDeviceOf(DeviceKind kind) {
    final circuit = state.copy(deep: false);
    circuit.addDeviceOf(kind);
    state = circuit;
  }

  void removeDevice(Device element) {
    final circuit = state.copy(deep: false);
    circuit.removeDevice(element);
    state = circuit;
  }

  void replaceDeviceWith(Device newDevice, int atIndex) {
    final circuit = state.copy(deep: false);
    circuit.replaceDeviceWith(newDevice, atIndex);
    state = circuit;
  }

  void newNode() {
    final circuit = state.copy(deep: false);
    circuit.newNode();
    state = circuit;
  }

  void removeNode(Node node) {
    final circuit = state.copy(deep: false);
    circuit.removeNode(node);
    state = circuit;
  }

  void startWireAt(Terminal terminal) {
    final circuit = state.copy(deep: false);
    circuit.startWireAt(terminal);
    state = circuit;
    print('startWireAt: ${terminal.device.terminals.indexOf(terminal)}');
  }
}

final circuitProvider = StateNotifierProvider<CircuitNotifier, Circuit>(
  (ref) => CircuitNotifier(),
);
