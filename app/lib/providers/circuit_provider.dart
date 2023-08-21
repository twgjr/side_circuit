import 'package:app/models/visual/vertex.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';

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

  void replaceDeviceWith(Device newDevice, int atIndex) {
    final circuit = state.copy();
    circuit.replaceDeviceWith(newDevice, atIndex);
    state = circuit;
  }

  void dragUpdateDevice(Device device, Offset delta) {
    final circuit = state.copy();
    circuit.devices[device.index].updatePosition(delta);
    state = circuit;
  }

  Vertex startWire({Terminal? terminal, required Offset position}) {
    final circuit = state.copy();
    Vertex last = circuit.startWire(terminal: terminal, position: position);
    state = circuit;
    return last;
  }

  void dragUpdateVertex(Vertex vertex, Offset position) {
    final circuit = state.copy();
    vertex.updatePosition(position);
    state = circuit;
  }
}

final circuitProvider = StateNotifierProvider<CircuitNotifier, Circuit>(
  (ref) => CircuitNotifier(),
);
