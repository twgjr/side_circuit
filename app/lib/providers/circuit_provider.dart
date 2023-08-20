import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/wire.dart';

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

  Wire startWireAt(Terminal terminal) {
    final circuit = state.copy();
    circuit.startWireAt(terminal.index, terminal.device.index);
    state = circuit;
    return terminal.wire!;
  }

  void addVertexToWireAt(Offset position, Wire wire) {
    final circuit = state.copy();
    circuit.addVertexToWireAt(position, 0, wire);
    state = circuit;
  }

  void refresh() {
    final circuit = state.copy();
    state = circuit;
  }

  void dragUpdateDevice(Device device, Offset delta) {
    final circuit = state.copy();
    circuit.devices[device.index].updatePosition(delta);
    state = circuit;
  }
}

final circuitProvider = StateNotifierProvider<CircuitNotifier, Circuit>(
  (ref) => CircuitNotifier(),
);
