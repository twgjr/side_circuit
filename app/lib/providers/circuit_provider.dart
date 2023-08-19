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

  Wire startWireAt(Terminal terminal) {
    final terminalIndex = terminal.device.terminals.indexOf(terminal);
    final deviceIndex = state.devices.indexOf(terminal.device);
    final circuit = state.copy(deep: false);
    circuit.startWireAt(terminalIndex, deviceIndex);
    state = circuit;
    return terminal.wire!;
  }

  void addVertexToWireAt(Offset position, Wire wire) {
    final circuit = state.copy(deep: false);
    circuit.addVertexToWireAt(position, 0, wire);
    state = circuit;
  }

  void refresh() {
    final circuit = state.copy(deep: false);
    state = circuit;
  }

  void dragUpdateDevice(Device device, DragUpdateDetails details) {
    final int index = state.devices.indexOf(device);
    final circuit = state.copy(deep: false);
    circuit.devices[index].symbol.position += details.delta;
    state = circuit;
  }
}

final circuitProvider = StateNotifierProvider<CircuitNotifier, Circuit>(
  (ref) => CircuitNotifier(),
);
