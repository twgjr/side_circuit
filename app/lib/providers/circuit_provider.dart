import 'package:app/models/segment.dart';
import 'package:app/models/wire.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/terminal.dart';
import 'package:app/models/circuit.dart';
import 'package:app/models/device.dart';
import 'package:app/models/vertex.dart';

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
    device.updatePosition(delta);
    state = circuit;
  }

  void dragUpdateVertex(Vertex vertex, Offset position) {
    final circuit = state.copy();
    vertex.updatePosition(position);
    state = circuit;
  }

  Wire startWireAt({Terminal? terminal}) {
    final circuit = state.copy();
    Wire wire = circuit.startWireAt(terminal: terminal);
    state = circuit;
    return wire;
  }

  void endWireAt({required Wire wire, Terminal? terminal}) {
    final circuit = state.copy();
    circuit.endWireAt(wire: wire, terminal: terminal);
    state = circuit;
  }

  void removeWireContaining(Segment segment) {
    final circuit = state.copy();
    segment.wire.net.deleteWire(segment.wire);
    state = circuit;
  }
}

final circuitProvider = StateNotifierProvider<CircuitNotifier, Circuit>(
  (ref) => CircuitNotifier(),
);
