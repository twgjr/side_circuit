import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/diagram_symbol.dart';
import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

enum DeviceKind { V, I, R, VC, CC, SW, L, C, VG, CG, BLOCK }

class Device {
  Circuit circuit;
  List<Terminal> terminals;
  DeviceKind kind;
  int id;
  DiagramSymbol _symbol = DiagramSymbol();

  Device({
    required this.circuit,
    required this.kind,
  })  : id = circuit.maxDeviceIdOf(kind) + 1,
        terminals = [];

  int get index => circuit.devices.indexOf(this);
  void addTerminal(Device device) => terminals.add(Terminal(device));
  void removeTerminalAt(int index) => terminals.removeAt(index);

  Device copyWith({Circuit? circuit}) {
    final newDevice = Device(
      circuit: circuit ?? this.circuit,
      kind: kind,
    );
    newDevice.id = id;
    newDevice.terminals = [];
    for (Terminal terminal in terminals) {
      newDevice.terminals.add(terminal.copyWith(device: newDevice));
    }
    newDevice._symbol = _symbol.copy();
    return newDevice;
  }

  void updatePosition(Offset delta) {
    _symbol.position += delta;
  }

  Offset position({BoxConstraints? constraints}) {
    if (constraints != null) {
      return Offset(
        // center in box
        constraints.maxWidth / 2,
        constraints.maxHeight / 2,
      );
    } else {
      return _symbol.position; // relative to device
    }
  }

  Shape get shape {
    return _symbol.shape;
  }
}
