import 'package:app/models/circuit.dart';
import 'package:app/models/terminal.dart';
import 'package:app/models/diagram_symbol.dart';
import 'package:flutter/material.dart';

enum DeviceKind { V, I, R, VC, CC, SW, L, C, VG, CG, BLOCK }

class Device {
  Circuit circuit;
  List<Terminal> terminals = [];
  DeviceKind kind;
  int id = 0;
  DiagramSymbol symbol = DiagramSymbol();

  Device(this.circuit, this.kind) {
    id = circuit.maxDeviceIdOf(kind) + 1;
  }

  int index() => circuit.devices.indexOf(this);
  void addTerminal(Device device) => terminals.add(Terminal(this));
  void removeTerminalAt(int terminalIndex) => terminals.removeAt(terminalIndex);

  Device copy() {
    final newDevice = Device(circuit, kind);
    newDevice.id = id;
    newDevice.terminals = [];
    for (Terminal terminal in terminals) {
      newDevice.terminals.add(terminal.copyWith(device: newDevice));
    }
    newDevice.symbol = symbol.copy();
    return newDevice;
  }

  void updatePosition(Offset delta) {
    symbol.position += delta;
  }

  Offset position({bool offsetOverride = false, Offset? offset}) {
    if (offset == null) {
      return symbol.position;
    } else if (offsetOverride) {
      return offset;
    } else {
      return symbol.position + offset;
    }
  }
}
