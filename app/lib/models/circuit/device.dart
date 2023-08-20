import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/terminal.dart';
import 'package:app/models/visual/symbol.dart';
import 'package:flutter/material.dart';

enum DeviceKind { V, I, R, VC, CC, SW, L, C, VG, CG, BLOCK }

class Device {
  Circuit circuit;
  List<Terminal> terminals;
  DeviceKind kind;
  int id;
  Symbol symbol = Symbol();

  Device({
    required this.circuit,
    required this.kind,
  })  : id = circuit.maxDeviceIdOf(kind) + 1,
        terminals = [];

  int get index => circuit.devices.indexOf(this);
  void addTerminal(Device device) => terminals.add(Terminal(device, ""));
  void removeTerminalAt(int index) {
    final terminal = terminals[index];
    if (terminal.wire != null) {
      terminal.wire?.removeTerminal(terminal);
    }
    terminals.removeAt(index);
  }

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
    newDevice.symbol = symbol.copy();
    return newDevice;
  }

  void updatePosition(Offset delta) {
    symbol.position += delta;
  }
}
