import 'package:app/models/terminal.dart';
import 'package:app/models/diagram_symbol.dart';
import 'package:app/widgets/general/shape.dart';
import 'package:flutter/material.dart';

enum DeviceKind { V, I, R, VC, CC, SW, L, C, VG, CG, BLOCK }

class Device {
  List<Terminal> terminals = [];
  DeviceKind kind;
  int Function(Device) index;
  int id = 0;
  DiagramSymbol _symbol = DiagramSymbol();

  Device({required this.index, required this.kind}) {
    id = index(this) + 1;
  }

  void addTerminal(Device device) => terminals.add(Terminal());
  void removeTerminalAt(int terminalIndex) => terminals.removeAt(terminalIndex);

  Device copy() {
    final newDevice = Device(index: index, kind: kind);
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

  Offset position({bool offsetOverride = false, Offset? offset}) {
    if (offset == null) {
      return _symbol.position;
    } else if (offsetOverride) {
      return offset;
    } else {
      return _symbol.position + offset;
    }
  }

  Shape get shape {
    return _symbol.shape;
  }
}
