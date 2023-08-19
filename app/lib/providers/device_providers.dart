import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';

class DeviceChangeNotifier extends StateNotifier<Device> {
  DeviceChangeNotifier()
      : super(Device(circuit: Circuit(), kind: DeviceKind.R));

  void update(Device device) {
    state = device;
  }

  void refresh() {
    final device = state.copyWith();
    state = device;
  }

  void dragUpdateTerminal(Terminal terminal, DragUpdateDetails details) {
    final terminalIndex = state.terminals.indexOf(terminal);
    final device = state.copyWith();
    device.terminals[terminalIndex].symbol.position += details.delta;
    state = device;
  }

  void addTerminal() {
    final device = state.copyWith();
    device.addTerminal(device);
    state = device;
  }

  void removeTerminal(Terminal terminal) {
    final terminalIndex = state.terminals.indexOf(terminal);
    final device = state.copyWith();
    device.removeTerminalAt(terminalIndex);
    state = device;
  }
}

final deviceChangeProvider =
    StateNotifierProvider<DeviceChangeNotifier, Device>(
  (ref) => DeviceChangeNotifier(),
);

final deviceOpenProvider = StateNotifierProvider<DeviceChangeNotifier, Device>(
  (ref) => DeviceChangeNotifier(),
);
