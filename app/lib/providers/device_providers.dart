import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/device.dart';
import 'package:app/models/terminal.dart';

class DeviceChangeNotifier extends StateNotifier<Device> {
  DeviceChangeNotifier()
      : super(Device(
            index: (device) {
              return 0;
            },
            kind: DeviceKind.V));

  void update(Device device) {
    state = device;
  }

  void refresh() {
    final device = state.copy();
    state = device;
  }

  void dragUpdateTerminal(Terminal terminal, DragUpdateDetails details) {
    final device = state.copy();
    terminal.addPosition(details.delta);
    state = device;
  }

  void addTerminal() {
    final device = state.copy();
    device.addTerminal(device);
    state = device;
  }

  void removeTerminal(Terminal terminal) {
    final terminalIndex = state.terminals.indexOf(terminal);
    final device = state.copy();
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
