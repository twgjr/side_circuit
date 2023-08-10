import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/circuit.dart';
import 'package:app/models/circuit/device.dart';

class DeviceChangeNotifier extends StateNotifier<Device> {
  DeviceChangeNotifier()
      : super(Device(circuit: Circuit(), kind: DeviceKind.R));

  void update(Device device) {
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
