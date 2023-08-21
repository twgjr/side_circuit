import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/net.dart';

class ActiveWireNotifier extends StateNotifier<Net> {
  ActiveWireNotifier() : super(Net());
}

final activeWireProvider = StateNotifierProvider<ActiveWireNotifier, Net>(
  (ref) => ActiveWireNotifier(),
);
