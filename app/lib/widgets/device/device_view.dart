import 'package:app/widgets/device/device_widget.dart';
import 'package:app/widgets/diagram/draggable_item.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/device.dart';

class DeviceView extends ConsumerWidget {
  final Device device;
  final bool editable;

  DeviceView({required this.device, required this.editable});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (editable) {
      return DeviceWidget(device: device, editable: editable);
    } else {
      return DraggableItem(
        symbol: device.symbol,
        child: DeviceWidget(device: device, editable: editable),
      );
    }
  }
}
