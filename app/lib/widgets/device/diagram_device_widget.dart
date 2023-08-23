import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/models/device.dart';
import 'package:app/widgets/device/device_widget.dart';

class DiagramDeviceWidget extends ConsumerWidget {
  final Device device;
  final Offset? offset;

  DiagramDeviceWidget({
    super.key,
    required this.device,
    this.offset,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: device.position().dx,
      top: device.position().dy,
      child: GestureDetector(
        onPanUpdate: (details) {
          ref
              .read(circuitProvider.notifier)
              .dragUpdateDevice(device, details.delta);
        },
        child: DeviceBaseWidget(device: device, editable: false),
      ),
    );
  }
}
