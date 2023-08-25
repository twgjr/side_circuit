import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/device.dart';
import 'package:app/widgets/device/device_widget.dart';

class EditorDeviceWidget extends ConsumerWidget {
  final Device device;
  final Offset? offset;

  EditorDeviceWidget({
    super.key,
    required this.device,
    this.offset,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: device.position(offset: offset, offsetOverride: true).dx,
      top: device.position(offset: offset, offsetOverride: true).dy,
      child: DeviceWidget(device: device, editable: true),
    );
  }
}
