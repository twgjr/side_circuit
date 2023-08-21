import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/circuit_provider.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/widgets/device/device_widget.dart';

class DeviceView extends ConsumerWidget {
  final Device device;
  final BoxConstraints? editorConstraints;

  DeviceView({
    super.key,
    required this.device,
    this.editorConstraints,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return _buildDevice(context, ref, editorConstraints);
  }

  Widget _editorDevice(
      BuildContext context, WidgetRef ref, BoxConstraints constraints) {
    return Positioned(
      left: device.position(constraints: constraints).dx,
      top: device.position(constraints: constraints).dy,
      child: DeviceWidget(device: device, editable: true),
    );
  }

  Widget _diagramDevice(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: device.position().dx,
      top: device.position().dy,
      child: GestureDetector(
        onPanUpdate: (details) {
          ref
              .read(circuitProvider.notifier)
              .dragUpdateDevice(device, details.delta);
        },
        child: DeviceWidget(device: device, editable: false),
      ),
    );
  }

  Widget _buildDevice(
      BuildContext context, WidgetRef ref, BoxConstraints? constraints) {
    if (constraints != null) {
      return _editorDevice(context, ref, constraints);
    } else {
      return _diagramDevice(context, ref);
    }
  }
}
