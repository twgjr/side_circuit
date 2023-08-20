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

  Widget _editableDevice(
      BuildContext context, WidgetRef ref, BoxConstraints constraints) {
    return Positioned(
      left: constraints.maxWidth / 2,
      top: constraints.maxHeight / 2,
      child: DeviceWidget(device: device, editable: true),
    );
  }

  Widget _staticDevice(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: device.symbol.position.dx,
      top: device.symbol.position.dy,
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
      return _editableDevice(context, ref, constraints);
    } else {
      return _staticDevice(context, ref);
    }
  }
}
