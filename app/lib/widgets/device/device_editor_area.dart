import 'package:flutter/material.dart';

class DeviceEditorArea extends StatelessWidget {
  final Widget child;

  DeviceEditorArea({required this.child});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: Theme.of(context).primaryColorDark,
        ),
        child: child,
      ),
    );
  }
}
