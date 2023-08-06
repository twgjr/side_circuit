import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/terminal.dart';

class TerminalView extends ConsumerWidget {
  final Device device;
  final Terminal terminal;
  final int terminalIndex;
  final int terminalCount;
  final double terminalRadius;

  TerminalView({
    super.key,
    required this.device,
    required this.terminal,
    required this.terminalIndex,
    required this.terminalCount,
    required this.terminalRadius,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Positioned(
      left: terminal.visual.startPosition.x,
      top: terminal.visual.startPosition.y,
      child: GestureDetector(
        onTap: () {
          print('TerminalView: onTap: terminal=$terminal');
        },
        child: Container(
          width: terminalRadius * 2,
          height: terminalRadius * 2,
          decoration: BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.black,
              width: 1.0,
            ),
          ),
          child: Text(
            terminal.name,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 10,
            ),
          ),
        ),
      ),
    );
  }
}
