import 'dart:math' as math;
import 'package:flutter/material.dart';

import 'package:app/models/diagram_symbol.dart';
import 'package:app/widgets/symbol/shape_widget.dart';

class SymbolWidget extends StatelessWidget {
  final DiagramSymbol symbol;
  SymbolWidget(this.symbol, {super.key});

  @override
  Widget build(BuildContext context) {
    return Transform(
      transform: Matrix4.rotationZ(symbol.rotation * math.pi / 180),
      child: ShapeWidget(
        shape: symbol.shape,
      ),
    );
  }
}
