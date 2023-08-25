import 'package:app/widgets/symbol/symbol_widget.dart';
import 'package:flutter/material.dart';

import 'package:app/models/terminal.dart';

class TerminalWidget extends StatelessWidget {
  final Terminal terminal;
  final bool editable;

  TerminalWidget({super.key, required this.terminal, required this.editable});

  SymbolWidget _selectWidget() {
    if (terminal.vertex != null) {
      terminal.symbol.fillColor = Colors.black;
      return SymbolWidget(terminal.symbol);
    } else {
      terminal.symbol.fillColor = Colors.white;
      return SymbolWidget(terminal.symbol);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        _selectWidget(),
      ],
    );
  }
}
