import 'package:flutter/material.dart';

import 'package:app/models/visual/symbol.dart';

class DraggableItem extends StatefulWidget {
  final Widget child;
  final Symbol symbol;

  DraggableItem({super.key, required this.child, required this.symbol});

  @override
  _DraggableItemState createState() => _DraggableItemState();
}

class _DraggableItemState extends State<DraggableItem> {
  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: widget.symbol.position.dx,
      top: widget.symbol.position.dy,
      child: GestureDetector(
        onPanUpdate: (details) {
          setState(() {
            Offset new_local = widget.symbol.position + details.delta;
            widget.symbol.position = new_local;
          });
        },
        child: widget.child,
      ),
    );
  }
}
