import 'package:flutter/material.dart';

import 'package:app/models/visual/visual.dart';

class DraggableItem extends StatefulWidget {
  final Widget child;
  final Visual visual;

  DraggableItem({super.key, required this.child, required this.visual});

  @override
  _DraggableItemState createState() => _DraggableItemState();
}

class _DraggableItemState extends State<DraggableItem> {
  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: widget.visual.position.dx,
      top: widget.visual.position.dy,
      child: GestureDetector(
        onPanUpdate: (details) {
          setState(() {
            Offset new_local = widget.visual.position + details.delta;
            widget.visual.position = new_local;
          });
        },
        child: widget.child,
      ),
    );
  }
}
