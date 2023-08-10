import 'dart:math';
import 'package:flutter/material.dart';

import 'package:app/models/view/visual.dart';

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
      left: widget.visual.position.x,
      top: widget.visual.position.y,
      child: GestureDetector(
        onPanUpdate: (details) {
          setState(() {
            double new_x = widget.visual.position.x + details.delta.dx;
            double new_y = widget.visual.position.y + details.delta.dy;
            widget.visual.position = Point(new_x, new_y);
          });
        },
        child: widget.child,
      ),
    );
  }
}
