import 'dart:math';
import 'package:flutter/material.dart';

import '../../models/view/visual.dart';

class DraggableItem extends StatefulWidget {
  final Widget child;
  final Visual visual;

  DraggableItem({
    super.key,
    required this.child,
    required this.visual,
  });

  @override
  _DraggableItemState createState() => _DraggableItemState();
}

class _DraggableItemState extends State<DraggableItem> {
  bool _visible = true;

  @override
  Widget build(BuildContext context) {
    print('build draggable item');

    return Positioned(
      left: widget.visual.startPosition.x,
      top: widget.visual.startPosition.y,
      child: Draggable(
        dragAnchorStrategy: childDragAnchorStrategy,
        onDragStarted: () {
          setState(() {
            _visible = false;
          });
        },
        onDragEnd: (details) {
          setState(() {
            final RenderBox box = context.findRenderObject() as RenderBox;
            final Offset localOffset = box.globalToLocal(details.offset);
            _visible = true;
            double new_x = widget.visual.startPosition.x + localOffset.dx;
            double new_y = widget.visual.startPosition.y + localOffset.dy;
            widget.visual.startPosition = Point(new_x, new_y);
          });
        },
        feedback: widget.child,
        child: Visibility(
          visible: _visible,
          child: widget.child,
        ),
      ),
    );
  }
}
