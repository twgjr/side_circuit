import 'dart:math';

import '/models/view/data_obj.dart';
import '/widgets/device/device_view.dart';
import '/widgets/node/node_view.dart';
import 'package:app/models/circuit/device.dart';
import 'package:app/models/circuit/node.dart';

import 'package:flutter/material.dart';

class DraggableItem extends StatefulWidget {
  final BuildContext cktViewCtx;
  final DataObj dataObj;

  DraggableItem({
    super.key,
    required this.cktViewCtx,
    required this.dataObj,
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
      left: widget.dataObj.position.x,
      top: widget.dataObj.position.y,
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
            double new_x = widget.dataObj.position.x + localOffset.dx;
            double new_y = widget.dataObj.position.y + localOffset.dy;
            widget.dataObj.position = Point(new_x, new_y);
          });
        },
        feedback: widgetToDisplay(),
        child: Visibility(
          visible: _visible,
          child: widgetToDisplay(),
        ),
      ),
    );
  }

  Widget widgetToDisplay() {
    print(widget.dataObj.obj.runtimeType);
    if (widget.dataObj.obj is Device) {
      return DeviceView(
        device: widget.dataObj.obj,
        cktViewCtx: widget.cktViewCtx,
      );
    } else if (widget.dataObj.obj is Node) {
      return NodeView(node: widget.dataObj.obj, cktViewCtx: widget.cktViewCtx);
    } else {
      throw Exception('Unknown data type');
    }
  }
}
