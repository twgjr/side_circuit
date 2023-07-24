import 'package:flutter/material.dart';
import '../../../models/diagram.dart';

class DiagramChildItem extends StatefulWidget {
  final DiagramItem dItem;
  final void Function(DiagramItem) downLevel;
  final void Function(DiagramItem) deleteItem;
  final BuildContext diagramAreaContext;

  DiagramChildItem(
      {super.key,
      required this.dItem,
      required this.downLevel,
      required this.deleteItem,
      required this.diagramAreaContext});

  @override
  _DiagramChildItemState createState() => _DiagramChildItemState();
}

class _DiagramChildItemState extends State<DiagramChildItem> {
  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: widget.dItem.yPosition,
      left: widget.dItem.xPosition,
      child: GestureDetector(
        onPanUpdate: (tapInfo) {
          setState(() {
            widget.dItem.xPosition += tapInfo.delta.dx;
            widget.dItem.yPosition += tapInfo.delta.dy;
          });
        },
        onDoubleTap: () {
          widget.downLevel(widget.dItem);
        },
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition);
        },
        child: Container(
          child: Column(
            children: [
              Text("depth:${widget.dItem.depth()}"),
              Text("breadth:${widget.dItem.breadth()}"),
              if (widget.dItem.terminals
                  .isNotEmpty) // check if terminals list is not empty
                Column(
                  children: widget.dItem.terminals.map((terminal) {
                    return Text("terminal: (${terminal.x}, ${terminal.y})");
                  }).toList(),
                ),
            ],
          ),
        ),
      ),
    );
  }

  void _showPopupMenu(Offset position) {
    final RenderBox overlay = Overlay.of(widget.diagramAreaContext)
        .context
        .findRenderObject() as RenderBox;

    final relativePosition = RelativeRect.fromSize(
      Rect.fromLTWH(position.dx, position.dy, 0, 0),
      overlay.size,
    );

    showMenu(
      context: context,
      position: relativePosition,
      items: [
        PopupMenuItem(value: "edit", child: Text("Edit")),
        PopupMenuItem(value: "delete", child: Text("Delete")),
      ],
    ).then((value) {
      if (value != null) {
        switch (value) {
          case "delete":
            widget.deleteItem(widget.dItem);
            break;
          case "edit":
            print("Not implemented yet");
            break;
        }
      }
    });
  }
}
