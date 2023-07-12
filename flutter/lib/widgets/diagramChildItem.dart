import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramArea.dart';

class DiagramChildItem extends StatefulWidget {
  final DiagramItem _dItem;
  final DiagramAreaState diagramAreaState;
  final BuildContext _diagramAreaContext;

  DiagramChildItem(Key key, this._dItem, this.diagramAreaState,
      this._diagramAreaContext)
      : super(key: key);

  @override
  _DiagramChildItemState createState() =>
      _DiagramChildItemState(_dItem, diagramAreaState, _diagramAreaContext);
}

class _DiagramChildItemState extends State<DiagramChildItem> {
  final DiagramItem _dItem;
  final DiagramAreaState diagramAreaState;
  final BuildContext _diagramAreaContext;

  _DiagramChildItemState(this._dItem, this.diagramAreaState,
      this._diagramAreaContext);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: _dItem.yPosition,
      left: _dItem.xPosition,
      child: GestureDetector(
        onPanUpdate: (tapInfo) {
          setState(() {
            _dItem.xPosition += tapInfo.delta.dx;
            _dItem.yPosition += tapInfo.delta.dy;
          });
        },
        onDoubleTap: () {
          diagramAreaState.downLevel(_dItem);
        },
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition);
        },
        child: Container(
          child: Column(
            children: [
              Text("depth:${_dItem.depth()}"),
              Text("breadth:${_dItem.breadth()}"),
              if (_dItem.terminals.isNotEmpty) // check if terminals list is not empty
                Column(
                  children: _dItem.terminals.map((terminal) {
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
    final RenderBox overlay = Overlay.of(_diagramAreaContext)!.context.findRenderObject() as RenderBox;

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
            diagramAreaState.deleteItem(_dItem);
            break;
          case "edit":
            print("Not implemented yet");
            break;
        }
      }
    });
  }
}