import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import '../models/diagram.dart';
import 'diagramArea.dart';

class DiagramChildItem extends StatefulWidget {
  final DiagramItem _dItem;
  final DiagramAreaState diagramAreaState;
  final BuildContext _diagramAreaContext;

  DiagramChildItem(
      Key key, this._dItem, this.diagramAreaState, this._diagramAreaContext)
      : super(key: key);

  @override
  _DiagramChildItemState createState() =>
      _DiagramChildItemState(_dItem, diagramAreaState, _diagramAreaContext);
}

class _DiagramChildItemState extends State<DiagramChildItem> {
  final DiagramItem _dItem;
  final DiagramAreaState diagramAreaState;

  final BuildContext _diagramAreaContext;

  _DiagramChildItemState(
      this._dItem, this.diagramAreaState, this._diagramAreaContext);

  void _showPopupMenu() async {
    final RenderBox overlay =
    Overlay.of(_diagramAreaContext).context.findRenderObject();

    await showMenu(
      context: context,
      position: RelativeRect.fromSize(
          Rect.fromLTWH(_dItem.xPosition, _dItem.yPosition, 50, 100),
          overlay.size),
      items: [
        PopupMenuItem(value: "delete", child: Text("delete")),
      ],
      //elevation: 8.0,
    ).then((value) {
      if (value != null) {
        switch (value) {
          case "delete":
            diagramAreaState.deleteItem(_dItem);
            break;
        }
      }
    });
  }

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
        onSecondaryTap: () {
          _showPopupMenu();
        },
        child: Container(
          child: Column(
            children: [
              Text("depth:${_dItem.depth()}"),
              Text("breadth:${_dItem.breadth()}"),
              //Text(_dItem.equations[0].equationString),
            ],
          ),
        ),
      ),
    );
  }
}
