import 'package:flutter/material.dart';

import '../models/diagram.dart';

class DiagramChildItem extends StatefulWidget {
  final DiagramItem _dItem;
  final Function _downLevel;
  final BuildContext _diagramAreaContext;

  DiagramChildItem(Key key, this._dItem, this._downLevel, this._diagramAreaContext) :
        super(key: key);

  @override
  _DiagramChildItemState createState() =>
      _DiagramChildItemState(_dItem, _downLevel,_diagramAreaContext);
}

class _DiagramChildItemState extends State<DiagramChildItem> {
  final DiagramItem _dItem;
  final Function _downLevel;
  final BuildContext _diagramAreaContext;

  _DiagramChildItemState(this._dItem, this._downLevel,this._diagramAreaContext);

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
          // move proxy root to this item
          _downLevel(_dItem);
        },
        onSecondaryTap: () {
          final RenderBox overlay = Overlay.of(_diagramAreaContext).context.findRenderObject();
          
          double width = 50;
          double height = 100;
          Rect menuArea = Rect.fromLTWH(_dItem.xPosition, _dItem.yPosition, width, height);
          RelativeRect position = RelativeRect.fromSize(menuArea, overlay.size);
          List<PopupMenuItem> items = [
            PopupMenuItem(child: Text("delete")),
            PopupMenuItem(child: Text("action1")),
            PopupMenuItem(child: Text("action2")),
          ];
          showMenu(context: context, position: position, items: items);
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
