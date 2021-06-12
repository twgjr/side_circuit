import 'package:flutter/material.dart';

import '../models/diagram.dart';

class DiagramChildItem extends StatefulWidget {
  final DiagramItem _dItem;
  final Function _downLevel;

  DiagramChildItem(Key key,this._dItem, this._downLevel):super(key:key);

  @override
  _DiagramChildItemState createState() =>
      _DiagramChildItemState(_dItem, _downLevel);
}

class _DiagramChildItemState extends State<DiagramChildItem> {
  final DiagramItem _dItem;
  final Function _downLevel;

  _DiagramChildItemState(this._dItem, this._downLevel);

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
