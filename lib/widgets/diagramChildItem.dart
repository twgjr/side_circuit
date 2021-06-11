import 'package:flutter/material.dart';

import '../models/diagram.dart';

class DiagramChildItem extends StatefulWidget {
  final DiagramItem dItem;

  DiagramChildItem(this.dItem);

  @override
  State<StatefulWidget> createState() {
    return _DiagramChildItemState(this.dItem);
  }
}

class _DiagramChildItemState extends State<DiagramChildItem> {
  final DiagramItem dItem;

  _DiagramChildItemState(this.dItem);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: dItem.yPosition,
      left: dItem.xPosition,
      child: GestureDetector(
        onPanUpdate: (tapInfo) {
          setState(() {
            dItem.xPosition += tapInfo.delta.dx;
            dItem.yPosition += tapInfo.delta.dy;
          });
        },
        child: Container(
          child: Text(dItem.equations[0].equationString),
        ),
      ),
    );
  }
}
