import 'package:app/models/view/data_obj.dart';
import 'package:flutter/material.dart';

import 'package:app/widgets/diagram/draggable_item.dart';

import 'package:app/models/circuit/circuit.dart';

class Diagram extends StatefulWidget {
  final Circuit circuit;
  final List<DataObj> dataObjs;
  Diagram({
    super.key,
    required this.circuit,
    required this.dataObjs,
  });

  @override
  _DiagramState createState() => _DiagramState();
}

class _DiagramState extends State<Diagram> {
  @override
  Widget build(BuildContext context) {
    ;
    return Stack(
      children: [
        for (DataObj dataObj in widget.dataObjs)
          DraggableItem(
            dataObj: dataObj,
            cktViewCtx: context,
          ),
      ],
    );
  }
}
