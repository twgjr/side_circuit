import 'package:flutter/material.dart';

import 'package:app/widgets/device_editor/device_editor_top_bar.dart';
import 'package:app/widgets/device_editor/device_editor_toolbar.dart';
import 'package:app/widgets/device/device_editable.dart';

class DeviceEditor extends StatefulWidget {
  DeviceEditor({super.key});

  @override
  DeviceEditorState createState() => DeviceEditorState();
}

class DeviceEditorState extends State<DeviceEditor> {
  Offset position = Offset(0, 0);
  double width = 400;
  double height = 400;

  void addOffset(Offset position) {
    setState(() {
      this.position += position;
    });
  }

  @override
  void initState() {
    final RenderBox overlay =
        Overlay.of(context).context.findRenderObject() as RenderBox;
    width = overlay.size.width / 2;
    height = overlay.size.height / 2;
    position = Offset(width / 2, height / 2);
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        child: Stack(
          children: [
            Positioned(
              left: position.dx,
              top: position.dy,
              child: Card(
                clipBehavior: Clip.hardEdge,
                elevation: 10,
                shadowColor: Colors.black,
                child: Container(
                  decoration: BoxDecoration(
                    color: Theme.of(context).primaryColorDark,
                    border: Border.all(
                      color: Theme.of(context).primaryColorDark,
                      width: 1.0,
                    ),
                  ),
                  width: width,
                  height: height,
                  child: Column(
                    children: [
                      DeviceEditorTopBar(onDrag: addOffset),
                      DeviceEditorToolbar(),
                      Expanded(child: Center(child: DeviceEditable())),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
