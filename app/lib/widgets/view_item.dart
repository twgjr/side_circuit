import 'package:flutter/material.dart';
import 'package:app/models/circuit/device.dart';

class ViewItem extends StatefulWidget {
  final Device device;
  final void Function(Device) onDeleteDevice;
  final BuildContext cktViewCtx;

  ViewItem(
      {super.key,
      required this.device,
      required this.onDeleteDevice,
      required this.cktViewCtx});

  @override
  _ViewItemState createState() => _ViewItemState();
}

class _ViewItemState extends State<ViewItem> {
  double xPosition = 0;
  double yPosition = 0;

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: yPosition,
      left: xPosition,
      child: GestureDetector(
        onPanUpdate: (tapInfo) {
          setState(() {
            xPosition += tapInfo.delta.dx;
            yPosition += tapInfo.delta.dy;
          });
        },
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition);
        },
        child: Card(
          child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(widget.device.kind!.name)),
        ),
      ),
    );
  }

  void _showPopupMenu(Offset position) {
    final RenderBox overlay =
        Overlay.of(widget.cktViewCtx).context.findRenderObject() as RenderBox;

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
            widget.onDeleteDevice(widget.device);
            break;
          case "edit":
            print("Not implemented yet");
            break;
        }
      }
    });
  }
}
