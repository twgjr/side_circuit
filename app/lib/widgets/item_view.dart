import 'package:app/models/circuit/device.dart';
import 'package:flutter/material.dart';

class ItemView extends StatefulWidget {
  final Device device;
  final void Function(Device) onDeleteDevice;
  final BuildContext cktViewCtx;

  ItemView(
      {super.key,
      required this.device,
      required this.onDeleteDevice,
      required this.cktViewCtx});

  @override
  _ItemViewState createState() => _ItemViewState();
}

class _ItemViewState extends State<ItemView> {
  bool _visible = true;

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: widget.device.x,
      top: widget.device.y,
      child: GestureDetector(
        onSecondaryTapDown: (details) {
          _showPopupMenu(details.globalPosition);
        },
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
              widget.device.x += localOffset.dx;
              widget.device.y += localOffset.dy;
            });
          },
          feedback: StaticItem(
            Name: widget.device.kind.name,
            Id: widget.device.id,
          ),
          child: Visibility(
            visible: _visible,
            child: StaticItem(
              Name: widget.device.kind.name,
              Id: widget.device.id,
            ),
          ),
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

class StaticItem extends StatelessWidget {
  final String Name;
  final int Id;

  StaticItem({
    required this.Name,
    required this.Id,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            Text('${Name}${Id}'),
          ],
        ),
      ),
    );
  }
}
