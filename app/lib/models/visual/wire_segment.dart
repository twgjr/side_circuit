import 'package:app/models/visual/symbol.dart';
import 'package:flutter/material.dart';

import 'package:app/models/visual/wire.dart';
import 'package:app/models/visual/end_point.dart';

class WireSegment {
  final Wire wire;
  final EndPoint endPoint;
  final Symbol symbol = Symbol();

  WireSegment({required this.wire, required this.endPoint});

  int get index => wire.segments.indexOf(this);

  bool _isFirst() => index == 0;

  Offset start() {
    if (_isFirst()) {
      return wire.terminal!.diagramPosition();
    } else {
      return wire.segments[index - 1].endPoint.position;
    }
  }

  Offset end() {
    return endPoint.position;
  }
}
