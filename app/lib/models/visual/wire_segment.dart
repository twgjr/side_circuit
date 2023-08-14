import 'package:flutter/material.dart';

import 'package:app/models/visual/wire.dart';
import 'package:app/models/visual/end_point.dart';

class WireSegment {
  final Wire wire;
  final EndPoint endPoint;

  WireSegment({required this.wire, required this.endPoint});

  int get _index => wire.segments.indexOf(this);

  bool _isFirst() => _index == 0;

  Offset start() {
    if (_isFirst()) {
      print(
          'WireSegment.start: terminal global =${wire.terminal!.globalCenter()}');
      return wire.terminal!.globalCenter();
    } else {
      print(
          'WireSegment.start: last endpoint =${wire.segments[_index - 1].endPoint.position}');
      return wire.segments[_index - 1].endPoint.position;
    }
  }

  Offset end() {
    return endPoint.position;
  }
}
