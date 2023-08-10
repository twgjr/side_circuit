import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class OverlayEntryNotifier extends StateNotifier<OverlayEntry> {
  OverlayEntryNotifier()
      : super(
          OverlayEntry(
            builder: (context) => Container(),
          ),
        ); //dummy OverylayEntry initially

  void update(OverlayEntry entry) {
    state = entry;
  }
}

final deviceEditOverlayEntryProvider =
    StateNotifierProvider<OverlayEntryNotifier, OverlayEntry>(
  (ref) => OverlayEntryNotifier(),
);
