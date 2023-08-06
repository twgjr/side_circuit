import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final overlayEntryProvider =
    StateNotifierProvider<OverlayEntryNotifier, List<OverlayEntry>>((ref) {
  return OverlayEntryNotifier();
});

class OverlayEntryNotifier extends StateNotifier<List<OverlayEntry>> {
  OverlayEntryNotifier() : super([]);

  void addOverlay(OverlayEntry overlay) {
    state = [
      ...state,
      overlay,
    ];
  }

  void removeOverlay(OverlayEntry overlay) {
    overlay.remove();
    state = state.where((oE) => oE != overlay).toList();
  }

  void removeLastOverlay() {
    if (state.isNotEmpty) {
      state.last.remove();
      state = state.sublist(0, state.length - 1);
    }
  }
}
