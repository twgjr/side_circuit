import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class Selection {
  final Set<Widget> _selectedWidgets = {};

  void toggleSelection(Widget widget) {
    if (_selectedWidgets.contains(widget)) {
      _selectedWidgets.remove(widget);
    } else {
      _selectedWidgets.add(widget);
    }
  }

  void applyChanges(Function(Widget) changeFunction) {
    _selectedWidgets.forEach(changeFunction);
  }
}

class SelectionNotifier extends StateNotifier<Selection> {
  SelectionNotifier() : super(Selection());

  void toggleSelection(Widget widget) {
    state.toggleSelection(widget);
    state = state; // trigger state update
  }

  void applyChanges(Function(Widget) changeFunction) {
    state.applyChanges(changeFunction);
  }
}

final selectionProvider = StateNotifierProvider<SelectionNotifier, Selection>(
  (ref) => SelectionNotifier(),
);
