import 'package:flutter_riverpod/flutter_riverpod.dart';

enum ModeState {
  addWire,
}

class ModeNotifier extends StateNotifier<Mode> {
  ModeNotifier() : super(Mode());

  void invertModeState(ModeState gestureState) {
    Mode newState = state.copy();
    newState.invert(gestureState);
    state = newState;
  }

  void reset() {
    state = Mode();
  }
}

final modeProvider = StateNotifierProvider<ModeNotifier, Mode>(
  (ref) => ModeNotifier(),
);

class Mode {
  bool addWire;

  Mode({
    this.addWire = false,
  });

  Mode copy() {
    return Mode(
      addWire: this.addWire,
    );
  }

  void invert(ModeState gestureState) {
    switch (gestureState) {
      case ModeState.addWire:
        addWire = !addWire;
        break;
    }
  }
}
