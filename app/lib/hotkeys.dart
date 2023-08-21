import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:app/providers/mode_provider.dart';

class HotKeys extends ConsumerStatefulWidget {
  final Widget child;

  const HotKeys({super.key, required this.child});

  @override
  HotKeysState createState() => HotKeysState();
}

class HotKeysState extends ConsumerState<HotKeys> {
  @override
  Widget build(BuildContext context) {
    return Shortcuts(
      shortcuts: {
        LogicalKeySet(LogicalKeyboardKey.escape): EscapeIntent(ref),
        LogicalKeySet(LogicalKeyboardKey.keyW): WIntent(ref),
      },
      child: Actions(
        actions: {
          EscapeIntent: EscapeAction(),
          WIntent: WAction(),
        },
        child: Focus(child: widget.child, autofocus: true),
      ),
    );
  }
}

class EscapeIntent extends Intent {
  final WidgetRef ref;
  const EscapeIntent(WidgetRef this.ref);
}

class EscapeAction extends Action<EscapeIntent> {
  @override
  void invoke(covariant EscapeIntent intent) {
    intent.ref.read(modeProvider.notifier).reset();
  }
}

class WIntent extends Intent {
  final WidgetRef ref;
  const WIntent(WidgetRef this.ref);
}

class WAction extends Action<WIntent> {
  @override
  void invoke(covariant WIntent intent) {
    intent.ref.read(modeProvider.notifier).invertModeState(ModeState.addWire);
  }
}
