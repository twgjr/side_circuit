import 'package:universal_html/html.dart';

/// needed to prevent the default context menu from popping up in web
/// wrapped in a class to prevent errors if directly added to main()
class PreventDefault{
  PreventDefault() {
    window.document.onContextMenu.listen((evt) => evt.preventDefault());
  }
}