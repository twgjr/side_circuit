import 'values.dart';

class Range {
  String type;
  Values value;

  Range.include(this.value) {
    this.type = "include";
  }

  Range.exclude(this.value) {
    this.type = "exclude";
  }
}