import 'expression.dart';
import 'values.dart';

class ValueRange {
  //Expression expr; // expression related to this range
  Boundary upper;
  Boundary lower;

  ValueRange(this.lower, this.upper);

  ValueRange.target(Values target)
      : this.upper = Boundary.includes(target),
        this.lower = Boundary.includes(target);

  ValueRange.num()
      : this.upper = Boundary.posInf(),
        this.lower = Boundary.negInf();

  ValueRange.logic()
      : this.upper = Boundary.logicHigh(),
        this.lower = Boundary.logicLow();

  void setRangeTo(Values target) {
    this.upper = Boundary.includes(target);
    this.lower = Boundary.includes(target);
  }

  Values mid() {
    var upperVal = upper.value.value;
    var lowerVal = lower.value.value;
    return Values.number((upperVal + lowerVal) / 2);
  }

  Values min() {
    return lower.value;
  }

  Values max() {
    return upper.value;
  }

  num rangeWidth() {
    return upper.value.value - lower.value.value;
  }

  void setUpper(Boundary newUpper) {
    if (newUpper.isExclusive() && this.upper.isExclusive()) {
      if (newUpper.value.value < this.upper.value.value) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isInclusive() && this.upper.isInclusive()) {
      if (newUpper.value.value < this.upper.value.value) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isExclusive() && this.upper.isInclusive()) {
      if (newUpper.value.value <= this.upper.value.value) {
        this.upper = newUpper;
      }
    }
    if (newUpper.isInclusive() && this.upper.isExclusive()) {
      if (newUpper.value.value < this.upper.value.value) {
        this.upper = newUpper;
      }
    }
  }

  void setLower(Boundary newLower) {
    if (newLower.isExclusive() && this.lower.isExclusive()) {
      if (newLower.value.value > this.lower.value.value) {
        this.lower = newLower;
      }
    }
    if (newLower.isInclusive() && this.lower.isInclusive()) {
      if (newLower.value.value > this.lower.value.value) {
        this.lower = newLower;
      }
    }
    if (newLower.isExclusive() && this.lower.isInclusive()) {
      if (newLower.value.value >= this.lower.value.value) {
        this.lower = newLower;
      }
    }
    if (newLower.isInclusive() && this.lower.isExclusive()) {
      if (newLower.value.value > this.lower.value.value) {
        this.lower = newLower;
      }
    }
  }

  bool contains(var value) {
    if (value is bool) {
      return value;
    }

    // otherwise value is num
    if (lower.isInclusive() && upper.isInclusive()) {
      if ((lower.value.value <= value) && (value <= upper.value.value)) {
        return true;
      }
    }
    if (lower.isInclusive() && upper.isExclusive()) {
      if ((lower.value.value <= value) && (value < upper.value.value)) {
        return true;
      }
    }
    if (lower.isExclusive() && upper.isInclusive()) {
      if ((lower.value.value < value) && (value <= upper.value.value)) {
        return true;
      }
    }
    if (lower.isExclusive() && upper.isExclusive()) {
      if ((lower.value.value < value) && (value < upper.value.value)) {
        return true;
      }
    }
    return false;
  }
}

class Boundary {
  int type = 0;
  Values value;

  Boundary(this.value, this.type);

  Boundary.logicLow()
      : this.type = 0,
        this.value = Values.logic(false);

  Boundary.logicHigh()
      : this.type = 1,
        this.value = Values.logic(true);

  Boundary.includes(this.value) : this.type = 2;

  Boundary.excludes(this.value) : this.type = 3;

  Boundary.negInf()
      : this.value = Values.negInf(),
        this.type = 2;

  Boundary.posInf() :
    this.value = Values.posInf(),
    this.type = 2;

  void includes(Values value) {
    this.type = 2;
    this.value = value;
  }

  void excludes(Values value) {
    this.type = 3;
    this.value = value;
  }

  bool isLogicLow() => this.type == 0;

  bool isLogicHigh() => this.type == 1;

  bool isInclusive() => this.type == 2;

  bool isExclusive() => this.type == 3;
}
