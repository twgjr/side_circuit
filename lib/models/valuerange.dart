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

  ValueRange.copy(ValueRange copy)
      : this.upper = Boundary.copy(copy.upper),
        this.lower = Boundary.copy(copy.lower);

  ValueRange.splitLeft(ValueRange rangeToSplit)
      : this.upper = Boundary.copy(rangeToSplit.upper),
        this.lower = Boundary.copy(rangeToSplit.lower){
    this.upper = this.mid();
  }

  ValueRange.splitRight(ValueRange rangeToSplit)
      : this.upper = Boundary.copy(rangeToSplit.upper),
        this.lower = Boundary.copy(rangeToSplit.lower){
    this.lower = this.mid();
  }

  void setRangeTo(Values target) {
    this.upper = Boundary.includes(target);
    this.lower = Boundary.includes(target);
  }

  Boundary mid() {
    var upperVal = upper.value.value;
    var lowerVal = lower.value.value;
    return Boundary.includes(Values.number((upperVal + lowerVal) / 2));
  }

  Boundary min() {
    return lower;
  }

  Boundary max() {
    return upper;
  }

  num rangeWidth() {
    return upper.value.value - lower.value.value;
  }

  bool isLogic() {
    return this.upper.hasLogicValue();
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

  Boundary.posInf()
      : this.value = Values.posInf(),
        this.type = 2;

  Boundary.copy(Boundary copy)
      : this.type = copy.type,
        this.value = Values.copy(copy.value);

  void includes(Values value) {
    this.type = 2;
    this.value = value;
  }

  void excludes(Values value) {
    this.type = 3;
    this.value = value;
  }

  bool hasLogicValue() => this.value.isLogic();

  bool isInclusive() => this.type == 2;

  bool isExclusive() => this.type == 3;
}
