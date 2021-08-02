import 'package:universal_html/html.dart';

///  Dynamic and abstract numbers. Used throughout solver and model
///  Needed to help handle abstract numbers such as infinity
class Value {
  var stored;
  bool isUpper;
  bool isExclusive;

  Value.dynamic(this.stored)
      : this.isExclusive = false,
        this.isUpper = false;

  Value.empty()
      : this.stored = null,
        this.isExclusive = false,
        this.isUpper = false;

  Value.number(num val)
      : this.stored = val,
        this.isUpper = false,
        this.isExclusive = false;

  Value.logic(bool val)
      : this.stored = val,
        this.isUpper = false,
        this.isExclusive = false;

  Value.negInf()
      : this.stored = -1073741823,
        // dart VM minimum small integer on 32 bit system
        this.isUpper = false,
        this.isExclusive = false;

  Value.posInf()
      : this.stored = 1073741823,
        // dart VM maximum small integer on 32 bit system
        this.isUpper = true,
        this.isExclusive = false;

  Value.upperBound(num val, this.isExclusive)
      : this.stored = val,
        this.isUpper = true;

  Value.lowerBound(num val, this.isExclusive)
      : this.stored = val,
        this.isUpper = false;

  Value.copyShiftLeft(num shift, Value toCopy)
      : this.stored = toCopy.stored - shift,
        this.isUpper = true,
        this.isExclusive = toCopy.isExclusive;

  Value.copyShiftRight(num shift, Value toCopy)
      : this.stored = toCopy.stored + shift,
        this.isUpper = true,
        this.isExclusive = toCopy.isExclusive;

  Value.copy(Value copyValue)
      : this.stored = copyValue.stored,
        this.isUpper = copyValue.isUpper,
        this.isExclusive = copyValue.isExclusive;

  Value.subtract(Value a, Value minusB, this.isUpper)
      : this.isExclusive = false
  {
    this.stored = a.stored - minusB.stored;
    this.isExclusive = a.isExclusive || minusB.isExclusive;
  }


  bool isLogic() => this.stored is bool;

  bool isNumber() => this.stored is num;

  bool isSet() => this.stored != null;

  bool get isNotExclusive => !this.isExclusive;

  bool isZeroExclusive() => this.isExclusive && this.stored == 0;

  bool isHigherThan(Value value) {
    if (this.isExclusive == value.isExclusive) {
      print("${this.stored} > ${value.stored}?  ${this.stored > value.stored}");
        return this.stored > value.stored;
    } else {
      print("${this.stored} >= ${value.stored}?  ${this.stored >= value.stored}");
        return this.stored >= value.stored;
    }
  }

  bool isLowerThan(Value value) {
    if (this.isExclusive == value.isExclusive) {
      print("${this.stored} < ${value.stored}?  ${this.stored < value.stored}");
      return this.stored < value.stored;
    } else {
      print("${this.stored} <= ${value.stored}?  ${this.stored <= value.stored}");
      return this.stored <= value.stored;
    }
  }

  bool isSameAs(Value value) {
    return this.isExclusive == value.isExclusive &&
        this.stored == value.stored &&
        this.isUpper == value.isUpper;
  }
}
