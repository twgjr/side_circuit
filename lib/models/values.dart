enum ValueState { Upper, Lower, UpperExclusive, LowerExclusive, Value }

///  Dynamic and abstract numbers. Used throughout solver and model
///  Needed to help handle abstract numbers such as infinity
class Value {
  var stored;
  var _state = ValueState.Value;

  Value.dynamic(this.stored);

  Value.empty() : this.stored = null;

  Value.number(num val) : this.stored = val;

  Value.logic(bool val) : this.stored = val;

  Value.negInf()
      : this.stored = -1073741823,
        // dart VM minimum small integer on 32 bit system
        _state = ValueState.Lower;

  Value.posInf()
      : this.stored = 1073741823,
        // dart VM maximum small integer on 32 bit system
        _state = ValueState.Upper;

  Value.upperBound(num val, bool isExclusive) : this.stored = val {
    if (isExclusive) {
      _state = ValueState.UpperExclusive;
    } else {
      _state = ValueState.Upper;
    }
  }

  Value.lowerBound(num val, bool isExclusive) : this.stored = val {
    if (isExclusive) {
      _state = ValueState.UpperExclusive;
    } else {
      _state = ValueState.Upper;
    }
  }

  Value.copyShiftLeft(num shift, Value toCopy)
      : this.stored = toCopy.stored - shift,
        this._state = toCopy._state;

  Value.copy(Value copyValue)
      : this.stored = copyValue.stored,
        this._state = copyValue._state;

  Value.subtract(Value a, Value minusB, bool isUpper) {
    this.stored = a.stored - minusB.stored;
    if(a.isExclusive || minusB.isExclusive){
      if(isUpper){
        this._state = ValueState.UpperExclusive;
      } else {
        this._state = ValueState.LowerExclusive;
      }
    } else {
      if(isUpper){
        this._state = ValueState.Upper;
      } else {
        this._state = ValueState.Lower;
      }
    }
  }

  bool isLogic() => this.stored is bool;

  bool isNumber() => this.stored is num;

  bool isSet() => this.stored != null;

  bool get isExclusive =>
      this._state == ValueState.LowerExclusive ||
      this._state == ValueState.UpperExclusive;

  bool get isNotExclusive => !this.isExclusive;

  bool get isUpper =>
      this._state == ValueState.UpperExclusive ||
          this._state == ValueState.Upper;

  bool get isNotUpper => !this.isNotUpper;

  bool get isLower =>
      this._state == ValueState.LowerExclusive ||
          this._state == ValueState.Lower;

  bool get isNotLower => !this.isLower;

  bool get isValue =>
      this._state == ValueState.Value;

  bool get isNotValue => !this.isValue;

  bool isHigherThan(Value value) {
    if (this.isExclusive == value.isExclusive) {
      print("${this.stored} > ${value.stored}?  ${this.stored > value.stored}");
      return this.stored > value.stored;
    } else {
      print(
          "${this.stored} >= ${value.stored}?  ${this.stored >= value.stored}");
      return this.stored >= value.stored;
    }
  }

  bool isLowerThan(Value value) {
    if (this.isExclusive == value.isExclusive) {
      print("${this.stored} < ${value.stored}?  ${this.stored < value.stored}");
      return this.stored < value.stored;
    } else {
      print(
          "${this.stored} <= ${value.stored}?  ${this.stored <= value.stored}");
      return this.stored <= value.stored;
    }
  }

  bool isSameAs(Value value) {
    return this.stored == value.stored &&
        this._state == value._state;
  }

  String valueString() {
    switch(_state){
      case ValueState.Value:
        {
          return stored.toString();
        }
      case ValueState.UpperExclusive:
        {
          return stored.toString()+")";
        }
      case ValueState.Upper:
        {
          return stored.toString()+"]";
        }
      case ValueState.LowerExclusive:
        {
          return "("+stored.toString();
        }
      case ValueState.Lower:
        {
          return "["+stored.toString();
        }
    }
  }
}
