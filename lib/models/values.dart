enum ValueState { Upper, Lower, UpperExclusive, LowerExclusive, Value }

///  Dynamic and abstract numbers. Used throughout solver and model
///  Needed to help handle abstract numbers such as infinity
class Value {
  var _stored;
  var _state = ValueState.Value;

  Value(var value, this._state) {
    this.stored = value;
  }

  Value.dynamic(var value) {
    this.stored = value;
  }

  Value.empty();

  Value.number(num value) {
    this.stored = value;
  }

  Value.logic(bool value) {
    this.stored = value;
  }

  Value.logicLowBound() : this._state = ValueState.Lower {
    this.stored = false;
  }

  Value.logicHighBound() : this._state = ValueState.Upper {
    this.stored = true;
  }

  Value.negInf() : _state = ValueState.Lower {
    this.stored = -1073741823; // dart VM minimum small integer on 32 bit system
  }

  Value.posInf() : _state = ValueState.Upper {
    this.stored = 1073741823; // dart VM maximum small integer on 32 bit system
  }

  Value.upperBound(num value, bool isExclusive) {
    this.stored = value;
    if (isExclusive) {
      _state = ValueState.UpperExclusive;
    } else {
      _state = ValueState.Upper;
    }
  }

  Value.lowerBound(num value, bool isExclusive) {
    this.stored = value;
    if (isExclusive) {
      _state = ValueState.LowerExclusive;
    } else {
      _state = ValueState.Lower;
    }
  }

  Value.copyShiftLeft(num shift, Value toCopy) : this._state = toCopy._state {
    this.stored = toCopy.stored - shift;
  }

  Value.copy(Value copyValue) : this._state = copyValue._state {
    this.stored = copyValue.stored;
  }

  Value.subtract(Value a, Value minusB, bool isUpper) {
    this.stored = a.stored - minusB.stored;
    if (a.isExclusive || minusB.isExclusive) {
      if (isUpper) {
        this._state = ValueState.UpperExclusive;
      } else {
        this._state = ValueState.LowerExclusive;
      }
    } else {
      if (isUpper) {
        this._state = ValueState.Upper;
      } else {
        this._state = ValueState.Lower;
      }
    }
  }

  set stored(var value) {
    assert((value is bool) || (value is num));
    _stored = value;
  }

  dynamic get stored {
    return _stored;
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

  bool get isValue => this._state == ValueState.Value;

  bool get isNotValue => !this.isValue;

  bool get isBoundary {
    return (this.isUpper || this.isLower);
  }

  bool get isNotBoundary {
    return !this.isBoundary;
  }

  /// for comparing boundaries
  bool isAbove(Value boundary) {
    assert(
        this.isBoundary && boundary.isBoundary, "must compare two boundaries");
    if (this.isNotExclusive && boundary.isExclusive) {
      return this.stored >= boundary.stored;
    } else {
      return this.stored > boundary.stored;
    }
  }

  /// for comparing boundaries
  bool isBelow(Value boundary) {
    assert(
        this.isBoundary && boundary.isBoundary, "must compare two boundaries");
    if (this.isExclusive && boundary.isNotExclusive) {
      return this.stored <= boundary.stored;
    } else {
      return this.stored < boundary.stored;
    }
  }

  bool isSameAs(Value value) {
    return this.stored == value.stored && this._state == value._state;
  }

  bool boundaryContains(Value value) {
    assert(
        this.isBoundary && value.isValue, "must compare a boundary to a value");
    if (this.stored is num) {
      switch (this._state) {
        case ValueState.Upper:
          return (value <= this).stored;
        case ValueState.UpperExclusive:
          return (value < this).stored;
        case ValueState.Lower:
          return (this <= value).stored;
        case ValueState.LowerExclusive:
          return (this < value).stored;
        case ValueState.Value:
          assert(false, "invalid boundary state");
          break;
      }
    } else {
      // is bool
      assert(this.stored is bool, "not a bool");
      switch (this._state) {
        case ValueState.Upper:
        case ValueState.Lower:
          return value.stored == this.stored;
        case ValueState.UpperExclusive:
        case ValueState.LowerExclusive:
        case ValueState.Value:
          assert(false, "invalid boundary state");
          break;
      }
    }
    return false;
  }

  String toString() {
    switch (_state) {
      case ValueState.Value:
        {
          return stored.toString();
        }
      case ValueState.UpperExclusive:
        {
          return stored.toString() + ")";
        }
      case ValueState.Upper:
        {
          return stored.toString() + "]";
        }
      case ValueState.LowerExclusive:
        {
          return "(" + stored.toString();
        }
      case ValueState.Lower:
        {
          return "[" + stored.toString();
        }
    }
  }

  Value operator +(Value value) {
    assert((this.stored is num) && (value.stored is num),
        "+ operator args not num");
    return Value.number(this.stored + value.stored);
  }

  Value operator -(Value value) {
    assert((this.stored is num) && (value.stored is num),
        "- operator args not num");
    return Value.number(this.stored - value.stored);
  }

  Value operator *(Value value) {
    assert((this.stored is num) && (value.stored is num),
        "* operator args not num");
    return Value.number(this.stored * value.stored);
  }

  Value operator /(Value value) {
    assert((this.stored is num) && (value.stored is num),
        "/ operator args not num");
    return Value.number(this.stored / value.stored);
  }

  Value and(Value value) {
    assert((this.stored is bool) && (value.stored is bool),
        "and operator args not bool");
    return Value.logic(this.stored && value.stored);
  }

  Value or(Value value) {
    assert((this.stored is bool) && (value.stored is bool),
        "or operator args not bool");
    return Value.logic(this.stored || value.stored);
  }

  Value equals(Value value) {
    assert(
        ((this.stored is bool) && (value.stored is bool)) ||
            (this.stored is num) && (value.stored is num),
        "equals operator args not same type");
    return Value.logic(this.stored == value.stored);
  }

  Value operator <(Value value) {
    assert((this.stored is num) && (value.stored is num),
        "< operator args not num");
    return Value.logic(this.stored < value.stored);
  }

  Value operator >(Value value) {
    assert((this.stored is num) && (value.stored is num),
        "> operator args not num");
    return Value.logic(this.stored > value.stored);
  }

  Value operator <=(Value value) {
    assert((this.stored is num) && (value.stored is num),
        "<= operator args not num");
    return Value.logic(this.stored <= value.stored);
  }

  Value operator >=(Value value) {
    assert((this.stored is num) && (value.stored is num),
        ">= operator args not num");
    return Value.logic(this.stored >= value.stored);
  }
}
