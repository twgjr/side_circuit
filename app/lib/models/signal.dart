enum SignalClass { PULSE, SIN }

class SignalFunction {
  late String kind;

  SignalFunction(SignalClass kind) {
    this.kind = kind.toString().split('.').last;
  }

  String toSpice() {
    List<String> args = [];
    this.toMap().forEach((key, value) {
      args.add('$key $value');
    });
    return args.join(' ');
  }

  Map<String, dynamic> toMap() {
    return {'kind': kind};
  }
}

class Sin extends SignalFunction {
  late double V0;
  late double VA;
  late double FREQ;
  late double TD;
  late double THETA;
  late double PHASE;

  Sin(double dc, double ampl, double freq,
      {double delay = 0.0, double decay = 0.0, double phase = 0.0})
      : super(SignalClass.SIN) {
    this.V0 = dc;
    this.VA = ampl;
    this.FREQ = freq;
    this.TD = delay;
    this.THETA = decay;
    this.PHASE = phase;
  }

  @override
  Map<String, dynamic> toMap() {
    var map = super.toMap();
    map.addAll({
      'V0': V0,
      'VA': VA,
      'FREQ': FREQ,
      'TD': TD,
      'THETA': THETA,
      'PHASE': PHASE,
    });
    return map;
  }
}

class Pulse extends SignalFunction {
  late double V1;
  late double V2;
  late double TD;
  late double TR;
  late double TF;
  late double PW;
  late double PER;
  late int NP;

  Pulse(double val1, double val2, double freq,
      {double delay = 0.0,
      double? riseTime,
      double? fallTime,
      double dutyRatio = 0.5,
      int numPulses = 0})
      : super(SignalClass.PULSE) {
    this.V1 = val1;
    this.V2 = val2;
    this.TD = delay;
    this.TR = riseTime ?? (fallTime != null ? fallTime : (PER / 1000));
    this.TF = fallTime ?? (riseTime != null ? riseTime : (PER / 1000));
    var period = 1 / freq;
    this.PW = period * dutyRatio;
    this.PER = period;
    this.NP = numPulses;
  }

  @override
  Map<String, dynamic> toMap() {
    var map = super.toMap();
    map.addAll({
      'V1': V1,
      'V2': V2,
      'TD': TD,
      'TR': TR,
      'TF': TF,
      'PW': PW,
      'PER': PER,
      'NP': NP,
    });
    return map;
  }
}
