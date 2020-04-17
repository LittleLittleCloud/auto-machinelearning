// <copyright file="Parameters.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace MLNet.Sweeper
{
    public delegate void SignatureSweeperParameter();

    public abstract class BaseParamOptions
    {
        public string Name;
    }

    public abstract class NumericParamOptions : BaseParamOptions
    {
        public int NumSteps = 100;

        public Double? StepSize = null;

        public bool LogBase = false;
    }

    public class FloatParamOptions : NumericParamOptions
    {
        public float Min;

        public float Max;
    }

    public class LongParamOptions : NumericParamOptions
    {
        public long Min;

        public long Max;
    }

    public class DiscreteParamOptions : BaseParamOptions
    {
        public string[] Values = null;
    }

    public sealed class LongParameterValue : IParameterValue<long>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly long _value;

        public string Name
        {
            get { return _name; }
        }

        public string ValueText
        {
            get { return _valueText; }
        }

        public long Value
        {
            get { return _value; }
        }

        public object RawValue => _value;

        public LongParameterValue(string name, long value)
        {
            _name = name;
            _value = value;
            _valueText = _value.ToString("D");
        }

        public bool Equals(IParameterValue other)
        {
            return Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var lpv = obj as LongParameterValue;
            return lpv != null && Name == lpv.Name && _value == lpv._value;
        }
    }

    public sealed class FloatParameterValue : IParameterValue<float>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly float _value;

        public string Name
        {
            get { return _name; }
        }

        public string ValueText
        {
            get { return _valueText; }
        }

        public float Value
        {
            get { return _value; }
        }

        public object RawValue => _value;

        public FloatParameterValue(string name, float value)
        {
            _name = name;
            _value = value;
            _valueText = _value.ToString("R");
        }

        public bool Equals(IParameterValue other)
        {
            return Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var fpv = obj as FloatParameterValue;
            return fpv != null && Name == fpv.Name && _value == fpv._value;
        }
    }

    public sealed class StringParameterValue : IParameterValue<string>
    {
        private readonly string _name;
        private readonly string _value;

        public string Name
        {
            get { return _name; }
        }

        public string ValueText
        {
            get { return _value; }
        }

        public string Value
        {
            get { return _value; }
        }

        public object RawValue => _value;

        public StringParameterValue(string name, string value)
        {
            _name = name;
            _value = value;
        }

        public bool Equals(IParameterValue other)
        {
            return Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var spv = obj as StringParameterValue;
            return spv != null && Name == spv.Name && ValueText == spv.ValueText;
        }
    }

    public interface INumericValueGenerator : IValueGenerator
    {
        float NormalizeValue(IParameterValue value);

        bool InRange(IParameterValue value);
    }

    /// <summary>
    /// The integer type parameter sweep.
    /// </summary>
    public class LongValueGenerator : INumericValueGenerator
    {
        private readonly LongParamOptions _options;
        private IParameterValue[] _gridValues;

        public string Name { get { return _options.Name; } }

        public LongValueGenerator(LongParamOptions options)
        {
            _options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            long val;
            if (_options.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !_options.StepSize.HasValue
                    ? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1))
                    : _options.StepSize.Value;
                var logMax = Math.Log(_options.Max, logBase);
                var logMin = Math.Log(_options.Min, logBase);
                val = (long)(_options.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
                val = (long)(_options.Min + normalizedValue * (_options.Max - _options.Min));

            return new LongParameterValue(_options.Name, val);
        }

        private void EnsureParameterValues()
        {
            if (_gridValues != null)
                return;

            var result = new List<IParameterValue>();
            if ((_options.StepSize == null && _options.NumSteps > (_options.Max - _options.Min)) ||
                (_options.StepSize != null && _options.StepSize <= 1))
            {
                for (long i = _options.Min; i <= _options.Max; i++)
                    result.Add(new LongParameterValue(_options.Name, i));
            }
            else
            {
                if (_options.LogBase)
                {
                    // REVIEW: review the math below, it only works for positive Min and Max
                    var logBase = _options.StepSize ?? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1));

                    long prevValue = long.MinValue;
                    var maxPlusEpsilon = _options.Max * Math.Sqrt(logBase);
                    for (Double value = _options.Min; value <= maxPlusEpsilon; value *= logBase)
                    {
                        var longValue = (long)value;
                        if (longValue > prevValue)
                            result.Add(new LongParameterValue(_options.Name, longValue));
                        prevValue = longValue;
                    }
                }
                else
                {
                    var stepSize = _options.StepSize ?? (Double)(_options.Max - _options.Min) / (_options.NumSteps - 1);
                    long prevValue = long.MinValue;
                    var maxPlusEpsilon = _options.Max + stepSize / 2;
                    for (Double value = _options.Min; value <= maxPlusEpsilon; value += stepSize)
                    {
                        var longValue = (long)value;
                        if (longValue > prevValue)
                            result.Add(new LongParameterValue(_options.Name, longValue));
                        prevValue = longValue;
                    }
                }
            }

            _gridValues = result.ToArray();
        }

        public IParameterValue this[int i]
        {
            get
            {
                EnsureParameterValues();
                return _gridValues[i];
            }
        }

        public int Count
        {
            get
            {
                EnsureParameterValues();
                return _gridValues.Length;
            }
        }

        public float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;

            if (_options.LogBase)
            {
                float logBase = (float)(_options.StepSize ?? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1)));
                return (float)((Math.Log(valueTyped.Value, logBase) - Math.Log(_options.Min, logBase)) / (Math.Log(_options.Max, logBase) - Math.Log(_options.Min, logBase)));
            }
            else
            {
                return (float)(valueTyped.Value - _options.Min) / (_options.Max - _options.Min);
            }
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            return (_options.Min <= valueTyped.Value && valueTyped.Value <= _options.Max);
        }
    }

    /// <summary>
    /// The floating point type parameter sweep.
    /// </summary>
    public class FloatValueGenerator : INumericValueGenerator
    {
        private readonly FloatParamOptions _options;
        private IParameterValue[] _gridValues;

        public string Name { get { return _options.Name; } }

        public FloatValueGenerator(FloatParamOptions options)
        {
            _options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            float val;
            if (_options.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !_options.StepSize.HasValue
                    ? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1))
                    : _options.StepSize.Value;
                var logMax = Math.Log(_options.Max, logBase);
                var logMin = Math.Log(_options.Min, logBase);
                val = (float)(_options.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
                val = (float)(_options.Min + normalizedValue * (_options.Max - _options.Min));

            return new FloatParameterValue(_options.Name, val);
        }

        private void EnsureParameterValues()
        {
            if (_gridValues != null)
                return;

            var result = new List<IParameterValue>();
            if (_options.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = _options.StepSize ?? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1));

                float prevValue = float.NegativeInfinity;
                var maxPlusEpsilon = _options.Max * Math.Sqrt(logBase);
                for (Double value = _options.Min; value <= maxPlusEpsilon; value *= logBase)
                {
                    var floatValue = (float)value;
                    if (floatValue > prevValue)
                    {
                        result.Add(new FloatParameterValue(_options.Name, floatValue));
                    }

                    prevValue = floatValue;
                }
            }
            else
            {
                var stepSize = _options.StepSize ?? (Double)(_options.Max - _options.Min) / (_options.NumSteps - 1);
                float prevValue = float.NegativeInfinity;
                var maxPlusEpsilon = _options.Max + stepSize / 2;
                for (Double value = _options.Min; value <= maxPlusEpsilon; value += stepSize)
                {
                    var floatValue = (float)value;
                    if (floatValue > prevValue)
                        result.Add(new FloatParameterValue(_options.Name, floatValue));
                    prevValue = floatValue;
                }
            }

            _gridValues = result.ToArray();
        }

        public IParameterValue this[int i]
        {
            get
            {
                EnsureParameterValues();
                return _gridValues[i];
            }
        }

        public int Count
        {
            get
            {
                EnsureParameterValues();
                return _gridValues.Length;
            }
        }

        public float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;

            if (_options.LogBase)
            {
                float logBase = (float)(_options.StepSize ?? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1)));
                return (float)((Math.Log(valueTyped.Value, logBase) - Math.Log(_options.Min, logBase)) / (Math.Log(_options.Max, logBase) - Math.Log(_options.Min, logBase)));
            }
            else
                return (valueTyped.Value - _options.Min) / (_options.Max - _options.Min);
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;
            return (_options.Min <= valueTyped.Value && valueTyped.Value <= _options.Max);
        }
    }

    /// <summary>
    /// The discrete parameter sweep.
    /// </summary>
    public class DiscreteValueGenerator : IValueGenerator
    {
        private readonly DiscreteParamOptions _options;

        public string Name { get { return _options.Name; } }

        public DiscreteValueGenerator(DiscreteParamOptions options)
        {
            _options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            return new StringParameterValue(_options.Name, _options.Values[(int)(_options.Values.Length * normalizedValue)]);
        }

        public IParameterValue this[int i]
        {
            get
            {
                return new StringParameterValue(_options.Name, _options.Values[i]);
            }
        }

        public int Count
        {
            get
            {
                return _options.Values.Length;
            }
        }
    }
}
