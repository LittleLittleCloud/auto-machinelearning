// <copyright file="Int32ValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.AutoPipeline;
using Microsoft.ML.Runtime;
using Microsoft.ML.Sweeper;

namespace Microsoft.ML.AutoPipeline.Paramaters
{
    internal class Int32ValueGenerator : INumericValueGenerator
    {
        private readonly IntParamOptions _options;
        private IParameterValue[] _gridValues;

        public string Name { get { return _options.Name; } }

        public Int32ValueGenerator(IntParamOptions options)
        {
            Contracts.Check(options.Min < options.Max, "min must be less than max");
            // REVIEW: this condition can be relaxed if we change the math below to deal with it
            Contracts.Check(!options.LogBase || options.Min > 0, "min must be positive if log scale is used");
            Contracts.Check(!options.LogBase || options.StepSize == null || options.StepSize > 1, "StepSize must be greater than 1 if log scale is used");
            Contracts.Check(options.LogBase || options.StepSize == null || options.StepSize > 0, "StepSize must be greater than 0 if linear scale is used");
            _options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            Int32 val;
            if (_options.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !_options.StepSize.HasValue
                    ? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1))
                    : _options.StepSize.Value;
                var logMax = Math.Log(_options.Max, logBase);
                var logMin = Math.Log(_options.Min, logBase);
                val = (Int32)(_options.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
                val = (Int32)(_options.Min + normalizedValue * (_options.Max - _options.Min));

            return new ParamaterValue()
            {
                Name = _options.Name,
                RawValue = val,
            };
        }

        private void EnsureParameterValues()
        {
            if (_gridValues != null)
                return;

            var result = new List<IParameterValue>();
            if ((_options.StepSize == null && _options.NumSteps > (_options.Max - _options.Min)) ||
                (_options.StepSize != null && _options.StepSize <= 1))
            {
                for (Int32 i = _options.Min; i <= _options.Max; i++)
                    result.Add(new ParamaterValue() { Name = _options.Name, RawValue = i});
            }
            else
            {
                if (_options.LogBase)
                {
                    // REVIEW: review the math below, it only works for positive Min and Max
                    var logBase = _options.StepSize ?? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1));

                    Int32 prevValue = Int32.MinValue;
                    var maxPlusEpsilon = _options.Max * Math.Sqrt(logBase);
                    for (Double value = _options.Min; value <= maxPlusEpsilon; value *= logBase)
                    {
                        var int32Value = (Int32)value;
                        if (int32Value > prevValue)
                            result.Add(new ParamaterValue() { Name = _options.Name, RawValue = int32Value });
                        prevValue = int32Value;
                    }
                }
                else
                {
                    var stepSize = _options.StepSize ?? (Double)(_options.Max - _options.Min) / (_options.NumSteps - 1);
                    long prevValue = Int32.MinValue;
                    var maxPlusEpsilon = _options.Max + stepSize / 2;
                    for (Double value = _options.Min; value <= maxPlusEpsilon; value += stepSize)
                    {
                        var int32Value = (Int32)value;
                        if (int32Value > prevValue)
                            result.Add(new ParamaterValue() { Name = _options.Name, RawValue = int32Value });
                        prevValue = int32Value;
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
            var valueTyped = value as ParamaterValue;
            Contracts.Check(valueTyped != null, "Int32ValueGenerator could not normalized parameter because it is not of the correct type");
            Contracts.Check(_options.Min <= (Int32)valueTyped.RawValue && (Int32)valueTyped.RawValue <= _options.Max, "Value not in correct range");

            if (_options.LogBase)
            {
                float logBase = (float)(_options.StepSize ?? Math.Pow(1.0 * _options.Max / _options.Min, 1.0 / (_options.NumSteps - 1)));
                return (float)((Math.Log((Int32)valueTyped.RawValue, logBase) - Math.Log(_options.Min, logBase)) / (Math.Log(_options.Max, logBase) - Math.Log(_options.Min, logBase)));
            }
            else
                return (float)((Int32)valueTyped.RawValue - _options.Min) / (_options.Max - _options.Min);
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            Contracts.Check(valueTyped != null, "Parameter should be of type LongParameterValue");
            return (_options.Min <= valueTyped.Value && valueTyped.Value <= _options.Max);
        }

        public string ToStringParameter(IHostEnvironment env)
        {
            throw new NotImplementedException();
        }
    }

    public class IntParamOptions : NumericParamOptions
    {
        public Int32 Min;

        public Int32 Max;
    }
}
