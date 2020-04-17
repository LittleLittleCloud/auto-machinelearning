// <copyright file="Int32ValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Runtime;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;

namespace MLNet.AutoPipeline
{
    internal class Int32ValueGenerator : INumericValueGenerator
    {
        private readonly IntParamOptions _options;
        private IParameterValue[] _gridValues;

        public string Name => this._options.Name;

        public Int32ValueGenerator(IntParamOptions options)
        {
            this._options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(Double normalizedValue)
        {
            Int32 val;
            if (this._options.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !this._options.StepSize.HasValue
                    ? Math.Pow(1.0 * this._options.Max / this._options.Min, 1.0 / (this._options.NumSteps - 1))
                    : this._options.StepSize.Value;
                var logMax = Math.Log(this._options.Max, logBase);
                var logMin = Math.Log(this._options.Min, logBase);
                val = (Int32)(this._options.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
            {
                val = (Int32)(this._options.Min + normalizedValue * (this._options.Max - this._options.Min));
            }

            return new ParamaterValue()
            {
                Name = this._options.Name,
                RawValue = val,
            };
        }

        private void EnsureParameterValues()
        {
            if (this._gridValues != null)
            {
                return;
            }

            var result = new List<IParameterValue>();
            if ((this._options.StepSize == null && this._options.NumSteps > (this._options.Max - this._options.Min)) ||
                (this._options.StepSize != null && this._options.StepSize <= 1))
            {
                for (Int32 i = this._options.Min; i <= this._options.Max; i++)
                {
                    result.Add(new ParamaterValue() { Name = this._options.Name, RawValue = i });
                }
            }
            else
            {
                if (this._options.LogBase)
                {
                    // REVIEW: review the math below, it only works for positive Min and Max
                    var logBase = this._options.StepSize ?? Math.Pow(1.0 * this._options.Max / this._options.Min, 1.0 / (this._options.NumSteps - 1));

                    Int32 prevValue = Int32.MinValue;
                    var maxPlusEpsilon = this._options.Max * Math.Sqrt(logBase);
                    for (Double value = this._options.Min; value <= maxPlusEpsilon; value *= logBase)
                    {
                        var int32Value = (Int32)value;
                        if (int32Value > prevValue)
                        {
                            result.Add(new ParamaterValue() { Name = this._options.Name, RawValue = int32Value });
                        }

                        prevValue = int32Value;
                    }
                }
                else
                {
                    var stepSize = this._options.StepSize ?? (Double)(this._options.Max - this._options.Min) / (this._options.NumSteps - 1);
                    long prevValue = Int32.MinValue;
                    var maxPlusEpsilon = this._options.Max + stepSize / 2;
                    for (Double value = this._options.Min; value <= maxPlusEpsilon; value += stepSize)
                    {
                        var int32Value = (Int32)value;
                        if (int32Value > prevValue)
                        {
                            result.Add(new ParamaterValue() { Name = this._options.Name, RawValue = int32Value });
                        }

                        prevValue = int32Value;
                    }
                }
            }
            this._gridValues = result.ToArray();
        }

        public IParameterValue this[int i]
        {
            get
            {
                this.EnsureParameterValues();
                return this._gridValues[i];
            }
        }

        public int Count
        {
            get
            {
                this.EnsureParameterValues();
                return this._gridValues.Length;
            }
        }

        public float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as ParamaterValue;

            if (this._options.LogBase)
            {
                float logBase = (float)(this._options.StepSize ?? Math.Pow(1.0 * this._options.Max / this._options.Min, 1.0 / (this._options.NumSteps - 1)));
                return (float)((Math.Log((Int32)valueTyped.RawValue, logBase) - Math.Log(this._options.Min, logBase)) / (Math.Log(this._options.Max, logBase) - Math.Log(this._options.Min, logBase)));
            }
            else
            {
                return (float)((Int32)valueTyped.RawValue - this._options.Min) / (this._options.Max - this._options.Min);
            }
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            return (this._options.Min <= valueTyped.Value && valueTyped.Value <= this._options.Max);
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
