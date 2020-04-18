// <copyright file="Int32ValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    internal class Int32ValueGenerator : INumericValueGenerator
    {
        private readonly Option _options;
        private IParameterValue[] _gridValues;

        public string Name => this._options.Name;

        public Int32ValueGenerator(Option options)
        {
            this._options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            int val;
            if (this._options.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !this._options.StepSize.HasValue
                    ? Math.Pow(1.0 * this._options.Max / this._options.Min, 1.0 / (this._options.NumSteps - 1))
                    : this._options.StepSize.Value;
                var logMax = Math.Log(this._options.Max, logBase);
                var logMin = Math.Log(this._options.Min, logBase);
                val = (int)(this._options.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
            {
                val = (int)(this._options.Min + normalizedValue * (this._options.Max - this._options.Min));
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
                for (int i = this._options.Min; i <= this._options.Max; i++)
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

                    int prevValue = int.MinValue;
                    var maxPlusEpsilon = this._options.Max * Math.Sqrt(logBase);
                    for (double value = this._options.Min; value <= maxPlusEpsilon; value *= logBase)
                    {
                        var intValue = (int)value;
                        if (intValue > prevValue)
                        {
                            result.Add(new ParamaterValue() { Name = this._options.Name, RawValue = intValue });
                        }

                        prevValue = intValue;
                    }
                }
                else
                {
                    var stepSize = this._options.StepSize ?? (double)(this._options.Max - this._options.Min) / (this._options.NumSteps - 1);
                    long prevValue = int.MinValue;
                    var maxPlusEpsilon = this._options.Max + stepSize / 2;
                    for (double value = this._options.Min; value <= maxPlusEpsilon; value += stepSize)
                    {
                        var intValue = (int)value;
                        if (intValue > prevValue)
                        {
                            result.Add(new ParamaterValue() { Name = this._options.Name, RawValue = intValue });
                        }

                        prevValue = intValue;
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
                return (float)((Math.Log((int)valueTyped.RawValue, logBase) - Math.Log(this._options.Min, logBase)) / (Math.Log(this._options.Max, logBase) - Math.Log(this._options.Min, logBase)));
            }
            else
            {
                return (float)((int)valueTyped.RawValue - this._options.Min) / (this._options.Max - this._options.Min);
            }
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            return this._options.Min <= valueTyped.Value && valueTyped.Value <= this._options.Max;
        }

        public string ToStringParameter(IHostEnvironment env)
        {
            throw new NotImplementedException();
        }

        public class Option : NumericValueGeneratorOptionBase
        {
            public int Min;

            public int Max;
        }
    }
}
