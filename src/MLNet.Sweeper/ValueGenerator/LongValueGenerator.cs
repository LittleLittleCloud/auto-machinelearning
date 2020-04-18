// <copyright file="LongValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;

namespace MLNet.Sweeper
{
    /// <summary>
    /// The integer type parameter sweep.
    /// </summary>
    public class LongValueGenerator : INumericValueGenerator
    {
        private readonly Option _options;
        private IParameterValue[] _gridValues;

        public string Name => this._options.Name;

        public LongValueGenerator(Option options)
        {
            this._options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            long val;
            if (this._options.LogBase)
            {
                // REVIEW: review the math below, it only works for positive Min and Max
                var logBase = !this._options.StepSize.HasValue
                    ? Math.Pow(1.0 * this._options.Max / this._options.Min, 1.0 / (this._options.NumSteps - 1))
                    : this._options.StepSize.Value;
                var logMax = Math.Log(this._options.Max, logBase);
                var logMin = Math.Log(this._options.Min, logBase);
                val = (long)(this._options.Min * Math.Pow(logBase, normalizedValue * (logMax - logMin)));
            }
            else
            {
                val = (long)(this._options.Min + normalizedValue * (this._options.Max - this._options.Min));
            }

            return new LongParameterValue(this._options.Name, val);
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
                for (long i = this._options.Min; i <= this._options.Max; i++)
                {
                    result.Add(new LongParameterValue(this._options.Name, i));
                }
            }
            else
            {
                if (this._options.LogBase)
                {
                    // REVIEW: review the math below, it only works for positive Min and Max
                    var logBase = this._options.StepSize ?? Math.Pow(1.0 * this._options.Max / this._options.Min, 1.0 / (this._options.NumSteps - 1));

                    long prevValue = long.MinValue;
                    var maxPlusEpsilon = this._options.Max * Math.Sqrt(logBase);
                    for (double value = this._options.Min; value <= maxPlusEpsilon; value *= logBase)
                    {
                        var longValue = (long)value;
                        if (longValue > prevValue)
                        {
                            result.Add(new LongParameterValue(this._options.Name, longValue));
                        }

                        prevValue = longValue;
                    }
                }
                else
                {
                    var stepSize = this._options.StepSize ?? (double)(this._options.Max - this._options.Min) / (this._options.NumSteps - 1);
                    long prevValue = long.MinValue;
                    var maxPlusEpsilon = this._options.Max + stepSize / 2;
                    for (double value = this._options.Min; value <= maxPlusEpsilon; value += stepSize)
                    {
                        var longValue = (long)value;
                        if (longValue > prevValue)
                        {
                            result.Add(new LongParameterValue(this._options.Name, longValue));
                        }

                        prevValue = longValue;
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
            var valueTyped = value as LongParameterValue;

            if (this._options.LogBase)
            {
                float logBase = (float)(this._options.StepSize ?? Math.Pow(1.0 * this._options.Max / this._options.Min, 1.0 / (this._options.NumSteps - 1)));
                return (float)((Math.Log(valueTyped.Value, logBase) - Math.Log(this._options.Min, logBase)) / (Math.Log(this._options.Max, logBase) - Math.Log(this._options.Min, logBase)));
            }
            else
            {
                return (float)(valueTyped.Value - this._options.Min) / (this._options.Max - this._options.Min);
            }
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            return this._options.Min <= valueTyped.Value && valueTyped.Value <= this._options.Max;
        }

        public class Option : NumericValueGeneratorOptionBase
        {
            public long Min;

            public long Max;
        }
    }
}
