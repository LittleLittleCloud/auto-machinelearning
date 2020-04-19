// <copyright file="Int32ValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using MLNet.Sweeper;

namespace MLNet.Sweeper
{
    public class Int32ValueGenerator : INumericValueGenerator
    {
        private readonly Option _options;

        public string Name => this._options.Name;

        public Int32ValueGenerator(Option options)
        {
            this._options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            var val = Utils.AXPlusB(this._options.Min, this._options.Max, normalizedValue, this._options.LogBase);

            return new Int32ParamaterValue(this._options.Name, Convert.ToInt32(val));
        }

        public IParameterValue this[int i]
        {
            get
            {
                return this.CreateFromNormalized(i * 1.0 / this._options.Steps);
            }
        }

        public int Count
        {
            get
            {
                return this._options.Steps + 1;
            }
        }

        public float NormalizeValue(IParameterValue value)
        {
            if (this._options.LogBase)
            {
                return (float)((Math.Log((int)value.RawValue) - Math.Log(this._options.Min)) / (Math.Log(this._options.Max) - Math.Log(this._options.Min)));
            }
            else
            {
                return (float)((int)value.RawValue - this._options.Min) / (this._options.Max - this._options.Min);
            }
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as LongParameterValue;
            return this._options.Min <= valueTyped.Value && valueTyped.Value <= this._options.Max;
        }

        public class Option : NumericValueGeneratorOptionBase
        {
            public int Min;

            public int Max;
        }
    }
}
