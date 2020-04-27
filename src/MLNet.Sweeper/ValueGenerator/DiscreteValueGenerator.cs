// <copyright file="DiscreteValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.Sweeper
{
    /// <summary>
    /// The discrete parameter sweep.
    /// </summary>
    public class DiscreteValueGenerator : IDiscreteValueGenerator
    {
        private readonly Option _options;

        public string Name => this._options.Name;

        public DiscreteValueGenerator(Option options)
        {
            this._options = options;
            this.ID = Guid.NewGuid().ToString("N");
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            var rawValue = this._options.Values[(int)(this._options.Values.Length * normalizedValue)];
            var value = new DiscreteParameterValue(this._options.Name, rawValue, this.OneHotEncodeValue(rawValue), this.ID);
            return value;
        }

        public double[] OneHotEncodeValue(IParameterValue value)
        {
            return this.OneHotEncodeValue(value.RawValue);
        }

        private double[] OneHotEncodeValue(object rawValue)
        {
            var index = Array.FindIndex(this._options.Values, (object val) => val == rawValue);
            if (index < 0)
            {
                throw new Exception($"can't find value {rawValue}");
            }
            else
            {
                var onehot = Enumerable.Repeat(0.0, this.Count).ToArray();
                onehot[index] = 1;
                return onehot;
            }
        }

        public IParameterValue this[int i] => new DiscreteParameterValue(this._options.Name, this._options.Values[i], this.OneHotEncodeValue(this._options.Values[i]));

        public int Count => this._options.Values.Length;

        public string ID { get; private set; }

        public class Option : ValueGeneratorOptionBase
        {
            public object[] Values = null;
        }
    }
}
