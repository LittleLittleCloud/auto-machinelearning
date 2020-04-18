// <copyright file="DiscreteValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;

namespace MLNet.Sweeper
{
    /// <summary>
    /// The discrete parameter sweep.
    /// </summary>
    public class DiscreteValueGenerator : IValueGenerator
    {
        private readonly Option _options;

        public string Name => this._options.Name;

        public DiscreteValueGenerator(Option options)
        {
            this._options = options;
        }

        // REVIEW: Is float accurate enough?
        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            return new StringParameterValue(this._options.Name, this._options.Values[(int)(this._options.Values.Length * normalizedValue)]);
        }

        public IParameterValue this[int i] => new StringParameterValue(this._options.Name, this._options.Values[i]);

        public int Count => this._options.Values.Length;

        public class Option : ValueGeneratorOptionBase
        {
            public string[] Values = null;
        }
    }
}
