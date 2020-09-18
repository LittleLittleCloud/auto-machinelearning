// <copyright file="DiscreteValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.Sweeper
{
    /// <summary>
    /// The discrete parameter sweep.
    /// </summary>
    public class DiscreteValueGenerator<TValue> : IDiscreteValueGenerator
    {
        private readonly Option<TValue> _options;

        public string Name
        {
            get
            {
                return this._options.Name;
            }

            set
            {
                this._options.Name = value;
            }
        }

        public DiscreteValueGenerator(Option<TValue> options)
        {
            this._options = options;
            this.ID = Guid.NewGuid().ToString("N");
        }

        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            var rawValue = this._options.Values[(int)(this._options.Values.Length * normalizedValue)];
            var value = Utils.CreateObjectParameterValue(this._options.Name, rawValue, this.OneHotEncodeValue(rawValue), this.ID);
            return value;
        }

        public double[] OneHotEncodeValue(IParameterValue value)
        {
            if (!(value is ObjectParameterValue<TValue>))
            {
                throw new Exception($"can't find value {value}");
            }

            return this.OneHotEncodeValue((value as ObjectParameterValue<TValue>).Value);
        }

        private double[] OneHotEncodeValue(TValue rawValue)
        {
            var index = Array.FindIndex(this._options.Values, (TValue val) => val.Equals(rawValue));
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

        public TValue[] Values
        {
            get => this._options.Values;
        }

        public IParameterValue this[int i] => Utils.CreateObjectParameterValue(this._options.Name, this._options.Values[i], this.OneHotEncodeValue(this._options.Values[i]), this.ID);

        public int Count => this._options.Values.Length;

        [JsonIgnore]
        public string ID { get; private set; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Parameter Name: {this.Name}");
            sb.AppendLine($"Parameter Type: Discrete");
            sb.AppendLine($"Parameter Value: {string.Join(",", this._options.Values.Select(x => x.ToString()))}");

            return sb.ToString();
        }

        public IParameterValue CreateFromString(string valueText)
        {
            var value = this.Values.Where(val => valueText == val.ToString()).FirstOrDefault();

            if (value == null)
            {
                throw new Exception($"can't find {valueText}");
            }

            return Utils.CreateObjectParameterValue(this._options.Name, value, this.OneHotEncodeValue(value), this.ID);
        }

        public class Option<TValue> : ValueGeneratorOptionBase
        {
            public TValue[] Values = null;
        }
    }
}
