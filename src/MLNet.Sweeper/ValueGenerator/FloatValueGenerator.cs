// <copyright file="FloatValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Sweeper
{
    /// <summary>
    /// The floating point type parameter sweep.
    /// </summary>
    public class FloatValueGenerator : INumericValueGenerator
    {
        private readonly Option _options;

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

        public FloatValueGenerator(Option options)
        {
            this._options = options;
            this.ID = Guid.NewGuid().ToString("N");
        }

        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            var val = Utils.AXPlusB(this._options.Min, this._options.Max, normalizedValue, this._options.LogBase);

            return new FloatParameterValue(this._options.Name, (float)val, this.ID);
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

        public string ID { get; private set; }

        public float NormalizeValue(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;

            if (this._options.LogBase)
            {
                return (float)((Math.Log(valueTyped.Value) - Math.Log(this._options.Min)) / (Math.Log(this._options.Max) - Math.Log(this._options.Min)));
            }
            else
            {
                return (valueTyped.Value - this._options.Min) / (this._options.Max - this._options.Min);
            }
        }

        public bool InRange(IParameterValue value)
        {
            var valueTyped = value as FloatParameterValue;
            return this._options.Min <= valueTyped.Value && valueTyped.Value <= this._options.Max;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Parameter Name: {this.Name}");
            sb.AppendLine($"Parameter Type: float");
            sb.AppendLine($"Min Value: {this._options.Min}");
            sb.AppendLine($"Max Value: {this._options.Max}");
            sb.AppendLine($"Steps: {this._options.Steps}");
            sb.AppendLine($"Log Base: {this._options.LogBase}");

            return sb.ToString();
        }

        public IParameterValue CreateFromString(string valueText)
        {
            var value = float.Parse(valueText);
            if (value < this._options.Min || value > this._options.Max)
            {
                throw new Exception($"{valueText} is out of range.");
            }

            return new FloatParameterValue(this._options.Name, value, this.ID);
        }

        public class Option : NumericValueGeneratorOptionBase
        {
            public float Min;

            public float Max;
        }
    }
}
