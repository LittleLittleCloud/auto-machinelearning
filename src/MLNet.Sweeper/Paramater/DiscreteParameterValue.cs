// <copyright file="StringParameterValue.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace MLNet.Sweeper
{
    public sealed class DiscreteParameterValue : IDiscreteParameterValue
    {
        private readonly string _name;
        private readonly object _value;

        public string Name => this._name;

        public string ValueText => this._value.ToString();

        public object Value => this._value;

        public object RawValue => this._value;

        public double[] OneHotEncode { get; set; }

        public string ID { get; private set; }

        public DiscreteParameterValue(string name, object value, double[] onehot = null, string id = null)
        {
            this._name = name;
            this._value = value;
            this.ID = id ?? name;
            this.OneHotEncode = onehot;
        }

        public bool Equals(IParameterValue other)
        {
            return this.Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var spv = obj as DiscreteParameterValue;
            return spv != null && this.Name == spv.Name && this.ValueText == spv.ValueText;
        }

        public override int GetHashCode()
        {
            int hashCode = -1927337353;
            hashCode = hashCode * -1521134295 + EqualityComparer<string>.Default.GetHashCode(this.Name);
            hashCode = hashCode * -1521134295 + EqualityComparer<string>.Default.GetHashCode(this.ValueText);
            hashCode = hashCode * -1521134295 + EqualityComparer<object>.Default.GetHashCode(this.Value);
            hashCode = hashCode * -1521134295 + EqualityComparer<object>.Default.GetHashCode(this.RawValue);
            hashCode = hashCode * -1521134295 + EqualityComparer<string>.Default.GetHashCode(this.ID);
            return hashCode;
        }
    }
}
