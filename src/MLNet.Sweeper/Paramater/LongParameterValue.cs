// <copyright file="LongParameterValue.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;

namespace MLNet.Sweeper
{
    public sealed class LongParameterValue : IParameterValue<long>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly long _value;

        public string Name => this._name;

        public string ValueText => this._valueText;

        public long Value => this._value;

        public object RawValue => this._value;

        public string ID { get; private set; }

        public LongParameterValue(string name, long value, string id = null)
        {
            this._name = name;
            this._value = value;
            this._valueText = this._value.ToString("D");
            this.ID = id ?? name;
        }

        public bool Equals(IParameterValue other)
        {
            return this.Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var lpv = obj as LongParameterValue;
            return lpv != null && this.Name == lpv.Name && this._value == lpv._value;
        }
    }
}
