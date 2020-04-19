// <copyright file="LongParameterValue - Copy.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;

namespace MLNet.Sweeper
{
    public sealed class Int32ParamaterValue : IParameterValue<int>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly int _value;

        public string Name => this._name;

        public string ValueText => this._valueText;

        public int Value => this._value;

        public object RawValue => this._value;

        public Int32ParamaterValue(string name, int value)
        {
            this._name = name;
            this._value = value;
            this._valueText = this._value.ToString("D");
        }

        public bool Equals(IParameterValue other)
        {
            return this.Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var lpv = obj as Int32ParamaterValue;
            return lpv != null && this.Name == lpv.Name && this._value == lpv._value;
        }
    }
}
