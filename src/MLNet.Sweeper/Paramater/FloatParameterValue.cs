// <copyright file="FloatParameterValue.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;

namespace MLNet.Sweeper
{

    public sealed class FloatParameterValue : IParameterValue<float>
    {
        private readonly string _name;
        private readonly string _valueText;
        private readonly float _value;

        public string Name => this._name;

        public string ValueText => this._valueText;

        public float Value => this._value;

        public object RawValue => this._value;

        public FloatParameterValue(string name, float value)
        {
            this._name = name;
            this._value = value;
            this._valueText = this._value.ToString("R");
        }

        public bool Equals(IParameterValue other)
        {
            return this.Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var fpv = obj as FloatParameterValue;
            return fpv != null && this.Name == fpv.Name && this._value == fpv._value;
        }
    }
}
