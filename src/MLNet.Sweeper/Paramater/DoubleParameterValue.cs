// <copyright file="DoubleParameterValue.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace MLNet.Sweeper
{
    [Serializable]
    public sealed class DoubleParameterValue : IParameterValue<double>
    {
        public string Name { get; private set; }

        public string ValueText { get; private set; }

        public double Value { get; private set; }

        public object RawValue { get => this.Value; }

        public string ID { get; private set; }

        public DoubleParameterValue(string name, double value, string id = null)
        {
            this.Name = name;
            this.Value = value;
            this.ValueText = this.Value.ToString("R");
            this.ID = id ?? name;
        }

        public bool Equals(IParameterValue other)
        {
            return this.Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var fpv = obj as DoubleParameterValue;
            return fpv != null && this.Name == fpv.Name && this.Value == fpv.Value;
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            throw new NotImplementedException();
        }
    }
}
