// <copyright file="ObjectParameterValue.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Sweeper
{
    internal class ObjectParameterValue : IParameterValue
    {
        public ObjectParameterValue(string name, object value, string id = null)
        {
            this.Name = name;
            this.RawValue = value;
            this.ID = id ?? name;
        }

        public string Name { get; private set; }

        public string ValueText { get => this.RawValue?.ToString(); }

        public object RawValue { get; private set; }

        public string ID { get; private set; }

        public bool Equals(IParameterValue other)
        {
            return other != null && other.ID == this.ID && other.Name == this.Name && other.RawValue == this.RawValue;
        }

        public override int GetHashCode()
        {
            int hashCode = 1120140629;
            hashCode = hashCode * -1521134295 + EqualityComparer<string>.Default.GetHashCode(this.Name);
            hashCode = hashCode * -1521134295 + EqualityComparer<string>.Default.GetHashCode(this.ValueText);
            hashCode = hashCode * -1521134295 + EqualityComparer<object>.Default.GetHashCode(this.RawValue);
            hashCode = hashCode * -1521134295 + EqualityComparer<string>.Default.GetHashCode(this.ID);
            return hashCode;
        }
    }
}
