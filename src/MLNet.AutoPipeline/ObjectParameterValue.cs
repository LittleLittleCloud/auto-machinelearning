// <copyright file="Class1.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
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
    }
}
