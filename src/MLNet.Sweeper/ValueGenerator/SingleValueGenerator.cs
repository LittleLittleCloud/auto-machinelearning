// <copyright file="SingleValueGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Sweeper
{
    /// <summary>
    /// value generator that only has one value.
    /// </summary>
    public class SingleValueGenerator<T> : IValueGenerator
    {
        private ObjectParameterValue value;

        public SingleValueGenerator(string name, T value)
        {
            this.ID = Guid.NewGuid().ToString("N");
            this.value = new ObjectParameterValue(name, value, this.ID);
        }

        public IParameterValue this[int i]
        {
            get
            {
                if (i < 0 || i >= this.Count)
                {
                    throw new IndexOutOfRangeException();
                }

                return this.value;
            }
        }

        public int Count => 1;

        public string Name
        {
            get => this.value.Name;
            set => this.value = new ObjectParameterValue(value, this.value.RawValue, this.ID);
        }

        public string ID { get; private set; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Parameter Name: {this.Name}");
            sb.AppendLine($"Parameter Type: {typeof(T).Name}");
            sb.AppendLine($"Parameter Value: {this.value.RawValue}");

            return sb.ToString();
        }

        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            return this[0];
        }
    }
}
