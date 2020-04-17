// <copyright file="ParamaterValue.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    internal class ParamaterValue : IParameterValue
    {
        public string Name { get; set; }

        public string ValueText => this.RawValue.ToString();

        public object RawValue { get; set; }

        public bool Equals(IParameterValue other)
        {
            return this.RawValue != null && this.Name == other.Name && this.ValueText == other.ValueText;
        }
    }
}
