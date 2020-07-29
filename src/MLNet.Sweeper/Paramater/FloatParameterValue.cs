// <copyright file="FloatParameterValue.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

namespace MLNet.Sweeper
{
    public sealed class FloatParameterValue : IParameterValue<float>
    {
        public string Name { get; private set; }

        public string ValueText { get; private set; }

        public float Value { get; private set; }

        public object RawValue { get; private set; }

        public string ID { get; private set; }

        public FloatParameterValue(string name, float value, string id = null)
        {
            this.Name = name;
            this.Value = value;
            this.RawValue = value;
            this.ValueText = this.Value.ToString("R");
            this.ID = id ?? name;
        }

        public bool Equals(IParameterValue other)
        {
            return this.Equals((object)other);
        }

        public override bool Equals(object obj)
        {
            var fpv = obj as FloatParameterValue;
            return fpv != null && this.Name == fpv.Name && this.Value == fpv.Value;
        }
    }
}
