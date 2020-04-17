using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.AutoPipeline
{
    internal class ParamaterValue : IParameterValue
    {
        public string Name { get; set; }

        public string ValueText { get => RawValue.ToString(); }

        public object RawValue { get; set; }

        public bool Equals(IParameterValue other)
        {
            return RawValue != null && Name == other.Name && ValueText == other.ValueText;
        }
    }
}
